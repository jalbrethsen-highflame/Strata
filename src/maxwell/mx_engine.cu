// src/maxwell/mx_engine.cu - strata-mx: the engine path for Maxwell-V1-Flash (GGUF architecture maxwell-moe).
//
// Maxwell is a 35B-A3B MoE: 45 layers of GQA (32 query heads, 4 KV heads, head 128, per-head QK RMS norm, NEOX RoPE
// over all 128 dims, base 2e7), 3 leading dense SwiGLU layers (FF 6144), then 42 MoE layers (256 experts, top 12 by
// sigmoid(logit) + bias, weights normalized; one shared expert FF 1024, no gate), and DeepSeek-V4 style mHC: 4
// residual streams, per sublayer a 24-way mix (pre 4 / post 4 / comb 4x4 Sinkhorn-normalized) and a learned
// collapse before the head.  Attention is the vendor's block-sparse GQA (MBSA, see kBlk below): dense up to 4096
// tokens (where every block is routed), past that 32 routed blocks of 128 per KV group, so a token's attention costs
// the same at any context length (STRATA_MX_DENSE=1: dense everywhere, as the llama.cpp port runs).
//
// Built from Strata's kernels: the dense Q8_0 matrices on the native MMVQ (decode) and the FP16 GEMM (prompt), the
// routed experts on the grouped native expert kernels (blobs [gate | up | down] per expert in VRAM slots), and the
// experts without a slot on the CPU expert pool (ggml-cpu's arithmetic), read in place from the mapped GGUF.
//
// Modes: --tokens IDS [--n N] [--force IDS] [--dump-full-logits F] (the parity check against tools/maxwell's
// llama.cpp dumps), --bench-pp N / --bench-tg N.  STRATA_MX_PROF=1: a per-phase time breakdown (syncs per phase).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <random>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace C = strata::kernels::cpu;
namespace MMQ = strata::prefill::mmq;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-mx: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);  \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kHC = 4;               // mHC streams
constexpr int kMix = (2 + kHC) * kHC; // 24: pre 4, post 4, comb 16
constexpr int kHD = 128;             // head dim
constexpr int kChunkCells = 256;     // decode attention split-K chunk
constexpr int kMaxK = 16;            // top-k cap
// MBSA (block-sparse GQA, the vendor's MaxwellMoeBlockSparseGQAAttention): per KV group a route of 8 prefix blocks,
// 16 remote blocks picked by the indexer (block score = the max token score in the block) and 8 local blocks, of 128
// tokens.  Every block is routed while the query's block is < 32, so below 4096 tokens it is dense attention.
constexpr int kBlk = 128, kPrefixBlk = 8, kLocalBlk = 8, kRemoteBlk = 16, kRoute = kPrefixBlk + kRemoteBlk + kLocalBlk;
constexpr int kIdxHeads = 4, kIdxDim = 128;
constexpr int kDenseUpTo = kRoute * kBlk;   // 4096: positions below take the dense path (the same attention)

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// The 4 streams of each row start as the token's embedding: X[t][s][i] = e[t][i].  grid1(T * N).
__global__ void k_hc_init(const float* e, float* X, int N, int64_t n) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int64_t t = j / N, i = j % N;
    const float v = e[j];
    for (int s = 0; s < kHC; ++s) X[(t * kHC + s) * N + i] = v;
}

/// mHC before a sublayer, one block per row: `raw` [T][24][nsplit] = partial sums of fn . flat(X[t]) (the 4 streams as one
/// 4N row; nsplit 1: the whole products), added in slice order.  mixes = raw / (rms(flat) + eps); pre = sigmoid(mix *
/// scale0 + base) + eps; post = 2 sigmoid(...); comb = Sinkhorn(mix * scale2 + base) as [dst + 4 src]: softmax over dst,
/// + eps, column norm, then (iters - 1) rounds of row and column norms (warp 0, one lane per cell).  h[t] = rms_norm(sum_s
/// pre[s] X[t][s]) * w (the sublayer's input norm).
__global__ void __launch_bounds__(1024) k_hc_pre(const float* X, const float* raw, int nsplit, const float* scale,
                                                  const float* base, const float* w, float* h, float* post, float* comb, int N,
                                                  int iters, float eps_hc, float eps) {
    __shared__ float sh[1024];
    __shared__ float m[kMix], pre[kHC];
    const int t = blockIdx.x;
    const float* xr = X + (size_t) t * kHC * N;
    float s = 0.f;
    for (int i = threadIdx.x; i < kHC * N; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float inv = 1.0f / (sqrtf(s / (float) (kHC * N)) + eps_hc);
    if (threadIdx.x < kMix) {
        const float* rp = raw + ((size_t) t * kMix + threadIdx.x) * nsplit;
        float v = rp[0];
        for (int k = 1; k < nsplit; ++k) v += rp[k];
        m[threadIdx.x] = v * inv;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        if (lane < kHC) {
            pre[lane] = sigm(m[lane] * scale[0] + base[lane]) + eps_hc;
            post[(size_t) t * kHC + lane] = 2.0f * sigm(m[kHC + lane] * scale[1] + base[kHC + lane]);
        }
        // the 4 x 4 cell of lane l (< 16): d = l % 4, src = l / 4, index d + 4 src = l; lanes 16..31 mirror 0..15
        const int l = lane & 15;
        float c = m[2 * kHC + l] * scale[2] + base[2 * kHC + l];
        // softmax over dst for each src: the 4 lanes of a src group (xor 1, 2)
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        float e = expf(c - mx);
        float sm = e + __shfl_xor_sync(0xffffffffu, e, 1);
        sm += __shfl_xor_sync(0xffffffffu, sm, 2);
        c = e / sm + eps_hc;
        auto col = [&](float v) {   // for each dst over src: lanes with the same l % 4 (xor 4, 8)
            float a = v + __shfl_xor_sync(0xffffffffu, v, 4);
            a += __shfl_xor_sync(0xffffffffu, a, 8);
            return v / (a + eps_hc);
        };
        auto row = [&](float v) {   // for each src over dst (xor 1, 2)
            float a = v + __shfl_xor_sync(0xffffffffu, v, 1);
            a += __shfl_xor_sync(0xffffffffu, a, 2);
            return v / (a + eps_hc);
        };
        c = col(c);
        for (int it = 1; it < iters; ++it) c = col(row(c));
        if (lane < 16) comb[(size_t) t * kHC * kHC + l] = c;
    }
    __syncthreads();
    float* hr = h + (size_t) t * N;
    float s2 = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = xr[i] * pre[0];
        for (int j = 1; j < kHC; ++j) v += xr[(size_t) j * N + i] * pre[j];
        hr[i] = v;
        s2 += v * v;
    }
    s2 = block_sum(s2, sh);
    const float r = rsqrtf(s2 / (float) N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) hr[i] = hr[i] * r * w[i];
}
/// Split-K f32 products for the mHC mappings of a few rows: part[t][o][z] = W[o][kz .. kz + K / nsplit] . x[t][..],
/// grid (n_out, nsplit, T), block 256.
__global__ void k_gemv_f32_split(const float* W, const float* x, float* part, int n_in, int n_out, int nsplit) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x, z = blockIdx.y, t = blockIdx.z;
    const int len = n_in / nsplit, k0 = z * len;
    const float* wr = W + (size_t) o * n_in + k0;
    const float* xr = x + (size_t) t * n_in + k0;
    float a = 0.f;
    for (int i = threadIdx.x; i < len; i += blockDim.x) a += wr[i] * xr[i];
    a = block_sum(a, sh);
    if (threadIdx.x == 0) part[((size_t) t * n_out + o) * nsplit + z] = a;
}

/// mHC after a sublayer: X[t][d][i] = post[d] f[t][i] + sum_s comb[d + 4 s] X[t][s][i] (in place).  grid1(T * N).
__global__ void k_hc_post(float* X, const float* f, const float* post, const float* comb, int N, int64_t n) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int64_t t = j / N, i = j % N;
    float* xr = X + (size_t) t * kHC * N + i;
    float r[kHC];
    for (int s = 0; s < kHC; ++s) r[s] = xr[(size_t) s * N];
    const float fv = f[j];
    const float* pp = post + t * kHC;
    const float* cc = comb + t * kHC * kHC;
    for (int d = 0; d < kHC; ++d) {
        float v = fv * pp[d];
        for (int s = 0; s < kHC; ++s) v += r[s] * cc[d + kHC * s];
        xr[(size_t) d * N] = v;
    }
}

/// The head's collapse, one block per row: raw [T][4] = fn . flat; mix = raw / sqrt(mean(flat^2) + eps); pre =
/// sigmoid(mix * scale + base) + eps; h = rms_norm(sum_s pre[s] X[s]) * w (the output norm).
__global__ void k_hc_head(const float* X, const float* raw, int nsplit, const float* scale, const float* base, const float* w,
                          float* h, int N, float eps_hc, float eps) {
    __shared__ float sh[kThreads];
    __shared__ float pre[kHC];
    const int t = blockIdx.x;
    const float* xr = X + (size_t) t * kHC * N;
    float s = 0.f;
    for (int i = threadIdx.x; i < kHC * N; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float inv = rsqrtf(s / (float) (kHC * N) + eps_hc);
    if (threadIdx.x < kHC) {
        const float* rp = raw + ((size_t) t * kHC + threadIdx.x) * nsplit;
        float v = rp[0];
        for (int k = 1; k < nsplit; ++k) v += rp[k];
        pre[threadIdx.x] = sigm(v * inv * scale[0] + base[threadIdx.x]) + eps_hc;
    }
    __syncthreads();
    float* hr = h + (size_t) t * N;
    float s2 = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = xr[i] * pre[0];
        for (int j = 1; j < kHC; ++j) v += xr[(size_t) j * N + i] * pre[j];
        hr[i] = v;
        s2 += v * v;
    }
    s2 = block_sum(s2, sh);
    const float r = rsqrtf(s2 / (float) N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) hr[i] = hr[i] * r * w[i];
}

/// The router, one warp per row (n_expert <= 256: 8 per lane): p = sigmoid(logit); the top K by p + bias (ties: lowest
/// id), in that order; weights p / (sum p + 1e-20).
__global__ void k_router(const float* logits, const float* bias, int n_expert, int K_, int32_t* ids, float* wts) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float p[8], sel[8];
    for (int i = 0; i < 8; ++i) {
        const int e = i * 32 + lane;
        p[i] = e < n_expert ? sigm(logits[(size_t) t * n_expert + e]) : 0.f;
        sel[i] = e < n_expert ? p[i] + bias[e] : -INFINITY;
    }
    float sum = 0.f, pk[kMaxK];
    for (int k = 0; k < K_; ++k) {
        float v = -INFINITY;
        int id = 0x7fffffff;
        for (int i = 0; i < 8; ++i)
            if (sel[i] > v) { v = sel[i]; id = i * 32 + lane; }   // ascending ids per lane: the first max is the lowest id
        for (int x = 16; x > 0; x >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, x);
            const int oi = __shfl_xor_sync(0xffffffffu, id, x);
            if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
        }
        if (id == 0x7fffffff) id = k;   // no finite score (a NaN row): any valid id, never an out-of-range read
        if ((id & 31) == lane) {
            sel[id >> 5] = -INFINITY;
            pk[k] = p[id >> 5];
        }
        pk[k] = __shfl_sync(0xffffffffu, pk[k], id & 31);
        if (lane == 0) ids[(size_t) t * K_ + k] = id;
    }
    if (lane == 0) {
        for (int k = 0; k < K_; ++k) sum += pk[k];
        sum += 1e-20f;   // the vendor's topk_scores / (sum + 1e-20)
        for (int k = 0; k < K_; ++k) wts[(size_t) t * K_ + k] = pk[k] / sum;
    }
}

/// f[t] = sum_k w_k part_k + shared[t] (the MoE block's output; no residual add: mHC's post mixes it in).
__global__ void k_combine(const float* parts, const int32_t* prow, const float* wts, int K_, const float* shared, float* f,
                          int N) {
    __shared__ const float* pk[kMaxK];
    const int t = blockIdx.x;
    if ((int) threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (prow: the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * N;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.y) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc += wr[k] * pk[k][i];
        f[(size_t) t * N + i] = acc + shared[(size_t) t * N + i];
    }
}

__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in % 128 == 0: a warp per row, 8 rows per block (grid (ceil(n_out
/// / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int mm = 16; mm > 0; mm >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, mm);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}

/// A layer's KV cache.  FP16 (k / v) or 8-bit (k8 / v8, one float scale per (kvh, cell): absmax / 127 over the head's
/// 128 values), [kvh][dstride][128] in VRAM.  --kv-host (8-bit): the VRAM region holds only the cells < `first` (the
/// prefix blocks and all of a context below 4096), a ring of the newest `ring` cells (the local blocks and recent routes)
/// sits beside it, and every cell is kept in pinned host memory (k8h ..., mapped, [kvh][hstride]): MBSA reads 32
/// blocks per KV group, so the other blocks stay in host memory and are read over PCIe when routed.
struct KVP {
    __half *k = nullptr, *v = nullptr;
    int8_t *k8 = nullptr, *v8 = nullptr;
    float *ks = nullptr, *vs = nullptr;
    int dstride = 0, first = 0x7fffffff;
    int8_t *k8r = nullptr, *v8r = nullptr;   // the ring (null: none)
    float *ksr = nullptr, *vsr = nullptr;
    int ring = 0;
    int8_t *k8h = nullptr, *v8h = nullptr;   // the host copy (null: none)
    float *ksh = nullptr, *vsh = nullptr;
    int hstride = 0;
};
constexpr int kRingMargin = 8;   // ring cells this close to the oldest one are read from the host (a rollback's rewrites)
/// Where 8-bit cell (g, pos) is read; `hi` = the newest cell written (the ring holds hi - ring + 1 .. hi).
__device__ __forceinline__ void kv_loc8(const KVP& kv, bool isv, int g, int pos, int hi, const int8_t*& qp, float& sc) {
    size_t r;
    if (!kv.k8h || pos < kv.first) {
        r = (size_t) g * kv.dstride + pos;
        qp = (isv ? kv.v8 : kv.k8) + r * kHD;
        sc = (isv ? kv.vs : kv.ks)[r];
    } else if (kv.k8r && pos > hi - kv.ring + kRingMargin) {
        r = (size_t) g * kv.ring + pos % kv.ring;
        qp = (isv ? kv.v8r : kv.k8r) + r * kHD;
        sc = (isv ? kv.vsr : kv.ksr)[r];
    } else {
        r = (size_t) g * kv.hstride + pos;
        qp = (isv ? kv.v8h : kv.k8h) + r * kHD;
        sc = (isv ? kv.vsh : kv.ksh)[r];
    }
}
/// 8 consecutive values (dims 8c .. 8c + 7) of cell (g, pos) of K (or V) as FP16 into dst.
__device__ __forceinline__ void kv_copy8(const KVP& kv, bool isv, int g, int pos, int hi, int c, __half* dst) {
    if (kv.k8) {
        const int8_t* src;
        float sc;
        kv_loc8(kv, isv, g, pos, hi, src, sc);
        const uint2 raw = *(const uint2*) (src + c * 8);
        const int8_t* bq = (const int8_t*) &raw;
        __align__(16) __half o[8];
        for (int i = 0; i < 8; ++i) o[i] = __float2half((float) bq[i] * sc);
        *(uint4*) dst = *(const uint4*) o;
    } else {
        *(uint4*) dst = *(const uint4*) ((isv ? kv.v : kv.k) + ((size_t) g * kv.dstride + pos) * kHD + c * 8);
    }
}
__device__ __forceinline__ float kv_get(const KVP& kv, bool isv, int g, int pos, int hi, int d) {
    if (kv.k8) {
        const int8_t* src;
        float sc;
        kv_loc8(kv, isv, g, pos, hi, src, sc);
        return (float) src[d] * sc;
    }
    return __half2float((isv ? kv.v : kv.k)[((size_t) g * kv.dstride + pos) * kHD + d]);
}
/// Cells [0, P) of each KV head as FP16 into k16 / v16 ([kvh][ctx][128]): the dense prompt path's cuBLAS operands.
/// grid1(HKV * P * 128).
__global__ void k_kv_deq(KVP kv, __half* k16, __half* v16, int ctx, int P, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * kHD) return;
    const int64_t per = (int64_t) P * kHD;
    const int g = (int) (i / per);
    const int64_t rest = i % per;
    const int pos = (int) (rest / kHD), d = (int) (rest % kHD);
    const size_t row = (size_t) g * ctx + pos;
    k16[row * kHD + d] = __float2half(kv_get(kv, false, g, pos, P - 1, d));
    v16[row * kHD + d] = __float2half(kv_get(kv, true, g, pos, P - 1, d));
}

/// The attention's per-row preparation, grid (H + HKV, T), block 128: q / k RMS-normed per head with their weights,
/// NEOX RoPE over all 128 dims (position *pos0 + t), k / v into the FP16 cache [kvh][ctx][128].
__global__ void k_attn_prep(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw, float* q,
                            KVP kv, int H, int HKV, const int32_t* pos0, int ctx, float base, float eps) {
    __shared__ float sh[kHD];
    __shared__ float buf[kHD];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = *pos0 + t;
    const bool isq = head < H;
    const float* src = isq ? qx + ((size_t) t * H + head) * kHD : kx + ((size_t) t * HKV + head - H) * kHD;
    const int i = threadIdx.x;
    const float v0 = src[i];
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) kHD + eps);
    buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    const int half = kHD / 2;
    const int p = i < half ? i : i - half;
    const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) kHD);
    float sn, cs;
    sincosf(theta, &sn, &cs);
    const float v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    if (isq) {
        q[((size_t) t * H + head) * kHD + i] = v;
    } else {
        const int kh = head - H;
        const float vv = vx[((size_t) t * HKV + kh) * kHD + i];
        (void) ctx;
        if (kv.k8) {   // per-cell absmax scales (the block is one head: uniform branch)
            const float ka = block_max(fabsf(v), sh), va = block_max(fabsf(vv), sh);
            const float ksc = ka > 0.f ? ka / 127.f : 1.f, vsc = va > 0.f ? va / 127.f : 1.f;
            const int8_t kq = (int8_t) __float2int_rn(v / ksc), vq = (int8_t) __float2int_rn(vv / vsc);
            auto put = [&](int8_t* k8, int8_t* v8, float* ks, float* vs, size_t r) {
                k8[r * kHD + i] = kq;
                v8[r * kHD + i] = vq;
                if (i == 0) { ks[r] = ksc; vs[r] = vsc; }
            };
            if (!kv.k8h || pos < kv.first) put(kv.k8, kv.v8, kv.ks, kv.vs, (size_t) kh * kv.dstride + pos);
            if (kv.k8h) put(kv.k8h, kv.v8h, kv.ksh, kv.vsh, (size_t) kh * kv.hstride + pos);
            if (kv.k8r) put(kv.k8r, kv.v8r, kv.ksr, kv.vsr, (size_t) kh * kv.ring + pos % kv.ring);
        } else {
            const size_t row = (size_t) kh * kv.dstride + pos;
            kv.k[row * kHD + i] = __float2half(v);
            kv.v[row * kHD + i] = __float2half(vv);
        }
    }
}

/// Split-K decode attention: one block per (256-cell chunk, query row); row = t * H + h sees cells [0, *pos0 + t].
__global__ void k_attn_part(const float* q, KVP kv, const int32_t* pos0, int t0, int H,
                            int HKV, int Ttot, float scale, int n_chunk, float* pacc, float* pm, float* pl) {
    __shared__ float qs[kHD], sc[kChunkCells], sh[kChunkCells];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    if (j < kHD) qs[j] = q[((size_t) t * H + h) * kHD + j];
    __syncthreads();
    const int p = p0 + j;
    const int hi = *pos0 + Ttot - 1;
    float s = -INFINITY;
    if (p < npos) {
        float a = 0.f;
        if (kv.k8) {
            const int8_t* kq;
            float ksc;
            kv_loc8(kv, false, kvh, p, hi, kq, ksc);
            const char2* kp = (const char2*) kq;
            for (int i = 0; i < kHD / 2; ++i) {
                const char2 c = kp[i];
                a += qs[2 * i] * (float) c.x + qs[2 * i + 1] * (float) c.y;
            }
            a *= ksc;
        } else {
            const __half2* kp = (const __half2*) (kv.k + ((size_t) kvh * kv.dstride + p) * kHD);
            for (int i = 0; i < kHD / 2; ++i) {
                const float2 k2 = __half22float2(kp[i]);
                a += qs[2 * i] * k2.x + qs[2 * i + 1] * k2.y;
            }
        }
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    const size_t o = (size_t) row * n_chunk + c;
    if (j < kHD) {
        float acc = 0.f;
        const size_t row0 = (size_t) kvh * kv.dstride + p0;
        if (kv.k8) {
            for (int k = 0; k < n; ++k) {
                const int8_t* vq;
                float vsc;
                kv_loc8(kv, true, kvh, p0 + k, hi, vq, vsc);
                acc += sc[k] * vsc * (float) vq[j];
            }
        } else {
            for (int k = 0; k < n; ++k) acc += sc[k] * __half2float(kv.v[(row0 + k) * kHD + j]);
        }
        pacc[o * kHD + j] = acc;
    }
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged: grid (n_rows), block 128.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const int32_t* pos0, int t0, int H,
                             int n_chunk, float* out) {
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = *pos0 + t + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * kHD + d];
        den += w * pl[o];
    }
    out[((size_t) t * H + h) * kHD + d] = num / den;
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> FP16 probabilities over [0, pos0 + t].
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}

/// The indexer's per-row preparation, grid (kIdxHeads + 1, T), block 128: x = [q heads 0..3 | k] (f32 [T][640]); RMS
/// norm with q_norm / k_norm, NEOX RoPE at the row's position (the main branch's), q -> iq [T][4][128] f32, k -> the
/// indexer K cache (FP16 [ctx][128]).
__global__ void k_index_prep(const float* x, const float* qw, const float* kw, float* iq, __half* ikc, const int32_t* pos0,
                             float base, float eps) {
    __shared__ float sh[kIdxDim];
    __shared__ float buf[kIdxDim];
    const int head = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = *pos0 + t;
    const bool isq = head < kIdxHeads;
    const float v0 = x[(size_t) t * (kIdxHeads + 1) * kIdxDim + head * kIdxDim + i];
    const float s = block_sum(v0 * v0, sh);
    buf[i] = v0 * rsqrtf(s / (float) kIdxDim + eps) * (isq ? qw[i] : kw[i]);
    __syncthreads();
    const int half = kIdxDim / 2;
    const int p = i < half ? i : i - half;
    const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) kIdxDim);
    float sn, cs;
    sincosf(theta, &sn, &cs);
    const float v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    if (isq) iq[((size_t) t * kIdxHeads + head) * kIdxDim + i] = v;
    else ikc[(size_t) pos * kIdxDim + i] = __float2half(v);
}
/// Block scores, grid (n_blocks, T * kIdxHeads), block 128: bs[t][g][b] = max over block b's tokens k <= pos of
/// (iq[t][g] . ik[k]) / sqrt(128), only for the eligible remote blocks (start >= the prefix window, end <= the local
/// window's start), else -inf.
__global__ void k_block_scores(const float* iq, const __half* ikc, const int32_t* pos0, int n_blocks, float* bs) {
    __shared__ float qs[kIdxDim], sh[kBlk];
    const int b = blockIdx.x, row = blockIdx.y, t = row / kIdxHeads, j = threadIdx.x;
    const int pos = *pos0 + t;
    const int cur = pos / kBlk;
    const int local_start = max(0, cur - kLocalBlk + 1);
    const bool elig = b >= kPrefixBlk && b + 1 <= local_start;
    if (!elig) {
        if (j == 0) bs[(size_t) row * n_blocks + b] = -INFINITY;
        return;
    }
    qs[j] = iq[(size_t) row * kIdxDim + j];
    __syncthreads();
    const int k = b * kBlk + j;   // every token of an eligible block is <= pos (the block ends before the local window)
    const __half2* kp = (const __half2*) (ikc + (size_t) k * kIdxDim);
    float a = 0.f;
    for (int i = 0; i < kIdxDim / 2; ++i) {
        const float2 kv = __half22float2(kp[i]);
        a += qs[2 * i] * kv.x + qs[2 * i + 1] * kv.y;
    }
    const float m = block_max(a * rsqrtf((float) kIdxDim), sh);
    if (j == 0) bs[(size_t) row * n_blocks + b] = m;
}
/// Block scores for a few decode rows, grid (n_blocks, T), block 128 (4 warps): a warp reads one key row of the
/// indexer cache per step, coalesced (32 lanes x 4 values), and scores all 4 index heads with it; bs[t][g][b] = the
/// block's max per head over eligible remote blocks (else -inf), as k_block_scores.
__global__ void k_block_scores4(const float* iq, const __half* ikc, const int32_t* pos0, int n_blocks, float* bs) {
    __shared__ float wmax[4][kIdxHeads];
    const int b = blockIdx.x, t = blockIdx.y, w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int pos = *pos0 + t;
    const int cur = pos / kBlk;
    const int local_start = max(0, cur - kLocalBlk + 1);
    const bool elig = b >= kPrefixBlk && b + 1 <= local_start;
    float* out = bs + (size_t) t * kIdxHeads * n_blocks + b;
    if (!elig) {
        if (threadIdx.x < kIdxHeads) out[(size_t) threadIdx.x * n_blocks] = -INFINITY;
        return;
    }
    float qv[kIdxHeads][4];
    for (int g = 0; g < kIdxHeads; ++g)
        for (int i = 0; i < 4; ++i) qv[g][i] = iq[((size_t) t * kIdxHeads + g) * kIdxDim + lane * 4 + i];
    float mx[kIdxHeads] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    const float sc = rsqrtf((float) kIdxDim);
    for (int k = w; k < kBlk; k += 4) {
        const uint2 raw = *(const uint2*) (ikc + ((size_t) b * kBlk + k) * kIdxDim + lane * 4);
        const __half2* hp = (const __half2*) &raw;
        const float2 a = __half22float2(hp[0]), c = __half22float2(hp[1]);
        float d[kIdxHeads];
        for (int g = 0; g < kIdxHeads; ++g) d[g] = qv[g][0] * a.x + qv[g][1] * a.y + qv[g][2] * c.x + qv[g][3] * c.y;
        for (int x = 16; x > 0; x >>= 1)
            for (int g = 0; g < kIdxHeads; ++g) d[g] += __shfl_xor_sync(0xffffffffu, d[g], x);
        for (int g = 0; g < kIdxHeads; ++g) mx[g] = fmaxf(mx[g], d[g] * sc);
    }
    if (lane == 0)
        for (int g = 0; g < kIdxHeads; ++g) wmax[w][g] = mx[g];
    __syncthreads();
    if (threadIdx.x < kIdxHeads) {
        const int g = threadIdx.x;
        out[(size_t) g * n_blocks] = fmaxf(fmaxf(wmax[0][g], wmax[1][g]), fmaxf(wmax[2][g], wmax[3][g]));
    }
}

/// Block scores from token scores (the prompt path): S [rows][ld] = iq . ik over keys [0, P) for rows r0 .. r0 + nr of
/// (row t, group g) = r / kIdxHeads; bs[r][b] = max over block b's tokens of S * scale for the eligible remote blocks,
/// else -inf.  grid (n_blocks, nr), block 128.
__global__ void k_block_max(const float* S, int ld, int r0, const int32_t* pos0, int n_blocks, float scale, float* bs) {
    __shared__ float sh[kBlk];
    const int b = blockIdx.x, rl = blockIdx.y, r = r0 + rl, t = r / kIdxHeads, j = threadIdx.x;
    const int pos = *pos0 + t;
    const int cur = pos / kBlk;
    const int local_start = max(0, cur - kLocalBlk + 1);
    const bool elig = b >= kPrefixBlk && b + 1 <= local_start;
    if (!elig) {
        if (j == 0) bs[(size_t) r * n_blocks + b] = -INFINITY;
        return;
    }
    const float m = block_max(S[(size_t) rl * ld + b * kBlk + j] * scale, sh);
    if (j == 0) bs[(size_t) r * n_blocks + b] = m;
}
__global__ void k_f32_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The route per (row, group), grid (T * kIdxHeads), block 256: [prefix 0..7 (<= cur) | the top 16 eligible blocks by
/// score, highest first, -1 past the finite ones | local cur-7 .. cur (>= 8)], -1 = empty.
__global__ void k_route_blocks(const float* bs, int n_blocks, const int32_t* pos0, int32_t* route) {
    __shared__ float sv[kThreads];
    __shared__ int si[kThreads];
    const int row = blockIdx.x, t = row / kIdxHeads, tid = threadIdx.x;
    const int pos = *pos0 + t;
    const int cur = pos / kBlk;
    int32_t* r = route + (size_t) row * kRoute;
    if (tid < kPrefixBlk) r[tid] = tid <= cur ? tid : -1;
    if (tid < kLocalBlk) {
        const int b = cur - (kLocalBlk - 1) + tid;
        r[kPrefixBlk + kRemoteBlk + tid] = b >= kPrefixBlk ? b : -1;
    }
    const float* sc = bs + (size_t) row * n_blocks;
    int last = -1;
    float lastv = INFINITY;
    for (int k = 0; k < kRemoteBlk; ++k) {
        // the best block strictly after (lastv, last) in the order (score desc, id asc)
        float v = -INFINITY;
        int id = 0x7fffffff;
        for (int b = tid; b < n_blocks; b += blockDim.x) {
            const float x = sc[b];
            const bool after = x < lastv || (x == lastv && b > last);
            if (after && (x > v || (x == v && b < id))) { v = x; id = b; }
        }
        sv[tid] = v;
        si[tid] = id;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) {
                const float ov = sv[tid + s2];
                const int oi = si[tid + s2];
                if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) { sv[tid] = ov; si[tid] = oi; }
            }
            __syncthreads();
        }
        lastv = sv[0];
        last = si[0];
        __syncthreads();
        if (tid == 0) r[kPrefixBlk + k] = (lastv > -INFINITY) ? last : -1;
    }
}
/// Sparse attention partials, grid (kRoute, T * HKV), block 256, dynamic smem = the K and V block (2 x 128 x 128 FP16):
/// for (row t, group g, route slot r) the block's keys k <= pos against the group's 8 query heads: per head the max,
/// the sum of exp and the exp-weighted V sum.
__global__ void k_sparse_part(const float* q, KVP kv, const int32_t* route, const int32_t* pos0,
                              int t0, int H, int HKV, int Ttot, float scale, float* pacc, float* pm, float* pl) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half* Ks = (__half*) smem;
    __half* Vs = Ks + kBlk * kHD;
    __shared__ float S[8][kBlk];   // heads per group <= 8
    __shared__ float qsh[8][kHD];
    const int r = blockIdx.x, row = blockIdx.y, t = t0 + row / HKV, g = row % HKV, tid = threadIdx.x;
    const int G = H / HKV;
    const int pos = *pos0 + t;
    const int b = route[((size_t) t * HKV + g) * kRoute + r];
    const size_t o0 = ((size_t) row * kRoute + r) * G;   // partial index of (row, slot, head 0)
    if (b < 0) {
        if (tid < G) { pm[o0 + tid] = -INFINITY; pl[o0 + tid] = 0.f; }
        return;
    }
    const int k0 = b * kBlk;
    const int nk = min(kBlk, pos + 1 - k0);   // keys <= pos
    {   // K / V block into shared memory (16-byte loads)
        const int n16 = nk * kHD / 8;
        const int hi = *pos0 + Ttot - 1;
        for (int i = tid; i < n16; i += blockDim.x) {
            kv_copy8(kv, false, g, k0 + i / (kHD / 8), hi, i % (kHD / 8), Ks + (size_t) i * 8);
            kv_copy8(kv, true, g, k0 + i / (kHD / 8), hi, i % (kHD / 8), Vs + (size_t) i * 8);
        }
        for (int i = tid; i < G * kHD; i += blockDim.x) qsh[i / kHD][i % kHD] = q[((size_t) t * H + g * G + i / kHD) * kHD + i % kHD];
    }
    __syncthreads();
    for (int i = tid; i < G * kBlk; i += blockDim.x) {   // scores
        const int hh = i / kBlk, k = i % kBlk;
        float a = -INFINITY;
        if (k < nk) {
            const __half2* kp = (const __half2*) (Ks + k * kHD);
            a = 0.f;
            for (int d = 0; d < kHD / 2; ++d) {
                const float2 kv = __half22float2(kp[d]);
                a += qsh[hh][2 * d] * kv.x + qsh[hh][2 * d + 1] * kv.y;
            }
            a *= scale;
        }
        S[hh][k] = a;
    }
    __syncthreads();
    // per head: max and exp (one warp per head, 8 warps)
    const int w = tid / 32, lane = tid % 32;
    if (w < G) {
        float m = -INFINITY;
        for (int k = lane; k < kBlk; k += 32) m = fmaxf(m, S[w][k]);
        for (int x = 16; x > 0; x >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, x));
        float l = 0.f;
        for (int k = lane; k < kBlk; k += 32) {
            const float e = k < nk ? expf(S[w][k] - m) : 0.f;
            S[w][k] = e;
            l += e;
        }
        for (int x = 16; x > 0; x >>= 1) l += __shfl_xor_sync(0xffffffffu, l, x);
        if (lane == 0) { pm[o0 + w] = m; pl[o0 + w] = l; }
    }
    __syncthreads();
    for (int i = tid; i < G * kHD; i += blockDim.x) {   // exp-weighted V sums
        const int hh = i / kHD, d = i % kHD;
        float acc = 0.f;
        for (int k = 0; k < nk; ++k) acc += S[hh][k] * __half2float(Vs[k * kHD + d]);
        pacc[(o0 + hh) * kHD + d] = acc;
    }
}
/// The route slots merged per (row, head): grid (T * H), block 128.
__global__ void k_sparse_merge(const float* pacc, const float* pm, const float* pl, int t0, int H, int HKV, float* out) {
    const int th = blockIdx.x, tl = th / H, t = t0 + tl, h = th % H, d = threadIdx.x;
    const int G = H / HKV, g = h / G, hh = h % G;
    const size_t base = ((size_t) (tl * HKV + g) * kRoute) * G + hh;
    float M = -INFINITY;
    for (int r = 0; r < kRoute; ++r) {
        const size_t o = base + (size_t) r * G;
        if (pl[o] > 0.f) M = fmaxf(M, pm[o]);
    }
    float num = 0.f, den = 0.f;
    for (int r = 0; r < kRoute; ++r) {
        const size_t o = base + (size_t) r * G;
        if (!(pl[o] > 0.f)) continue;
        const float wgt = expf(pm[o] - M);
        num += wgt * pacc[o * kHD + d];
        den += wgt * pl[o];
    }
    out[((size_t) t * H + h) * kHD + d] = num / den;
}
/// The prompt's block-sparse attention on tensor cores (FlashAttention over the union of the routes): grid (ceil(T /
/// 64), H), block 128 (4 warps, 16 query rows each).  Every block any of the tile's 64 rows routes (group h / G) is
/// visited once in ascending order, in two 64-key halves: S = Q K^T (WMMA, FP16 in, FP32 out), each row masked to its
/// own route and to keys <= its position, an online softmax, O += P V (WMMA).  The same attention as k_sparse_part +
/// k_sparse_merge per row (each row sees exactly its routed keys); dynamic smem kSpTileSmem.
namespace sp {
constexpr int QT = 64, KS = 64, LDH = kHD + 8, LDS = KS + 4, LDO = kHD + 4, LDP = KS + 8;
constexpr size_t kQ = (size_t) QT * LDH * 2, kK = (size_t) KS * LDH * 2, kSV = std::max((size_t) QT * LDS * 4, (size_t) KS * LDH * 2),
                 kO = (size_t) QT * LDO * 4;
constexpr size_t kSmem = kQ + kK + kSV + kO + QT * 4 * 2 + QT + 4096 / 8 * 1;
}
__global__ void __launch_bounds__(128) k_sparse_tile(const float* q, KVP kv, const int32_t* route,
                                                     const int32_t* pos0, int T, int H, int HKV, int ctx, int n_blocks,
                                                     float scale, float* out) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem[];
    __half* Qs = (__half*) smem;
    __half* Ks = (__half*) (smem + sp::kQ);                  // K half, then P (the probabilities, FP16)
    unsigned char* SV = smem + sp::kQ + sp::kK;              // S (FP32), then the V half
    float* Os = (float*) (SV + sp::kSV);
    float* mrow = Os + sp::QT * sp::LDO;
    float* lrow = mrow + sp::QT;
    unsigned char* inr = (unsigned char*) (lrow + sp::QT);   // the row routes the current block
    uint32_t* bm = (uint32_t*) (smem + ((sp::kQ + sp::kK + sp::kSV + sp::kO + sp::QT * 8 + sp::QT + 15) / 16 * 16));
    // the tile: 16 positions x 4 heads of one group (warp w = head h0 + w); row r = (position t0 + r % 16, head h0 + r / 16)
    constexpr int NP = 16;
    const int G = H / HKV, h0 = blockIdx.y * 4, g = h0 / G;
    const int t0 = blockIdx.x * NP, tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    const int p0 = *pos0;
    const int npos = min(NP, T - t0);
    const int nwords = (n_blocks + 31) / 32;
    for (int i = tid; i < nwords; i += blockDim.x) bm[i] = 0;
    for (int i = tid; i < sp::QT * sp::LDO; i += blockDim.x) Os[i] = 0.f;
    if (tid < sp::QT) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    for (int i = tid; i < sp::QT * kHD; i += blockDim.x) {   // Q tile, FP16
        const int r = i / kHD, d = i % kHD, ip = r % NP;
        Qs[r * sp::LDH + d] = __float2half(ip < npos ? q[((size_t) (t0 + ip) * H + h0 + r / NP) * kHD + d] : 0.f);
    }
    __syncthreads();
    for (int i = tid; i < npos * kRoute; i += blockDim.x) {   // the union of the positions' routes
        const int b = route[((size_t) (t0 + i / kRoute) * HKV + g) * kRoute + i % kRoute];
        if (b >= 0) atomicOr(&bm[b >> 5], 1u << (b & 31));
    }
    __syncthreads();
    for (int wd = 0; wd < nwords; ++wd) {
        uint32_t bits = bm[wd];
        while (bits) {
            const int b = wd * 32 + __ffs(bits) - 1;
            bits &= bits - 1;
            if (tid < NP) {   // does position tid route block b?
                bool in = false;
                if (tid < npos) {
                    const int32_t* rr = route + ((size_t) (t0 + tid) * HKV + g) * kRoute;
                    for (int j = 0; j < kRoute; ++j) in |= rr[j] == b;
                }
                inr[tid] = in;
            }
            for (int half = 0; half < kBlk / sp::KS; ++half) {
                const int k0 = b * kBlk + half * sp::KS;
                __syncthreads();
                {   // K half -> Ks
                    for (int i = tid; i < sp::KS * kHD / 8; i += blockDim.x) {
                        const int r = i / (kHD / 8), c = i % (kHD / 8);
                        if (k0 + r <= p0 + T - 1) kv_copy8(kv, false, g, k0 + r, p0 + T - 1, c, Ks + r * sp::LDH + c * 8);
                        else *(uint4*) (Ks + r * sp::LDH + c * 8) = make_uint4(0, 0, 0, 0);   // never written: zeros
                    }
                }
                __syncthreads();
                {   // S rows of this warp = Q K^T
                    float* Sw = (float*) SV;
                    for (int nt = 0; nt < sp::KS / 16; ++nt) {
                        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                        wmma::fill_fragment(acc, 0.f);
                        for (int kk = 0; kk < kHD / 16; ++kk) {
                            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                            wmma::load_matrix_sync(fa, Qs + (w * 16) * sp::LDH + kk * 16, sp::LDH);
                            wmma::load_matrix_sync(fb, Ks + (nt * 16) * sp::LDH + kk * 16, sp::LDH);
                            wmma::mma_sync(acc, fa, fb, acc);
                        }
                        wmma::store_matrix_sync(Sw + (w * 16) * sp::LDS + nt * 16, acc, sp::LDS, wmma::mem_row_major);
                    }
                }
                __syncthreads();   // every warp is done with Ks (P overwrites it)
                {   // the warp's rows: mask, online softmax, P (FP16) into Ks, O rescaled
                    const float* Sw = (const float*) SV;
                    __half* Ps = Ks;
                    for (int rr = 0; rr < 16; ++rr) {
                        const int r = w * 16 + rr;
                        const int pr = p0 + t0 + rr;
                        const bool on = rr < npos && inr[rr];
                        float v[2];
                        for (int c = 0; c < 2; ++c) {
                            const int j = lane + 32 * c;
                            v[c] = (on && k0 + j <= pr) ? Sw[r * sp::LDS + j] * scale : -INFINITY;
                        }
                        float mx = fmaxf(v[0], v[1]);
                        for (int x = 16; x > 0; x >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, x));
                        const float m_old = mrow[r];
                        const float m_new = fmaxf(m_old, mx);
                        float ps = 0.f;
                        for (int c = 0; c < 2; ++c) {
                            const float e = (m_new == -INFINITY || v[c] == -INFINITY) ? 0.f : expf(v[c] - m_new);
                            Ps[r * sp::LDP + lane + 32 * c] = __float2half(e);
                            ps += e;
                        }
                        for (int x = 16; x > 0; x >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, x);
                        const float alpha = (m_old == -INFINITY) ? 0.f : expf(m_old - m_new);
                        if (m_new != -INFINITY && alpha != 1.f)
                            for (int d = lane; d < kHD; d += 32) Os[r * sp::LDO + d] *= alpha;
                        __syncwarp();
                        if (lane == 0) {
                            if (m_new != -INFINITY) {
                                lrow[r] = lrow[r] * alpha + ps;
                                mrow[r] = m_new;
                            }
                        }
                        __syncwarp();
                    }
                }
                __syncthreads();   // S is dead: the V half takes its place
                {
                    __half* Vs = (__half*) SV;
                    for (int i = tid; i < sp::KS * kHD / 8; i += blockDim.x) {
                        const int r = i / (kHD / 8), c = i % (kHD / 8);
                        if (k0 + r <= p0 + T - 1) kv_copy8(kv, true, g, k0 + r, p0 + T - 1, c, Vs + r * sp::LDH + c * 8);
                        else *(uint4*) (Vs + r * sp::LDH + c * 8) = make_uint4(0, 0, 0, 0);
                    }
                }
                __syncthreads();
                {   // O rows of this warp += P V
                    const __half* Vs = (const __half*) SV;
                    const __half* Ps = Ks;
                    for (int dt = 0; dt < kHD / 16; ++dt) {
                        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                        wmma::load_matrix_sync(acc, Os + (w * 16) * sp::LDO + dt * 16, sp::LDO, wmma::mem_row_major);
                        for (int kk = 0; kk < sp::KS / 16; ++kk) {
                            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                            wmma::load_matrix_sync(fa, Ps + (w * 16) * sp::LDP + kk * 16, sp::LDP);
                            wmma::load_matrix_sync(fb, Vs + (kk * 16) * sp::LDH + dt * 16, sp::LDH);
                            wmma::mma_sync(acc, fa, fb, acc);
                        }
                        wmma::store_matrix_sync(Os + (w * 16) * sp::LDO + dt * 16, acc, sp::LDO, wmma::mem_row_major);
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < sp::QT * kHD; i += blockDim.x) {
        const int r = i / kHD, d = i % kHD, ip = r % NP;
        if (ip < npos) out[((size_t) (t0 + ip) * H + h0 + r / NP) * kHD + d] = Os[r * sp::LDO + d] / lrow[r];
    }
}

/// The prompt's block-sparse attention, FlashAttention-2 style on mma.sync m16n8k16 (FP16 in, FP32 accumulation): grid
/// (ceil(T / 16), H / 4), block 128; warp w = head h0 + w over the tile's 16 positions.  Q, S, P and O live in
/// registers (the m16n8 accumulator layout is known, so each row's online-softmax rescale is a register multiply); the
/// K / V halves of 64 keys pass through shared memory (V read with ldmatrix.trans).  Each row sees exactly its routed
/// keys <= its position, as k_sparse_tile / k_sparse_part.
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *(const uint32_t*) &h;
}
__global__ void __launch_bounds__(128) k_sparse_fa2(const float* q, KVP kv, const int32_t* route, const int32_t* pos0, int T,
                                                    int H, int HKV, int ctx, int n_blocks, float scale, float* out) {
    constexpr int NP = 16, KS = 64, LD = kHD + 8;
    __shared__ __align__(16) __half Ks[KS * LD];
    __shared__ __align__(16) __half Vs[KS * LD];
    __shared__ uint32_t bm[4096 / 32];
    __shared__ unsigned char inr[NP];
    const int G = H / HKV, h0 = blockIdx.y * 4, g = h0 / G, h = h0 + threadIdx.x / 32;
    const int t0 = blockIdx.x * NP, tid = threadIdx.x, lane = tid % 32, gid = lane >> 2, tig = lane & 3;
    const int p0 = *pos0;
    const int npos = min(NP, T - t0);
    const int nwords = (n_blocks + 31) / 32;
    for (int i = tid; i < nwords; i += blockDim.x) bm[i] = 0;
    // Q fragments for the warp's 16 rows (positions t0 + gid, t0 + gid + 8), k = 128 in 8 steps
    uint32_t qa[8][4];
    {
        const float* q0 = q + ((size_t) (t0 + gid) * H + h) * kHD;
        const float* q1 = q + ((size_t) (t0 + gid + 8) * H + h) * kHD;
        const bool r0 = gid < npos, r1 = gid + 8 < npos;
        for (int kk = 0; kk < 8; ++kk) {
            const int c = kk * 16 + tig * 2;
            qa[kk][0] = r0 ? pack_h2(q0[c], q0[c + 1]) : 0u;
            qa[kk][1] = r1 ? pack_h2(q1[c], q1[c + 1]) : 0u;
            qa[kk][2] = r0 ? pack_h2(q0[c + 8], q0[c + 9]) : 0u;
            qa[kk][3] = r1 ? pack_h2(q1[c + 8], q1[c + 9]) : 0u;
        }
    }
    __syncthreads();
    for (int i = tid; i < npos * kRoute; i += blockDim.x) {
        const int b = route[((size_t) (t0 + i / kRoute) * HKV + g) * kRoute + i % kRoute];
        if (b >= 0) atomicOr(&bm[b >> 5], 1u << (b & 31));
    }
    __syncthreads();
    float o[16][4];
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float mr[2] = {-INFINITY, -INFINITY}, lr[2] = {0.f, 0.f};
    const int pr[2] = {p0 + t0 + gid, p0 + t0 + gid + 8};
    for (int wd = 0; wd < nwords; ++wd) {
        uint32_t bits = bm[wd];
        while (bits) {
            const int b = wd * 32 + __ffs(bits) - 1;
            bits &= bits - 1;
            __syncthreads();   // the previous half's readers are done with Ks / Vs / inr
            if (tid < NP) {
                bool in = false;
                if (tid < npos) {
                    const int32_t* rr = route + ((size_t) (t0 + tid) * HKV + g) * kRoute;
                    for (int j = 0; j < kRoute; ++j) in |= rr[j] == b;
                }
                inr[tid] = in;
            }
            for (int hf = 0; hf < kBlk / KS; ++hf) {
                const int k0 = b * kBlk + hf * KS;
                if (hf) __syncthreads();
                for (int i = tid; i < KS * kHD / 8; i += blockDim.x) {
                    const int r = i / (kHD / 8), c = i % (kHD / 8);
                    if (k0 + r <= p0 + T - 1) {
                        kv_copy8(kv, false, g, k0 + r, p0 + T - 1, c, Ks + r * LD + c * 8);
                        kv_copy8(kv, true, g, k0 + r, p0 + T - 1, c, Vs + r * LD + c * 8);
                    } else {   // cells past the newest are never written (a borrowed staging buffer holds old bytes): zeros
                        *(uint4*) (Ks + r * LD + c * 8) = make_uint4(0, 0, 0, 0);
                        *(uint4*) (Vs + r * LD + c * 8) = make_uint4(0, 0, 0, 0);
                    }
                }
                __syncthreads();
                const bool on0 = gid < npos && inr[gid], on1 = gid + 8 < npos && inr[gid + 8];
                if (!__any_sync(0xffffffffu, on0 || on1)) continue;   // no row of the tile routes it (uniform: every warp
                                                                      // sees the same positions)
                float sc[8][4];
                for (int j = 0; j < 8; ++j) {
                    sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
                    const __half* kr = Ks + (j * 8 + gid) * LD + tig * 2;
                    for (int kk = 0; kk < 8; ++kk) {
                        const uint32_t b0 = *(const uint32_t*) (kr + kk * 16), b1 = *(const uint32_t*) (kr + kk * 16 + 8);
                        mma16816(sc[j], qa[kk], b0, b1);
                    }
                }
                // mask, online softmax per row (rows gid: c 0..1, gid + 8: c 2..3)
                float mx[2] = {-INFINITY, -INFINITY};
                for (int j = 0; j < 8; ++j)
                    for (int c = 0; c < 4; ++c) {
                        const int rr = c >> 1, key = k0 + j * 8 + tig * 2 + (c & 1);
                        const bool ok = (rr ? on1 : on0) && key <= pr[rr];
                        sc[j][c] = ok ? sc[j][c] * scale : -INFINITY;
                        mx[rr] = fmaxf(mx[rr], sc[j][c]);
                    }
                float alpha[2], sum[2] = {0.f, 0.f};
                for (int rr = 0; rr < 2; ++rr) {
                    mx[rr] = fmaxf(mx[rr], __shfl_xor_sync(0xffffffffu, mx[rr], 1));
                    mx[rr] = fmaxf(mx[rr], __shfl_xor_sync(0xffffffffu, mx[rr], 2));
                    const float mn = fmaxf(mr[rr], mx[rr]);
                    alpha[rr] = (mr[rr] == -INFINITY) ? 0.f : expf(mr[rr] - mn);
                    if (mn == -INFINITY) alpha[rr] = 1.f;
                    mx[rr] = mn;
                }
                for (int j = 0; j < 8; ++j)
                    for (int c = 0; c < 4; ++c) {
                        const int rr = c >> 1;
                        const float e = (sc[j][c] == -INFINITY) ? 0.f : expf(sc[j][c] - mx[rr]);
                        sc[j][c] = e;
                        sum[rr] += e;
                    }
                for (int rr = 0; rr < 2; ++rr) {
                    sum[rr] += __shfl_xor_sync(0xffffffffu, sum[rr], 1);
                    sum[rr] += __shfl_xor_sync(0xffffffffu, sum[rr], 2);
                    lr[rr] = lr[rr] * alpha[rr] + sum[rr];
                    if (mx[rr] != -INFINITY) mr[rr] = mx[rr];
                }
                for (int n = 0; n < 16; ++n) {
                    o[n][0] *= alpha[0]; o[n][1] *= alpha[0];
                    o[n][2] *= alpha[1]; o[n][3] *= alpha[1];
                }
                // O += P V: P from the S accumulators (two n8 tiles = one k16 A fragment), V by ldmatrix.trans
                for (int kk = 0; kk < KS / 16; ++kk) {
                    uint32_t pa[4];
                    pa[0] = pack_h2(sc[2 * kk][0], sc[2 * kk][1]);
                    pa[1] = pack_h2(sc[2 * kk][2], sc[2 * kk][3]);
                    pa[2] = pack_h2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]);
                    pa[3] = pack_h2(sc[2 * kk + 1][2], sc[2 * kk + 1][3]);
                    for (int n = 0; n < 16; ++n) {
                        const __half* vp = Vs + (kk * 16 + (lane & 15)) * LD + n * 8;
                        const uint32_t sa = (uint32_t) __cvta_generic_to_shared(vp);
                        uint32_t v0, v1;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(v0), "=r"(v1) : "r"(sa));
                        mma16816(o[n], pa, v0, v1);
                    }
                }
            }
        }
    }
    for (int n = 0; n < 16; ++n)
        for (int c = 0; c < 4; ++c) {
            const int rr = c >> 1, r = gid + rr * 8, d = n * 8 + tig * 2 + (c & 1);
            if (r < npos) out[((size_t) (t0 + r) * H + h) * kHD + d] = o[n][c] / lr[rr];
        }
}

__global__ void k_to_bf16(const float* x, __nv_bfloat16* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}
/// y[t][o] = W[o] . x[t] for BF16 W [n_out][n_in] (n_in % 256 == 0): a warp per row, 8 rows per block.
__global__ void k_gemv_bf16(const __nv_bfloat16* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const __nv_bfloat162* wr = (const __nv_bfloat162*) (W + (size_t) o * n_in);
    const float2* xr = (const float2*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 2; i += 32) {
        const float2 a = __bfloat1622float2(wr[i]), b = xr[i];
        acc += a.x * b.x + a.y * b.y;
    }
    for (int mm = 16; mm > 0; mm >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, mm);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}

/// y[r] = rms_norm(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// The MTP layer's input streams: X[t][s] = hout[t * 4 + s] + eout[t] (h_proj of each normed stream + e_proj of the
/// normed next-token embedding).  grid1(T * N).
__global__ void k_mtp_mix(const float* hout, const float* eout, float* X, int N, int64_t n) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int64_t t = j / N, i = j % N;
    for (int s = 0; s < kHC; ++s) X[(t * kHC + s) * N + i] = hout[(t * kHC + s) * N + i] + eout[j];
}

/// The routed entries grouped per expert, only those with a VRAM slot (slot_of[x] >= 0; null: every expert, slot =
/// expert): one group per expert used (ascending id), its blob address, its entries in entry order.  One block of 256
/// (n_expert <= 256): thread x counts expert x's entries, a block scan places the groups.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of, const uint8_t* blobs,
                        unsigned long long stride, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                        int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int scnt[256], sgrp[256];
    const int x = threadIdx.x;
    int c = 0;
    const bool ok = x < n_expert && (!slot_of || slot_of[x] >= 0);
    if (ok)
        for (int e = 0; e < n_ent; ++e) c += ids[e] == x;
    scnt[x] = c;
    sgrp[x] = c > 0;
    __syncthreads();
    for (int d = 1; d < 256; d <<= 1) {   // inclusive scans (Hillis-Steele)
        const int a = x >= d ? scnt[x - d] : 0, g = x >= d ? sgrp[x - d] : 0;
        __syncthreads();
        scnt[x] += a;
        sgrp[x] += g;
        __syncthreads();
    }
    const int off = scnt[x] - c, gi = sgrp[x] - (c > 0);
    if (c > 0) {
        grp_ptr[gi] = (unsigned long long) (blobs + (size_t) (slot_of ? slot_of[x] : x) * stride);
        grp_start[gi] = off;
        int p = off;
        for (int e = 0; e < n_ent; ++e)
            if (ids[e] == x) {
                ent_dst[p] = e;
                ent_tok[p] = e / K_;
                ++p;
            }
    }
    if (x == 255) {
        grp_start[sgrp[255]] = scnt[255];
        *n_groups = sgrp[255];
    }
}
/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's sorted rows; sorted
/// row r holds entry dst[r] of token src[r]; inv[e] = entry e's sorted row.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src, int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent).
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}

/// The greedy pick of each logits row (lowest id on ties), two stages.
constexpr int kArgBlocks = 128;
/// The probability of the row's top logit: 1 / sum_i exp(x_i - max).  Stage 1: kArgBlocks partial sums of exp(x - m)
/// with m = the row's max (from k_argmax's partials pv); stage 2 (one block): the total.
__global__ void k_top_prob(const float* x, int n, const float* pv, float* part, float* out, bool stage2) {
    __shared__ float sh[kThreads];
    float m = -INFINITY;
    for (int i = 0; i < kArgBlocks; ++i) m = fmaxf(m, pv[i]);
    float s = 0.f;
    if (stage2) {
        for (int i = threadIdx.x; i < kArgBlocks; i += blockDim.x) s += part[i];
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += expf(x[i] - m);
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if ((int) threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = 1.0f / sh[0];
        else part[blockIdx.x] = sh[0];
    }
}
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * kArgBlocks;
    pi += (size_t) blockIdx.y * kArgBlocks;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    Mat wq, wk, wv, wo;
    // the MBSA indexer: [q_proj (4 x 128 rows) | k_proj (128 rows)] BF16 [640][N], its q / k norms
    const __nv_bfloat16* idx_w = nullptr;
    const float *idx_qn = nullptr, *idx_kn = nullptr;
    // mHC: fn [24][4N] f32, base [24], scale [3]
    const float *hca_fn = nullptr, *hca_base = nullptr, *hca_scale = nullptr;
    const float *hcf_fn = nullptr, *hcf_base = nullptr, *hcf_scale = nullptr;
    bool dense = false;
    Mat fg, fu, fd;                       // dense FFN
    const float *router = nullptr, *bias = nullptr;   // [n_expert][N], [n_expert]
    Mat sg, su, sd;                       // the shared expert
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;             // VRAM slots, `stride` apart
    size_t stride = 0;
    int n_slots = 0;
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;   // the mapped GGUF's expert tensors
    size_t src_gsz = 0, src_dsz = 0;
    std::vector<C::NativeFmt> efmt;       // per expert: the CPU layout reaching its up / down rows in the mapping
};

struct Model {
    std::shared_ptr<strata::GgufModel> file;
    int64_t n_layers = 0, n_embd = 0, n_head = 0, n_head_kv = 0, n_ff = 0, n_ff_exp = 0, n_ff_sh = 0, n_expert = 0,
            top_k = 0, n_dense = 0, n_vocab = 0;
    int hc_iters = 20;
    float eps = 1e-6f, eps_hc = 1e-6f, rope_base = 2e7f;
    std::vector<Layer> layers;
    const float *out_norm = nullptr, *head_fn = nullptr, *head_base = nullptr, *head_scale = nullptr;
    Mat head;
    // the MTP layer (--mtp): layers[n_layers] (dense attention, no indexer), its input projections and its head
    bool has_mtp = false;
    std::shared_ptr<strata::GgufModel> mtp_file;
    Mat e_proj, h_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_fnorm = nullptr, *mtp_hc_fn = nullptr, *mtp_hc_base = nullptr,
                *mtp_hc_scale = nullptr;
    const uint8_t* embd = nullptr;   // host (the mapping), Q8_0 rows
    size_t embd_row = 0;
    int embd_type = -1;
    bool hybrid = false;
};

constexpr size_t kTail = 4096;   // zeroed bytes after the last expert slot
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}
size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}
size_t lcm_sz(size_t a, size_t b) {
    size_t x = a, y = b;
    while (y) { const size_t r = x % y; x = y; y = r; }
    return a / x * b;
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16, ex = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t u;
    if (ex == 0) {
        if (man == 0) u = sign;
        else {
            uint32_t mm = man, sh = 0;
            while ((mm & 0x400) == 0) { mm <<= 1; ++sh; }
            u = sign | ((uint32_t) (127 - 14 - sh) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (ex == 0x1f) u = sign | 0x7f800000u | (man << 13);
    else u = sign | ((ex + (127 - 15)) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

size_t g_weight_bytes = 0;
uint8_t* g_warena = nullptr;   // the dense weights' arena (bump allocated, 256-byte aligned)
size_t g_warena_left = 0;
void* walloc(size_t bytes) {
    bytes = (bytes + 255) / 256 * 256;
    if (g_warena && bytes <= g_warena_left) {
        void* p = g_warena;
        g_warena += bytes;
        g_warena_left -= bytes;
        return p;
    }
    void* p = nullptr;
    CK(cudaMalloc(&p, bytes));
    return p;
}
/// One arena for every tensor of `gm` that is uploaded (not the routed experts, not the token embeddings).
void warena_for(const strata::GgufModel& gm) {
    size_t total = 0;
    for (size_t i = 0; i < gm.size(); ++i)
        for (const auto& t : gm.shard(i).tensors()) {
            if (t.name.find("_exps") != std::string::npos || t.name == "token_embd.weight") continue;
            total += (strata::tensor_payload_bytes(t) + 255) / 256 * 256;
        }
    total += (size_t) 64 << 20;   // the indexer's two parts in one buffer, slack
    CK(cudaMalloc((void**) &g_warena, total));
    g_warena_left = total;
}
const float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    if (t.type != 0) {
        std::fprintf(stderr, "strata-mx: %s is %s, expected F32\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = (float*) walloc(t.elements() * 4);
    CK(cudaMemcpy(d, src, t.elements() * 4, cudaMemcpyHostToDevice));
    g_weight_bytes += t.elements() * 4;
    return d;
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    const strata::MetaValue* arch = gm.meta().get("general.architecture");
    if (!arch || arch->s != "maxwell-moe") {
        err = "strata-mx runs maxwell-moe files only";
        return false;
    }
    warena_for(gm);
    const std::string p = "maxwell-moe.";
    auto num = [&](const std::string& k) -> double {
        const strata::MetaValue* v = gm.meta().get(p + k);
        if (!v) {
            std::fprintf(stderr, "strata-mx: missing %s%s\n", p.c_str(), k.c_str());
            std::exit(1);
        }
        return v->num();
    };
    m.n_layers = (int64_t) num("block_count");
    m.n_embd = (int64_t) num("embedding_length");
    m.n_head = (int64_t) num("attention.head_count");
    m.n_head_kv = (int64_t) num("attention.head_count_kv");
    m.n_ff = (int64_t) num("feed_forward_length");
    m.n_ff_exp = (int64_t) num("expert_feed_forward_length");
    m.n_ff_sh = (int64_t) num("expert_shared_feed_forward_length");
    m.n_expert = (int64_t) num("expert_count");
    m.top_k = (int64_t) num("expert_used_count");
    m.n_dense = (int64_t) num("leading_dense_block_count");
    m.eps = (float) num("attention.layer_norm_rms_epsilon");
    m.rope_base = (float) num("rope.freq_base");
    m.hc_iters = (int) num("hyper_connection.sinkhorn_iterations");
    m.eps_hc = (float) num("hyper_connection.epsilon");
    if (num("hyper_connection.count") != kHC || num("attention.key_length") != kHD || num("attention.value_length") != kHD ||
        num("rope.dimension_count") != kHD || num("expert_gating_func") != 2 || m.top_k > kMaxK || m.n_expert > 256 ||
        num("expert_shared_count") != 1) {
        err = "a geometry this driver's kernels do not take (4 HC streams, head 128 fully rotated, sigmoid gating, <= 256 "
              "experts, top-k <= 16, one shared expert)";
        return false;
    }
    if (const strata::MetaValue* v = gm.meta().get(p + "expert_weights_scale"); v && v->num() != 1.0 && v->num() != 0.0) {
        err = "expert_weights_scale != 1 is not wired";
        return false;
    }
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-mx: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = (uint8_t*) walloc(nb);
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        g_weight_bytes += nb;
        w.d = d;
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) m.n_layers);
    for (int64_t l = 0; l < m.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.ffn_norm = f32(L(l, "ffn_norm.weight"));
        y.q_norm = f32(L(l, "attn_q_norm.weight"));
        y.k_norm = f32(L(l, "attn_k_norm.weight"));
        y.wq = mat(L(l, "attn_q.weight"));
        y.wk = mat(L(l, "attn_k.weight"));
        y.wv = mat(L(l, "attn_v.weight"));
        y.wo = mat(L(l, "attn_output.weight"));
        {
            size_t s1 = 0, s2 = 0;
            const strata::TensorInfo* tq = find(L(l, "indexer.q_proj.weight"), s1);
            const strata::TensorInfo* tk = find(L(l, "indexer.k_proj.weight"), s2);
            if (tq->type != 30 || tk->type != 30 || tq->shape[0] != (uint64_t) m.n_embd || tq->shape[1] != kIdxHeads * kIdxDim ||
                tk->shape[1] != kIdxDim) {
                err = L(l, "indexer projections are not BF16 [N][512] / [N][128]");
                return false;
            }
            const size_t qb = tq->elements() * 2, kb = tk->elements() * 2;
            uint8_t* d = (uint8_t*) walloc(qb + kb);
            CK(cudaMemcpy(d, gm.shard(s1).tensor_data(*tq), qb, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d + qb, gm.shard(s2).tensor_data(*tk), kb, cudaMemcpyHostToDevice));
            g_weight_bytes += qb + kb;
            y.idx_w = (const __nv_bfloat16*) d;
            y.idx_qn = f32(L(l, "indexer.q_norm.weight"));
            y.idx_kn = f32(L(l, "indexer.k_norm.weight"));
        }
        y.hca_fn = f32(L(l, "hc_attn_fn.weight"));
        y.hca_base = f32(L(l, "hc_attn_base.weight"));
        y.hca_scale = f32(L(l, "hc_attn_scale.weight"));
        y.hcf_fn = f32(L(l, "hc_ffn_fn.weight"));
        y.hcf_base = f32(L(l, "hc_ffn_base.weight"));
        y.hcf_scale = f32(L(l, "hc_ffn_scale.weight"));
        y.dense = l < m.n_dense;
        if (y.dense) {
            y.fg = mat(L(l, "ffn_gate.weight"));
            y.fu = mat(L(l, "ffn_up.weight"));
            y.fd = mat(L(l, "ffn_down.weight"));
            continue;
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.bias = f32(L(l, "exp_probs_b.bias"));
        y.sg = mat(L(l, "ffn_gate_shexp.weight"));
        y.su = mat(L(l, "ffn_up_shexp.weight"));
        y.sd = mat(L(l, "ffn_down_shexp.weight"));
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type || !K::native_expert_supported((int) tg->type, (int) td->type, m.n_embd, m.n_ff_exp)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the grouped expert kernels do not take";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, m.n_embd, m.n_ff_exp);
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.src_gsz = (size_t) m.n_ff_exp * y.L.gu_row;
        y.src_dsz = (size_t) m.n_embd * y.L.d_row;
    }
    m.out_norm = f32("output_norm.weight");
    m.head_fn = f32("output_hc_fn.weight");
    m.head_base = f32("output_hc_base.weight");
    m.head_scale = f32("output_hc_scale.weight");
    m.head = mat("output.weight");
    m.n_vocab = m.head.n_out;
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        if (m.embd_type != 8) {
            err = "token_embd must be Q8_0 (the host-side lookup)";
            return false;
        }
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        m.embd = gm.shard(s).tensor_data(*t);
    }
    std::fprintf(stderr, "strata-mx: %s: %lld layers (%lld dense), %lld experts top %lld, %.2f GiB of dense weights on the "
                 "GPU in %.1f s\n", path.c_str(), (long long) m.n_layers, (long long) m.n_dense, (long long) m.n_expert,
                 (long long) m.top_k, (double) g_weight_bytes / (1 << 30), ms_since(t0) / 1000.0);
    return true;
}

/// The MTP layer from tools/maxwell/convert_mtp.py's GGUF, appended as layers[n_layers].
bool load_mtp(const std::string& path, Model& m, std::string& err) {
    m.mtp_file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.mtp_file;
    const strata::MetaValue* arch = gm.meta().get("general.architecture");
    if (!arch || arch->s != "maxwell-moe-mtp") {
        err = path + " is not a maxwell-moe-mtp file (tools/maxwell/convert_mtp.py)";
        return false;
    }
    warena_for(gm);
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-mx: missing %s in the MTP file\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = (uint8_t*) walloc(nb);
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        g_weight_bytes += nb;
        w.d = d;
        return w;
    };
    m.mtp_enorm = f32("mtp.enorm.weight");
    m.mtp_hnorm = f32("mtp.hnorm.weight");
    m.mtp_fnorm = f32("mtp.final_norm.weight");
    m.mtp_hc_fn = f32("mtp.hc_head_fn.weight");
    m.mtp_hc_base = f32("mtp.hc_head_base.weight");
    m.mtp_hc_scale = f32("mtp.hc_head_scale.weight");
    m.e_proj = mat("mtp.e_proj.weight");
    m.h_proj = mat("mtp.h_proj.weight");
    m.layers.emplace_back();
    Layer& y = m.layers.back();
    const std::string B = "blk.0.";
    y.attn_norm = f32(B + "attn_norm.weight");
    y.ffn_norm = f32(B + "ffn_norm.weight");
    y.q_norm = f32(B + "attn_q_norm.weight");
    y.k_norm = f32(B + "attn_k_norm.weight");
    y.wq = mat(B + "attn_q.weight");
    y.wk = mat(B + "attn_k.weight");
    y.wv = mat(B + "attn_v.weight");
    y.wo = mat(B + "attn_output.weight");
    y.hca_fn = f32(B + "hc_attn_fn.weight");
    y.hca_base = f32(B + "hc_attn_base.weight");
    y.hca_scale = f32(B + "hc_attn_scale.weight");
    y.hcf_fn = f32(B + "hc_ffn_fn.weight");
    y.hcf_base = f32(B + "hc_ffn_base.weight");
    y.hcf_scale = f32(B + "hc_ffn_scale.weight");
    y.router = f32(B + "ffn_gate_inp.weight");
    y.bias = f32(B + "exp_probs_b.bias");
    y.sg = mat(B + "ffn_gate_shexp.weight");
    y.su = mat(B + "ffn_up_shexp.weight");
    y.sd = mat(B + "ffn_down_shexp.weight");
    size_t sg = 0, su = 0, sd = 0;
    const strata::TensorInfo* tg = find(B + "ffn_gate_exps.weight", sg);
    const strata::TensorInfo* tu = find(B + "ffn_up_exps.weight", su);
    const strata::TensorInfo* td = find(B + "ffn_down_exps.weight", sd);
    const Layer& L0 = m.layers[(size_t) m.n_dense];
    if ((int) tg->type != L0.L.gu_type || (int) tu->type != L0.L.gu_type || (int) td->type != L0.L.d_type) {
        err = "the MTP layer's experts must have the main layers' types";
        return false;
    }
    y.L = L0.L;
    y.stride = L0.stride;
    y.src_g = gm.shard(sg).tensor_data(*tg);
    y.src_u = gm.shard(su).tensor_data(*tu);
    y.src_d = gm.shard(sd).tensor_data(*td);
    y.src_gsz = L0.src_gsz;
    y.src_dsz = L0.src_dsz;
    m.has_mtp = true;
    std::fprintf(stderr, "strata-mx: MTP layer from %s\n", path.c_str());
    return true;
}

/// Expert e of layer L as its native blob [gate | up | down] at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ profiling
struct Prof {
    bool on = std::getenv("STRATA_MX_PROF") != nullptr;
    bool fine = on && std::getenv("STRATA_MX_PROF")[0] == '2';   // STRATA_MX_PROF=2: the attention's parts too
    cudaStream_t st = nullptr;
    Clock::time_point t;
    std::map<std::string, double> ms;
    std::map<std::string, long long> n;
    void begin() {
        if (!on) return;
        CK(cudaStreamSynchronize(st));
        t = Clock::now();
    }
    void mark(const char* what) {   // the time since the last mark (or begin) goes to `what`
        if (!on) return;
        CK(cudaStreamSynchronize(st));
        const auto now = Clock::now();
        ms[what] += std::chrono::duration<double, std::milli>(now - t).count();
        ++n[what];
        t = now;
    }
    void report(const char* title, double denom, const char* unit) {
        if (!on || ms.empty()) return;
        double tot = 0;
        for (auto& kv : ms) tot += kv.second;
        std::fprintf(stderr, "strata-mx PROF %s: %.1f ms total, %.3f ms per %s\n", title, tot, tot / denom, unit);
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : ms) v.emplace_back(kv.second, kv.first);
        std::sort(v.rbegin(), v.rend());
        for (auto& x : v)
            std::fprintf(stderr, "  %-14s %9.2f ms  %5.1f%%  %.3f ms per %s\n", x.second.c_str(), x.first,
                         100.0 * x.first / tot, x.first / denom, unit);
        ms.clear();
        n.clear();
    }
};

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

// The buffers whose size grows with the rows of a forward: a small permanent set (decode, MTP windows, short prompts)
// and, for a streamed prompt, a big set carved from the slots' VRAM the prompt borrows anyway.
#define MX_ROWBUFS(F)                                                                                                  \
    F(float*, X) F(float*, h) F(float*, f) F(float*, emb) F(float*, raw) F(float*, post) F(float*, comb) F(float*, qx)  \
    F(float*, kx) F(float*, vx) F(float*, q) F(float*, att) F(float*, logits_r) F(float*, ewts) F(float*, eparts)       \
    F(float*, g) F(float*, u) F(float*, hh) F(float*, shy) F(float*, hfin) F(void*, xq) F(void*, q8) F(__half*, x16)    \
    F(__half*, q16) F(int32_t*, ids) F(int32_t*, ent_dst) F(int32_t*, ent_tok) F(float*, idx_x) F(float*, iq)          \
    F(float*, bsc) F(int32_t*, route) F(__nv_bfloat16*, xb16) F(float*, Xm) F(float*, mtp_e) F(float*, mtp_eo)          \
    F(float*, mtp_hs) F(float*, mtp_ho) F(float*, att_s) F(__half*, att_p16) F(size_t, att_floats) F(int, tmax)         \
    F(int32_t*, rinv) F(int32_t*, ident) F(void*, mq_x) F(void*, mq_hq) F(float*, mq_gu) F(float*, mq_h)
struct RowBufs {
#define MX_DECL(t, n) t n{};
    MX_ROWBUFS(MX_DECL)
#undef MX_DECL
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 4096, tsmall = 64, tbig = 512;   // the permanent rows, the streamed prompt's chunk
    int64_t pos = 0;
    // activations
#define MX_DECL(t, n) t n{};
    MX_ROWBUFS(MX_DECL)
#undef MX_DECL
    float* logits = nullptr;
    float* emb_h = nullptr;   // pinned [tmax][N]
    void* escr = nullptr;
    int32_t *grp_start, *n_groups, *pos0, *pick, *ami;
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int n_chunk = 0, att_rows = 0;
    std::vector<KVP> kvs;               // per layer: the KV cache (FP16 or 8-bit, --kv)
    std::vector<__half*> ikc;           // per layer: the indexer K cache (FP16)
    bool kv8 = false;
    bool kv_host = false;                   // --kv-host: the main layers' KV in pinned host memory (see KVP)
    int kv_first = 4096, kv_ring = 4096;
    int64_t ring_upto = -1;                 // the ring holds the cells before and up to this one (-1: to be refilled)
    std::vector<uint8_t*> kvh_base;         // per main layer: its host block (k8 | v8 | ks | vs)
    uint8_t* kvstage = nullptr;             // a streamed prompt's per-layer staging of the host KV (in the arena)
    __half *kd16 = nullptr, *vd16 = nullptr;   // --kv q8: the dense prompt path's FP16 copies [HKV][ctx][128]
    float *spacc = nullptr, *spm = nullptr, *spl = nullptr;
    static constexpr int kSparseRows = 64;   // rows per sparse attention launch (the partials' size)
    bool force_dense = std::getenv("STRATA_MX_DENSE") != nullptr;   // dense attention at every position (A/B, llama.cpp)
    long long sparse_rows = 0;
    int dump_attn_layer = std::getenv("STRATA_MX_DUMP_ATTN") ? std::atoi(std::getenv("STRATA_MX_DUMP_ATTN")) : -1;
    FILE* dump_attn = nullptr;   // STRATA_MX_DUMP_ATTN=L (file STRATA_MX_DUMP_ATTN_FILE): layer L's attention at decode steps
    template <class T_> void dget(std::vector<T_>& v, const T_* src, size_t n) {
        v.resize(n);
        CK(cudaMemcpy(v.data(), src, n * sizeof(T_), cudaMemcpyDeviceToHost));
    }
    void dump_attention(size_t li, int64_t p0) {
        if (!dump_attn) {
            const char* f = std::getenv("STRATA_MX_DUMP_ATTN_FILE");
            dump_attn = std::fopen(f ? f : "mx_attn.bin", "wb");
        }
        CK(cudaStreamSynchronize(st));
        const int H = (int) m->n_head, HKV = (int) m->n_head_kv, N = (int) m->n_embd;
        const int P = (int) p0 + 1;
        std::vector<float> hv, iqv, qv, av;
        std::vector<int32_t> rv;
        std::vector<__half> kv, vv, ikv;
        dget(hv, h, (size_t) N);
        dget(iqv, iq, (size_t) kIdxHeads * kIdxDim);
        dget(qv, q, (size_t) H * kHD);
        dget(av, att, (size_t) H * kHD);
        dget(rv, route, (size_t) kIdxHeads * kRoute);
        dget(ikv, ikc[li], (size_t) P * kIdxDim);
        kv.resize((size_t) HKV * P * kHD);
        vv.resize(kv.size());
        const __half *kb = kvs[li].k, *vb = kvs[li].v;
        if (kv8) {
            k_kv_deq<<<grid1((int64_t) HKV * P * kHD), kThreads, 0, st>>>(kvs[li], kd16, vd16, ctx, P, HKV);
            CK(cudaStreamSynchronize(st));
            kb = kd16;
            vb = vd16;
        }
        for (int g = 0; g < HKV; ++g) {
            CK(cudaMemcpy(kv.data() + (size_t) g * P * kHD, kb + (size_t) g * ctx * kHD, (size_t) P * kHD * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(vv.data() + (size_t) g * P * kHD, vb + (size_t) g * ctx * kHD, (size_t) P * kHD * 2, cudaMemcpyDeviceToHost));
        }
        const int32_t hdr[4] = {(int32_t) li, P, H, HKV};
        std::fwrite(hdr, 4, 4, dump_attn);
        std::fwrite(hv.data(), 4, hv.size(), dump_attn);
        std::fwrite(iqv.data(), 4, iqv.size(), dump_attn);
        std::fwrite(ikv.data(), 2, ikv.size(), dump_attn);
        std::fwrite(qv.data(), 4, qv.size(), dump_attn);
        std::fwrite(kv.data(), 2, kv.size(), dump_attn);
        std::fwrite(vv.data(), 2, vv.size(), dump_attn);
        std::fwrite(rv.data(), 4, rv.size(), dump_attn);
        std::fwrite(av.data(), 4, av.size(), dump_attn);
        std::fflush(dump_attn);
    }
    int32_t* hpin = nullptr;
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;

    Prof prof;
    // the hybrid mode
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;
        int32_t* slot_of_d = nullptr;
    };
    std::vector<Hy> hy;
    std::unique_ptr<C::ExpertPool> pool;
    std::vector<C::ExpertJobMulti> jobs;
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr;
    uint8_t* acts = nullptr;
    cudaEvent_t ev_route = nullptr;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;
    double t_wait_route = 0, t_pool = 0, t_jobs = 0, t_run = 0, t_fwd = 0, t_mtp = 0;   // decode: host time waiting for the router, and in the CPU pool
    // STRATA_MX_ROUTE_STATS: per decode step and layer, how many of the token's experts the previous token also used
    bool route_stats = std::getenv("STRATA_MX_ROUTE_STATS") != nullptr;
    std::vector<std::vector<int>> prev_ids;
    long long rs_shared = 0, rs_total = 0, rs_win4 = 0, rs_win4_n = 0;
    std::vector<std::vector<std::vector<int>>> hist_ids;
    static constexpr int kRebalanceEvery = 32;
    static constexpr int kGemmT = 8;
    // ---- the streamed prompt (hybrid mode, a prompt of >= stream_min tokens): layer by layer over the whole prompt,
    // each MoE layer's three expert tensors copied once into a staging buffer (two, a copy thread one layer ahead) in
    // VRAM borrowed from the slots (one arena), the experts on MMQ; the slots restored after the prompt
    uint8_t* arena = nullptr;
    size_t arena_bytes = 0, stage_bytes = 0;
    bool use_stream = false;
    int stream_min = 256;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *pos_tab = nullptr;
    double stream_wait_ms = 0, restore_ms = 0;
    size_t restore_used = 0;   // the arena bytes borrowed since the last restore (windows of one prompt)
    struct Streamer {
        const Model* m = nullptr;
        std::vector<int> seq;   // the MoE layers in order
        uint8_t* buf[2] = {};
        size_t gsz_all = 0, dsz_all = 0;   // a layer's gate (= up) and down tensor bytes
        cudaStream_t cst = nullptr;
        cudaEvent_t used[2] = {};
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        int ready = -1, released = -1;
        double copy_ms = 0;
        // pinned bounce pieces: the mapped file's pages copied in by kCopyThreads threads, then DMA at full PCIe speed
        static constexpr int kMaxPieces = 16, kMaxCopyThreads = 16;
        int kPieces = 4, kCopyThreads = 4;   // STRATA_MX_PIECES / STRATA_MX_COPY_THREADS / STRATA_MX_PIECE_MB
        size_t kPiece = (size_t) 32 << 20;
        uint8_t* pin[kMaxPieces] = {};
        cudaEvent_t pin_ev[kMaxPieces] = {};
        long long piece_n = 0;
        bool bounce = std::getenv("STRATA_MX_PAGEABLE") == nullptr;
        void copy_in(uint8_t* dst, const uint8_t* src, size_t bytes) {
            if (!bounce) {
                CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, cst));
                return;
            }
            for (size_t off = 0; off < bytes; off += kPiece) {
                const size_t n = std::min(kPiece, bytes - off);
                const int k = (int) (piece_n++ % kPieces);
                CK(cudaEventSynchronize(pin_ev[k]));   // its last DMA is done
                std::thread th[kMaxCopyThreads];
                const size_t part = (n / kCopyThreads + 4095) / 4096 * 4096;
                for (int i = 0; i < kCopyThreads; ++i) {
                    const size_t a = std::min(n, (size_t) i * part), b = std::min(n, a + part);
                    th[i] = std::thread([=] { if (b > a) std::memcpy(pin[k] + a, src + off + a, b - a); });
                }
                for (int i = 0; i < kCopyThreads; ++i) th[i].join();
                CK(cudaMemcpyAsync(dst + off, pin[k], n, cudaMemcpyHostToDevice, cst));
                CK(cudaEventRecord(pin_ev[k], cst));
            }
        }
        void start(const std::vector<int>& s_) {
            seq = s_;
            ready = released = -1;
            copy_ms = 0;
            th = std::thread([this] {
                for (int i = 0; i < (int) seq.size(); ++i) {
                    const int b = i % 2;
                    if (i >= 2) {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return released >= i - 2; });
                        lk.unlock();
                        CK(cudaEventSynchronize(used[b]));
                    }
                    const auto t0 = Clock::now();
                    const Layer& L = m->layers[(size_t) seq[(size_t) i]];
                    copy_in(buf[b], L.src_g, gsz_all);
                    copy_in(buf[b] + gsz_all, L.src_u, gsz_all);
                    copy_in(buf[b] + 2 * gsz_all, L.src_d, dsz_all);
                    CK(cudaStreamSynchronize(cst));
                    copy_ms += ms_since(t0);
                    std::lock_guard<std::mutex> lk(mu);
                    ready = i;
                    cv.notify_all();
                }
            });
        }
        const uint8_t* take(int i) {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return ready >= i; });
            return buf[i % 2];
        }
        void release(int i, cudaStream_t st) {
            CK(cudaEventRecord(used[i % 2], st));
            std::lock_guard<std::mutex> lk(mu);
            released = i;
            cv.notify_all();
        }
        void join() {
            if (th.joinable()) th.join();
        }
    } streamer;
    FILE* dump_layers = nullptr;           // --dump-layers: X after every layer, [pos][layer][4 N] f32
    std::vector<float> dump_buf;
    std::vector<float> inject;             // STRATA_MX_INJECT=F: the reference X before each layer (local errors only)

    /// The row buffers for T rows through `al` (bytes -> device pointer); `big`: the streamed prompt's set (MMQ's buffers
    /// too, a bigger attention score buffer).
    template <class A> void alloc_rows(int T_, bool big, A al) {
        tmax = T_;
        const size_t T = (size_t) T_, N = (size_t) m->n_embd, Kp = (size_t) m->top_k, H = (size_t) m->n_head,
                     HKV = (size_t) m->n_head_kv, FF = (size_t) m->n_ff_exp;
        const size_t FFmax = (size_t) std::max(m->n_ff, m->n_ff_sh);
        const size_t max_in = (size_t) std::max<int64_t>({m->n_embd, m->n_head * kHD, (int64_t) FFmax});
        const size_t big_in = std::max<size_t>(max_in, (size_t) kHC * N);   // the MTP h_proj: 4 rows per token
        auto F = [&](size_t n) { return (float*) al(n * 4); };
        X = F(T * kHC * N); h = F(T * N); f = F(T * N); emb = F(T * N);
        raw = F(T * kMix * kHcSplit); post = F(T * kHC); comb = F(T * kHC * kHC);
        qx = F(T * H * kHD); kx = F(T * HKV * kHD); vx = F(T * HKV * kHD); q = F(T * H * kHD); att = F(T * H * kHD);
        logits_r = F(T * (size_t) m->n_expert); ewts = F(T * Kp); eparts = F(T * Kp * N);
        g = F(T * FFmax); u = F(T * FFmax); hh = F(T * FFmax); shy = F(T * N); hfin = F(T * N);
        xq = al(K::native_q8_1_bytes((int) N, 1) * T);
        q8 = al(K::native_q8_1_bytes((int) big_in, 1) * T);
        x16 = (__half*) al(T * big_in * 2);
        q16 = (__half*) al(T * H * kHD * 2);
        ids = (int32_t*) al(T * Kp * 4); ent_dst = (int32_t*) al(T * Kp * 4); ent_tok = (int32_t*) al(T * Kp * 4);
        idx_x = F(T * (kIdxHeads + 1) * kIdxDim); iq = F(T * kIdxHeads * kIdxDim);
        bsc = F(T * kIdxHeads * (size_t) ((ctx + kBlk - 1) / kBlk));
        route = (int32_t*) al(T * kIdxHeads * kRoute * 4);
        xb16 = (__nv_bfloat16*) al(T * N * 2);
        if (m->has_mtp) {
            Xm = F(T * kHC * N); mtp_e = F(T * N); mtp_eo = F(T * N); mtp_hs = F(T * kHC * N); mtp_ho = F(T * kHC * N);
        } else {
            Xm = mtp_e = mtp_eo = mtp_hs = mtp_ho = nullptr;
        }
        att_floats = big ? (size_t) 16 << 20 : (size_t) 4 << 20;
        att_s = F(att_floats);
        att_p16 = (__half*) al(att_floats * 2);
        if (big) {
            rinv = (int32_t*) al(T * Kp * 4);
            ident = (int32_t*) al(T * Kp * 4);
            mq_x = al(MMQ::q8_bytes((int64_t) (T * Kp), m->n_embd));
            mq_hq = al(MMQ::q8_bytes((int64_t) (T * Kp), m->n_ff_exp));
            mq_gu = F(T * Kp * 2 * FF);
            mq_h = F(T * Kp * FF);
        } else {
            rinv = ident = nullptr;
            mq_x = mq_hq = nullptr;
            mq_gu = mq_h = nullptr;
        }
    }
    RowBufs save_rows() const {
        RowBufs r;
#define MX_SAVE(t, n) r.n = n;
        MX_ROWBUFS(MX_SAVE)
#undef MX_SAVE
        return r;
    }
    void load_rows(const RowBufs& r) {
#define MX_LOAD(t, n) n = r.n;
        MX_ROWBUFS(MX_LOAD)
#undef MX_LOAD
    }
    /// Bytes of the big set for T rows (256-byte aligned parts).
    size_t rows_bytes(int T_) {
        const RowBufs keep = save_rows();
        size_t total = 0;
        alloc_rows(T_, true, [&](size_t n) {
            total += (n + 255) / 256 * 256;
            return (void*) 0x1000;
        });
        load_rows(keep);
        return total;
    }

    void init(const Model& model, int ctx_, int tbig_) {
        m = &model;
        ctx = ctx_;
        tbig = std::max(tbig_, 1);
        tsmall = std::min(tsmall, tbig);
        const size_t T = (size_t) tsmall, N = (size_t) m->n_embd, Kp = (size_t) m->top_k;
        const size_t H = (size_t) m->n_head, HKV = (size_t) m->n_head_kv;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        prof.st = st;
        (void) Kp;
        logits = dalloc((size_t) m->n_vocab * 4);   // up to 4 head rows (a verify window of 3 drafts)
        CK(cudaHostAlloc((void**) &emb_h, (size_t) tbig * N * 4, cudaHostAllocDefault));
        alloc_rows(tsmall, false, [](size_t n) {
            void* p = nullptr;
            CK(cudaMalloc(&p, n));
            CK(cudaMemset(p, 0, n));
            return p;
        });
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-mx: cuBLAS handle failed\n");
            std::exit(1);
        }
        {
            std::string gerr;
            int64_t big = 0;
            for (const Layer& L : m->layers)
                for (const Mat* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.fg, &L.fu, &L.fd, &L.sg, &L.su, &L.sd})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            big = std::max<int64_t>(big, (int64_t) m->head.n_in * std::min<int64_t>(m->head.n_out, 1 << 16));
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-mx: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), m->n_ff_exp)));
        grp_ptr = dalloc_t<unsigned long long>((size_t) m->n_expert + 1);
        grp_start = dalloc_t<int32_t>((size_t) m->n_expert + 1);
        n_groups = dalloc_t<int32_t>(1);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(4);
        amv = dalloc((size_t) kArgBlocks * 4);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * 4);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        att_rows = (int) (H * std::min<int64_t>((int64_t) T, 8));
        pacc = dalloc((size_t) att_rows * n_chunk * kHD);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        for (size_t l = 0; l < m->layers.size(); ++l) {
            KVP kv;
            const bool host = kv_host && l < (size_t) m->n_layers;   // the MTP layer's dense attention: all in VRAM
            if (kv8 && host) {
                const size_t F = (size_t) std::min(kv_first, ctx), R = (size_t) kv_ring;
                kv.dstride = (int) F;
                kv.first = (int) F;
                kv.k8 = dalloc_t<int8_t>(HKV * F * kHD);
                kv.v8 = dalloc_t<int8_t>(HKV * F * kHD);
                kv.ks = dalloc(HKV * F);
                kv.vs = dalloc(HKV * F);
                kv.ring = (int) R;
                kv.k8r = dalloc_t<int8_t>(HKV * R * kHD);
                kv.v8r = dalloc_t<int8_t>(HKV * R * kHD);
                kv.ksr = dalloc(HKV * R);
                kv.vsr = dalloc(HKV * R);
                const size_t C = (size_t) ctx, nb = HKV * C * (2 * kHD + 8);
                uint8_t* hb = nullptr;
                CK(cudaHostAlloc((void**) &hb, nb, cudaHostAllocMapped | cudaHostAllocPortable));
                std::memset(hb, 0, nb);
                uint8_t* db = nullptr;
                CK(cudaHostGetDevicePointer((void**) &db, hb, 0));
                kvh_base.push_back(hb);
                kv.k8h = (int8_t*) db;
                kv.v8h = (int8_t*) (db + HKV * C * kHD);
                kv.ksh = (float*) (db + 2 * HKV * C * kHD);
                kv.vsh = (float*) (db + 2 * HKV * C * kHD + HKV * C * 4);
                kv.hstride = (int) C;
            } else if (kv8) {
                kv.dstride = ctx;
                kv.k8 = dalloc_t<int8_t>(HKV * ctx * kHD);
                kv.v8 = dalloc_t<int8_t>(HKV * ctx * kHD);
                kv.ks = dalloc(HKV * ctx);
                kv.vs = dalloc(HKV * ctx);
            } else {
                kv.dstride = ctx;
                kv.k = dalloc_t<__half>(HKV * ctx * kHD);
                kv.v = dalloc_t<__half>(HKV * ctx * kHD);
            }
            kvs.push_back(kv);
            ikc.push_back(dalloc_t<__half>((size_t) ctx * kIdxDim));
        }
        {
            const size_t np = (size_t) kSparseRows * HKV * kRoute * (H / HKV);
            spacc = dalloc(np * kHD);
            spm = dalloc(np);
            spl = dalloc(np);
        }
        Xlast = dalloc((size_t) kHC * N);
        pos_m = dalloc_t<int32_t>(1);
        top_part = dalloc(kArgBlocks);
        top_p = dalloc(1);
        if (kv8) {
            kd16 = dalloc_t<__half>(HKV * ctx * kHD);
            vd16 = dalloc_t<__half>(HKV * ctx * kHD);
        }
        CK(cudaFuncSetAttribute(k_sparse_part, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * kBlk * kHD * 2));
        CK(cudaFuncSetAttribute(k_sparse_tile, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sp::kSmem));
        CK(cudaDeviceSynchronize());
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        if (kv_host)
            std::fprintf(stderr, "strata-mx: --kv-host: the main layers' KV in pinned host memory (%.2f GiB), cells < %d and a ring of "
                         "%d in VRAM\n", (double) m->n_layers * HKV * ctx * (2 * kHD + 8) / (1 << 30), kv_first, kv_ring);
        std::fprintf(stderr, "strata-mx: context %d cells (%s KV + indexer K, %.2f GiB), prompt chunk %d, %s attention, %.2f GiB "
                     "of VRAM free\n", ctx, kv8 ? "8-bit" : "FP16",
                     (double) ((kv8 ? 2 * HKV * (kHD + 4) : 2 * HKV * kHD * 2) + kIdxDim * 2) * (size_t) ctx * m->layers.size() / (1 << 30),
                     tmax, force_dense ? "dense (STRATA_MX_DENSE)" : "block-sparse (MBSA) past 4096", (double) fr / (1 << 30));
    }

    void reset() {
        pos = 0;
        ring_upto = -1;
    }
    /// --kv-host: the ring refilled from the host copy when the next forward does not continue it (a prefix reuse, a KV
    /// file read, a rollback past the margin): cells max(first, p0 - ring + margin) .. p0 - 1.
    void ring_sync(int64_t p0) {
        if (!kv_host) return;
        if (ring_upto >= 0 && p0 <= ring_upto + 1 && p0 >= ring_upto + 1 - (kRingMargin - 1)) return;
        const int64_t lo = std::max<int64_t>(kv_first, p0 - kv_ring + kRingMargin), hi = p0;   // [lo, hi)
        const size_t HKV = (size_t) m->n_head_kv, C = (size_t) ctx, R = (size_t) kv_ring;
        for (size_t li = 0; li < (size_t) m->n_layers && lo < hi; ++li) {
            const KVP& kv = kvs[li];
            const uint8_t* hb = kvh_base[li];
            const int8_t* hk = (const int8_t*) hb;
            const int8_t* hv = (const int8_t*) (hb + HKV * C * kHD);
            const float* hks = (const float*) (hb + 2 * HKV * C * kHD);
            const float* hvs = hks + HKV * C;
            for (size_t g = 0; g < HKV; ++g)
                for (int64_t a = lo; a < hi;) {   // runs that do not wrap the ring
                    const int64_t slot = a % (int64_t) R, n = std::min<int64_t>(hi - a, (int64_t) R - slot);
                    const size_t hr = g * C + (size_t) a, rr = g * R + (size_t) slot;
                    CK(cudaMemcpyAsync(kv.k8r + rr * kHD, hk + hr * kHD, (size_t) n * kHD, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(kv.v8r + rr * kHD, hv + hr * kHD, (size_t) n * kHD, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(kv.ksr + rr, hks + hr, (size_t) n * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(kv.vsr + rr, hvs + hr, (size_t) n * 4, cudaMemcpyHostToDevice, st));
                    a += n;
                }
        }
        ring_upto = p0 - 1;
    }
    /// --kv-host: the VRAM first region refilled from the host copy for cells [a, b) (after a KV file read or a streamed
    /// prompt, which write the host copy only).
    void first_sync(int64_t a, int64_t b) {
        if (!kv_host) return;
        b = std::min<int64_t>(b, kv_first);
        if (a >= b) return;
        const size_t HKV = (size_t) m->n_head_kv, C = (size_t) ctx, F = (size_t) kv_first;
        for (size_t li = 0; li < (size_t) m->n_layers; ++li) {
            const KVP& kv = kvs[li];
            const uint8_t* hb = kvh_base[li];
            const int8_t* hk = (const int8_t*) hb;
            const int8_t* hv = (const int8_t*) (hb + HKV * C * kHD);
            const float* hks = (const float*) (hb + 2 * HKV * C * kHD);
            const float* hvs = hks + HKV * C;
            for (size_t g = 0; g < HKV; ++g) {
                const size_t hr = g * C + (size_t) a, dr = g * F + (size_t) a, n = (size_t) (b - a);
                CK(cudaMemcpyAsync(kv.k8 + dr * kHD, hk + hr * kHD, n * kHD, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(kv.v8 + dr * kHD, hv + hr * kHD, n * kHD, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(kv.ks + dr, hks + hr, n * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(kv.vs + dr, hvs + hr, n * 4, cudaMemcpyHostToDevice, st));
            }
        }
    }

    // ---- dense products: T <= kGemmT on the native MMVQ (q8_1 activations), else the FP16 GEMM
    void quant(const float* xin, int n, int T, void* qd) { K::native_quantize_q8_1(xin, qd, n * T, 1, st); }
    void prep(const float* xin, int n, int T, void* qd) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, qd);
    }
    void apply(const Mat& w, const void* qd, float* out, int T) {
        if (T > kGemmT) {
            gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
            return;
        }
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) qd + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out,
                           nc, st);
        }
    }
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// The mHC mapping products for T rows into raw: split-K partials (decode, kHcSplit slices) or whole sums (GEMM).
    static constexpr int kHcSplit = 16;
    int hc_mix(const float* W, const float* xin, int n_out, int T) {
        const int n_in = kHC * (int) m->n_embd;
        if (T <= kGemmT) {
            k_gemv_f32_split<<<dim3(n_out, kHcSplit, T), kThreads, 0, st>>>(W, xin, raw, n_in, n_out, kHcSplit);
            return kHcSplit;
        }
        f32mm(W, xin, raw, n_in, n_out, T);
        return 1;
    }
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-mx: cublasSgemm failed\n");
            std::exit(1);
        }
    }
    /// SwiGLU FFN: out = down(silu(gate h) * up h), FF wide.
    void ffn(const Mat& wg, const Mat& wu, const Mat& wd, const float* hin, float* out, int T) {
        const int N = (int) m->n_embd, FF = wg.n_out;
        prep(hin, N, T, xq);
        apply(wg, xq, g, T);
        apply(wu, xq, u, T);
        if (T <= kGemmT) {
            K::native_swiglu_quantize_q8_1(g, u, q8, FF, T, st);
            apply(wd, q8, out, T);
        } else {
            k_swiglu<<<grid1((int64_t) T * FF), kThreads, 0, st>>>(g, u, hh, (int64_t) T * FF);
            mm(wd, hh, out, T);
        }
    }

    // ---- attention: h -> f
    void attention(const Layer& L, size_t li, int T, int64_t p0) {
        const int H = (int) m->n_head, HKV = (int) m->n_head_kv;
        prep(h, (int) m->n_embd, T, xq);
        apply(L.wq, xq, qx, T);
        apply(L.wk, xq, kx, T);
        apply(L.wv, xq, vx, T);
        k_attn_prep<<<dim3(H + HKV, T), kHD, 0, st>>>(qx, kx, vx, L.q_norm, L.k_norm, q, kvs[li], H, HKV, pos0, ctx,
                                                      m->rope_base, m->eps);
        if (prof.fine) prof.mark("a_qkv");
        // the MBSA indexer: its K is cached for every token; its Q routes the rows past 4096
        const int NI = (kIdxHeads + 1) * kIdxDim, N = (int) m->n_embd;
        const bool has_idx = L.idx_w != nullptr;   // the MTP layer: plain dense attention (the vendor's MaxwellMoeAttention)
        if (!has_idx) {
        } else if (T <= kGemmT) {
            k_gemv_bf16<<<dim3(rows8(NI), T), kThreads, 0, st>>>(L.idx_w, h, idx_x, N, NI);
        } else {
            k_to_bf16<<<grid1((int64_t) T * N), kThreads, 0, st>>>(h, xb16, (int64_t) T * N);
            gemm.bf16((const uint16_t*) xb16, (const uint16_t*) L.idx_w, idx_x, T, NI, N);
        }
        if (has_idx)
            k_index_prep<<<dim3(kIdxHeads + 1, T), kIdxDim, 0, st>>>(idx_x, L.idx_qn, L.idx_kn, iq, ikc[li], pos0, m->rope_base,
                                                                   m->eps);
        const float scale = 1.0f / sqrtf((float) kHD);
        if (prof.fine) prof.mark("a_index");
        if (has_idx && !force_dense && p0 + T > kDenseUpTo) {
            const int nb = graph_mode ? (ctx + kBlk - 1) / kBlk : (int) ((p0 + T + kBlk - 1) / kBlk);
            if (T > kGemmT) {
                // token scores as FP16 GEMMs (FP32 accumulation) in row slices that fit the score buffer, then the block max
                const int P = nb * kBlk;   // whole blocks (cells past p0 + T are never read: their blocks are not eligible)
                const int rows = T * kIdxHeads;
                const int sl = (int) std::max<size_t>(1, std::min<size_t>((size_t) rows, att_floats / (size_t) P));
                k_f32_to_f16<<<grid1((int64_t) rows * kIdxDim), kThreads, 0, st>>>(iq, (__half*) att_p16, (int64_t) rows * kIdxDim);
                const float one = 1.0f, zero = 0.0f;
                for (int r0 = 0; r0 < rows; r0 += sl) {
                    const int nr = std::min(sl, rows - r0);
                    if (cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nr, kIdxDim, &one, ikc[li], CUDA_R_16F, kIdxDim,
                                     att_p16 + (size_t) r0 * kIdxDim, CUDA_R_16F, kIdxDim, &zero, att_s, CUDA_R_32F, P,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-mx: indexer scores GEMM failed\n");
                        std::exit(1);
                    }
                    k_block_max<<<dim3(nb, nr), kBlk, 0, st>>>(att_s, P, r0, pos0, nb, rsqrtf((float) kIdxDim), bsc);
                }
            } else {
                static const bool old_bs = std::getenv("STRATA_MX_OLD_BSCORES") != nullptr;   // A/B
                if (old_bs) k_block_scores<<<dim3(nb, T * kIdxHeads), kBlk, 0, st>>>(iq, ikc[li], pos0, nb, bsc);
                else k_block_scores4<<<dim3(nb, T), 128, 0, st>>>(iq, ikc[li], pos0, nb, bsc);
            }
            if (prof.fine) prof.mark("a_bscores");
            k_route_blocks<<<T * kIdxHeads, kThreads, 0, st>>>(bsc, nb, pos0, route);
            if (prof.fine) prof.mark("a_route");
            static const bool tile_off = std::getenv("STRATA_MX_SPARSE_ROWS") != nullptr;   // the per-row kernels (A/B)
            static const bool wmma_tile = std::getenv("STRATA_MX_SPARSE_WMMA") != nullptr;   // the WMMA tile kernel (A/B)
            if (T > kGemmT && !tile_off && !wmma_tile && nb <= 4096) {
                k_sparse_fa2<<<dim3((T + 15) / 16, H / 4), 128, 0, st>>>(q, kvs[li], route, pos0, T, H, HKV, ctx, nb, scale, att);
            } else if (T > kGemmT && !tile_off) {
                k_sparse_tile<<<dim3((T + 15) / 16, H / 4), 128, sp::kSmem, st>>>(q, kvs[li], route, pos0, T, H, HKV,
                                                                                       ctx, nb, scale, att);
            } else {
                for (int t0 = 0; t0 < T; t0 += kSparseRows) {
                    const int nt = std::min(kSparseRows, T - t0);
                    k_sparse_part<<<dim3(kRoute, nt * HKV), 256, 2 * kBlk * kHD * 2, st>>>(q, kvs[li], route, pos0, t0, H,
                                                                                        HKV, T, scale, spacc, spm, spl);
                    k_sparse_merge<<<nt * H, kHD, 0, st>>>(spacc, spm, spl, t0, H, HKV, att);
                }
            }
            sparse_rows += T;
        } else if (T > kGemmT) {
            // scores = Q K^T and out = P V as FP16 GEMMs per KV head (its query heads one strided batch over the shared
            // K / V straight from the FP16 cache), the causal softmax between; the queries in slices whose scores fit
            const int P = (int) p0 + T;
            const __half* kbase = kvs[li].k;
            const __half* vbase = kvs[li].v;
            if (kv8) {   // FP16 copies of the 8-bit cache's cells [0, P) for cuBLAS
                k_kv_deq<<<grid1((int64_t) HKV * P * kHD), kThreads, 0, st>>>(kvs[li], kd16, vd16, ctx, P, HKV);
                kbase = kd16;
                vbase = vd16;
            }
            const int ldp = (P + 7) / 8 * 8;
            const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (att_floats / ((size_t) H * ldp))));
            const int G = H / HKV;
            const float one = 1.0f, zero = 0.0f;
            k_to_f16<<<grid1((int64_t) T * H * kHD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * kHD);
            for (int t0 = 0; t0 < T; t0 += tq) {
                const int nt = std::min(tq, T - t0);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Kh = kbase + (size_t) kvh * ctx * kHD;
                    const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * kHD;
                    float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, kHD, &one, Kh, CUDA_R_16F, kHD, 0,
                                                   Qh, CUDA_R_16F, H * kHD, kHD, &zero, Sh, CUDA_R_32F, ldp,
                                                   (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                        CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-mx: attention scores GEMM failed\n");
                        std::exit(1);
                    }
                }
                k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0, scale);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Vh = vbase + (size_t) kvh * ctx * kHD;
                    const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                    float* Oh = att + ((size_t) t0 * H + kvh * G) * kHD;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, kHD, nt, P, &one, Vh, CUDA_R_16F, kHD, 0,
                                                   Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                   H * kHD, kHD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                        CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-mx: attention values GEMM failed\n");
                        std::exit(1);
                    }
                }
            }
        } else {
            const int rows_per = att_rows / H;
            for (int t0 = 0; t0 < T; t0 += rows_per) {
                const int nt = std::min(rows_per, T - t0);
                const int nc = graph_mode ? n_chunk
                                          : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                k_attn_part<<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q, kvs[li], pos0, t0, H, HKV, T, scale,
                                                                     n_chunk, pacc, pm, pl);
                k_attn_merge<<<nt * H, kHD, 0, st>>>(pacc, pm, pl, pos0, t0, H, n_chunk, att);
            }
        }
        if (prof.fine) prof.mark(has_idx && !force_dense && p0 + T > kDenseUpTo ? "a_sparse" : "a_dense");
        if (T == 1 && (int) li == dump_attn_layer && !force_dense && p0 + T > kDenseUpTo) dump_attention(li, p0);
        mm(L.wo, att, f, T);
    }

    // ---- MoE: h -> f
    void moe(const Layer& L, size_t li, int T) {
        const int N = (int) m->n_embd, Kp = (int) m->top_k, NE = (int) m->n_expert;
        const int64_t n_ent = (int64_t) T * Kp;
        f32mm(L.router, h, logits_r, N, NE, T);
        k_router<<<T, 32, 0, st>>>(logits_r, L.bias, NE, Kp, ids, ewts);
        prof.mark("router");
        const bool hyb = L.n_slots < NE;
        quant(h, N, T, xq);   // the experts' input (q8_1)
        if (hyb) {
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
            // in a decode graph: an external event node, so that the host's wait sees the replay
            if (capturing) CK(cudaEventRecordWithFlags(ev_route, st, cudaEventRecordExternal));
            else CK(cudaEventRecord(ev_route, st));
        }
        k_group<<<1, kThreads, 0, st>>>(ids, (int) n_ent, Kp, NE, hyb ? hy[li].slot_of_d : nullptr, L.blobs, L.stride, grp_ptr,
                                        grp_start, n_groups, ent_dst, ent_tok);
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, std::min<int64_t>(n_ent, NE), n_ent,
                                 xq, escr, eparts, st);
        // the shared expert
        ffn(L.sg, L.su, L.sd, h, shy, T);
        if (hyb) {
            if (capturing) {   // a decode graph's segment ends here: the host runs the CPU part between segments
                seg_end();
                moe_cpu(li, T);
                seg_begin();
            } else if (!prof.on) moe_cpu(li, T);
            else {   // the GPU part first, then the CPU part alone: their own times
                prof.mark("moe_gpu");
                moe_cpu(li, T);
                prof.mark("moe_cpu");
            }
        } else {
            prof.mark("moe_gpu");
        }
        k_combine<<<dim3(T, (unsigned) ((m->n_embd + kThreads - 1) / kThreads)), kThreads, 0, st>>>(eparts, nullptr, ewts, Kp, shy, f, N);
    }
    /// The entries without a VRAM slot on the CPU pool (read in place from the mapped GGUF), merged into eparts.
    void moe_cpu(size_t li, int T) {
        const int N = (int) m->n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = m->layers[li];
        const auto tw = Clock::now();
        CK(cudaEventSynchronize(ev_route));
        const auto tp = Clock::now();
        t_wait_route += std::chrono::duration<double, std::milli>(tp - tw).count();
        jobs.clear();
        std::vector<char> quantized((size_t) T, 0);
        std::map<int, size_t> open;
        int n_miss = 0;
        if (route_stats && T == 1) {
            if (prev_ids.size() < m->layers.size()) { prev_ids.resize(m->layers.size()); hist_ids.resize(m->layers.size()); }
            std::vector<int> cur(h_ids, h_ids + n_ent);
            if (!prev_ids[li].empty())
                for (int x : cur) rs_shared += std::count(prev_ids[li].begin(), prev_ids[li].end(), x);
            if (!prev_ids[li].empty()) rs_total += n_ent;
            auto& hh = hist_ids[li];
            hh.push_back(cur);
            if (hh.size() > 4) hh.erase(hh.begin());
            if (hh.size() == 4) {   // distinct experts over a window of 4 tokens
                std::vector<int> u;
                for (auto& v : hh) u.insert(u.end(), v.begin(), v.end());
                std::sort(u.begin(), u.end());
                rs_win4 += std::unique(u.begin(), u.end()) - u.begin();
                ++rs_win4_n;
            }
            prev_ids[li] = cur;
        }
        for (int e = 0; e < n_ent; ++e) {
            const int x = h_ids[e];
            H.score[(size_t) x] += 1.0;
            h_miss[e] = H.slot_of[(size_t) x] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(L.efmt[(size_t) x], h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(x);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = L.src_g + (size_t) x * L.src_gsz;
                j.fmt = &L.efmt[(size_t) x];
                jobs.push_back(j);
                it = open.insert_or_assign(x, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_miss;
        hy_gpu += n_ent - n_miss;
        if (n_miss == 0) return;
        const auto tr = Clock::now();
        t_jobs += std::chrono::duration<double, std::milli>(tr - tp).count();
        pool->run_split_multi_native(L.efmt[0], jobs.data(), (int) jobs.size());
        t_pool += ms_since(tp);
        t_run += ms_since(tr);
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
    }
    /// The MoE block of a prompt chunk on MMQ over the staged layer `stg` ([gate | up | down] tensors as in the file).
    void moe_stream(const Layer& L, int T, const uint8_t* stg) {
        const int N = (int) m->n_embd, Kp = (int) m->top_k, NE = (int) m->n_expert, FF = (int) m->n_ff_exp;
        const int64_t n_ent = (int64_t) T * Kp;
        f32mm(L.router, h, logits_r, N, NE, T);
        k_router<<<T, 32, 0, st>>>(logits_r, L.bias, NE, Kp, ids, ewts);
        CK(cudaMemsetAsync(rcnt, 0, (size_t) NE * 4, st));
        k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
        k_route_scan<<<1, 32, 0, st>>>(rcnt, NE, rbounds, rcur, rmax);
        k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
        CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        const int64_t maxr = hpin[10];
        prof.mark("router");
        MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
        MMQ::Product p;
        p.type = L.L.gu_type; p.w_rows = FF; p.w_cols = N; p.expert_bytes = L.src_gsz; p.n = NE; p.xq = mq_x;
        p.bounds = rbounds; p.ids = ident; p.total_rows = n_ent; p.max_rows = maxr; p.ld_dst = 2 * FF;
        p.w = stg;
        p.dst = mq_gu;
        mctx->run(p, st);
        p.w = stg + streamer.gsz_all;
        p.dst = mq_gu + FF;
        mctx->run(p, st);
        MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
        MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
        MMQ::Product d;
        d.w = stg + 2 * streamer.gsz_all; d.type = L.L.d_type; d.w_rows = N; d.w_cols = FF; d.expert_bytes = L.src_dsz;
        d.n = NE; d.xq = mq_hq; d.bounds = rbounds; d.ids = ident; d.total_rows = n_ent; d.max_rows = maxr; d.dst = eparts;
        d.ld_dst = N;
        mctx->run(d, st);
        ffn(L.sg, L.su, L.sd, h, shy, T);
        k_combine<<<dim3(T, (unsigned) ((m->n_embd + kThreads - 1) / kThreads)), kThreads, 0, st>>>(eparts, rinv, ewts, Kp, shy, f, N);
        prof.mark("moe_mmq");
    }
    /// A layer's attention half over T rows of X at positions p0.. (device *pos0): X += mHC(attention).
    void layer_attn(const Layer& L, size_t li, int T, int64_t p0) {
        const int N = (int) m->n_embd;
        const int ns = hc_mix(L.hca_fn, X, kMix, T);
        k_hc_pre<<<T, 1024, 0, st>>>(X, raw, ns, L.hca_scale, L.hca_base, L.attn_norm, h, post, comb, N, m->hc_iters,
                                        m->eps_hc, m->eps);
        prof.mark("hc");
        attention(L, li, T, p0);
        prof.mark("attn");
        k_hc_post<<<grid1((int64_t) T * N), kThreads, 0, st>>>(X, f, post, comb, N, (int64_t) T * N);
    }
    /// The FFN half: X += mHC(FFN); `stg`: the staged experts (the streamed prompt), else slots + the CPU pool.
    void layer_ffn(const Layer& L, size_t li, int T, const uint8_t* stg) {
        const int N = (int) m->n_embd;
        const int ns = hc_mix(L.hcf_fn, X, kMix, T);
        k_hc_pre<<<T, 1024, 0, st>>>(X, raw, ns, L.hcf_scale, L.hcf_base, L.ffn_norm, h, post, comb, N, m->hc_iters,
                                        m->eps_hc, m->eps);
        prof.mark("hc");
        if (L.dense) {
            ffn(L.fg, L.fu, L.fd, h, f, T);
            prof.mark("dense_ffn");
        } else if (stg) {
            moe_stream(L, T, stg);
        } else {
            moe(L, li, T);
        }
        k_hc_post<<<grid1((int64_t) T * N), kThreads, 0, st>>>(X, f, post, comb, N, (int64_t) T * N);
        prof.mark("hc");
    }
    /// The head over `heads` rows of X ending at Xend (exclusive): logits and greedy picks.
    void head(const float* Xh, int heads, bool mtp = false) {
        const int N = (int) m->n_embd;
        const int ns = hc_mix(mtp ? m->mtp_hc_fn : m->head_fn, Xh, kHC, heads);
        k_hc_head<<<heads, kThreads, 0, st>>>(Xh, raw, ns, mtp ? m->mtp_hc_scale : m->head_scale, mtp ? m->mtp_hc_base : m->head_base,
                                              mtp ? m->mtp_fnorm : m->out_norm, hfin, N, m->eps_hc, m->eps);
        mm(m->head, hfin, logits, heads);
        k_argmax<<<dim3(kArgBlocks, heads), kThreads, 0, st>>>(logits, (int) m->n_vocab, pick, amv, ami, false);
        k_argmax<<<dim3(1, heads), kThreads, 0, st>>>(logits, kArgBlocks, pick, amv, ami, true);
        prof.mark("head");
    }
    // ---- MTP (the vendor's MaxwellMoeMTPLayer): rows t take the main model's final streams at t and the embedding of
    // token t + 1; the layer (dense attention over its own KV cache at position t, MoE, mHC) and its hc_head + final
    // norm + the shared LM head predict token t + 2
    float* Xlast = nullptr;
    int32_t* pos_m = nullptr;
    long long mtp_drafts = 0, mtp_accepted = 0, mtp_skipped = 0, mtp_passes = 0;
    bool want_top_p = false;
    float *top_part = nullptr, *top_p = nullptr;   // the draft's top probability (--mtp-min-p)
    float draft_prob() {
        CK(cudaMemcpyAsync(hpin + 30, top_p, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        float v;
        std::memcpy(&v, hpin + 30, 4);
        return v;
    }
    void mtp_forward(const float* Xin, const int32_t* next_toks, int R, int64_t p_first, const uint8_t* stg, bool draft) {
        const int N = (int) m->n_embd;
        const size_t li = (size_t) m->n_layers;
        const Layer& L = m->layers[li];
        embed(next_toks, R);
        k_rms<<<R, kThreads, 0, st>>>(emb, m->mtp_enorm, mtp_e, N, m->eps);
        mm(m->e_proj, mtp_e, mtp_eo, R);
        k_rms<<<R * kHC, kThreads, 0, st>>>(Xin, m->mtp_hnorm, mtp_hs, N, m->eps);
        mm(m->h_proj, mtp_hs, mtp_ho, R * kHC);
        k_mtp_mix<<<grid1((int64_t) R * N), kThreads, 0, st>>>(mtp_ho, mtp_eo, Xm, N, (int64_t) R * N);
        float* Xs = X;
        int32_t* ps = pos0;
        X = Xm;
        hpin[20] = (int32_t) p_first;
        CK(cudaMemcpyAsync(pos_m, hpin + 20, 4, cudaMemcpyHostToDevice, st));
        pos0 = pos_m;
        if (prof.on) {   // the MTP layer's attention on its own (dense over the whole context, the vendor's MTP attention)
            prof.mark("mtp_in");
            prof.on = false;
            layer_attn(L, li, R, p_first);
            CK(cudaStreamSynchronize(st));
            prof.on = true;
            prof.mark("mtp_attn");
        } else {
            layer_attn(L, li, R, p_first);
        }
        layer_ffn(L, li, R, stg);
        if (draft) {
            head(Xm + (size_t) (R - 1) * kHC * N, 1, true);
            if (want_top_p) {
                k_top_prob<<<kArgBlocks, kThreads, 0, st>>>(logits, (int) m->n_vocab, amv, top_part, top_p, false);
                k_top_prob<<<1, kThreads, 0, st>>>(logits, (int) m->n_vocab, amv, top_part, top_p, true);
            }
        }
        X = Xs;
        pos0 = ps;
        CK(cudaStreamSynchronize(st));   // hpin[20] and the embeddings' staging are reused
    }
    size_t stage_al() const { return (stage_bytes + 255) / 256 * 256; }
    size_t xall_bytes(int64_t n) const { return ((size_t) n * kHC * m->n_embd * 4 + 255) / 256 * 256; }
    /// --kv-host: a streamed prompt's staging of one layer's KV, cells [0, end) of every KV head (k8 | v8 | ks | vs).
    size_t kvstage_bytes(int64_t end) const {
        return kv_host ? ((size_t) m->n_head_kv * (size_t) end * (2 * kHD + 8) + 255) / 256 * 256 : 0;
    }
    /// The big set's chunk for a prompt of n tokens: tbig, halved while staging + the streams + the set do not fit the
    /// arena (0: no streamed prompt).
    int stream_chunk(int64_t n) {
        if (!use_stream || n < stream_min) return 0;
        for (int tb = tbig; tb >= 128; tb /= 2)
            if (2 * stage_al() + xall_bytes(n) + rows_bytes(tb) + kvstage_bytes(pos + n) <= arena_bytes &&
                (n + tb - 1) / tb <= (int64_t) (ctx / tsmall + 2))
                return tb;
        return 0;
    }
    /// A whole prompt layer by layer (see Streamer); the last row's logits.
    void process_stream(const int32_t* toks, int64_t n, int tb, bool has_next = false, bool restore = true) {
        const int N = (int) m->n_embd;
        const int64_t p0 = pos;
        const auto t_all = Clock::now();
        prof.begin();
        // the big row set after the staging pair and the streams, in the borrowed arena
        const RowBufs small = save_rows();
        const size_t used = 2 * stage_al() + xall_bytes(n) + rows_bytes(tb) + kvstage_bytes(p0 + n);
        {
            uint8_t* bump = arena + 2 * stage_al() + xall_bytes(n);
            alloc_rows(tb, true, [&](size_t nb) {
                void* p = bump;
                bump += (nb + 255) / 256 * 256;
                return p;
            });
            MMQ::iota(ident, (int64_t) tb * m->top_k, st);
            kvstage = kv_host ? bump : nullptr;
        }
        std::vector<int> seq;
        for (size_t li = 0; li < m->layers.size(); ++li)   // the main MoE layers, then the MTP layer (its prompt pass)
            if (!m->layers[li].dense) seq.push_back((int) li);
        streamer.buf[0] = arena;
        streamer.buf[1] = arena + stage_al();
        float* Xall = (float*) (arena + 2 * stage_al());
        streamer.start(seq);
        const int nch = (int) ((n + tmax - 1) / tmax);
        std::vector<int32_t> pt((size_t) nch);
        for (int c = 0; c < nch; ++c) pt[(size_t) c] = (int32_t) (p0 + (int64_t) c * tmax);
        CK(cudaMemcpyAsync(pos_tab, pt.data(), (size_t) nch * 4, cudaMemcpyHostToDevice, st));
        for (int c = 0; c < nch; ++c) {   // the embeddings into every row's 4 streams
            const int T = (int) std::min<int64_t>(tmax, n - (int64_t) c * tmax);
            embed(toks + (size_t) c * tmax, T);
            k_hc_init<<<grid1((int64_t) T * N), kThreads, 0, st>>>(emb, Xall + (size_t) c * tmax * kHC * N, N, (int64_t) T * N);
            CK(cudaStreamSynchronize(st));   // emb_h / emb are reused by the next chunk
        }
        prof.mark("embed");
        float* Xsave = X;
        int32_t* psave = pos0;
        int si = 0;
        for (size_t li = 0; li < (size_t) m->n_layers; ++li) {
            const Layer& L = m->layers[li];
            const uint8_t* stg = nullptr;
            // --kv-host: the layer's KV cells [0, p0) from the host copy into VRAM staging, read there by every chunk; the
            // new cells go to the staging and the host copy (the first region and the ring are refilled after the prompt)
            const KVP kv_keep = kvs[li];
            if (kv_host) {
                const size_t HKV = (size_t) m->n_head_kv, C = (size_t) ctx, Pc = (size_t) (p0 + n);
                const uint8_t* hb = kvh_base[li];
                KVP t = kv_keep;
                t.k8 = (int8_t*) kvstage;
                t.v8 = (int8_t*) (kvstage + HKV * Pc * kHD);
                t.ks = (float*) (kvstage + 2 * HKV * Pc * kHD);
                t.vs = t.ks + HKV * Pc;
                t.dstride = (int) Pc;
                t.first = 0x7fffffff;
                t.k8r = t.v8r = nullptr;
                t.ksr = t.vsr = nullptr;
                for (size_t g = 0; g < HKV && p0 > 0; ++g) {
                    const size_t hr = g * C, sr = g * Pc, cnt = (size_t) p0;
                    CK(cudaMemcpyAsync(t.k8 + sr * kHD, (const int8_t*) hb + hr * kHD, cnt * kHD, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(t.v8 + sr * kHD, (const int8_t*) (hb + HKV * C * kHD) + hr * kHD, cnt * kHD,
                                       cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(t.ks + sr, (const float*) (hb + 2 * HKV * C * kHD) + hr, cnt * 4, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(t.vs + sr, (const float*) (hb + 2 * HKV * C * kHD) + HKV * C + hr, cnt * 4,
                                       cudaMemcpyHostToDevice, st));
                }
                kvs[li] = t;
            }
            if (!L.dense) {
                const auto tw = Clock::now();
                stg = streamer.take(si);
                stream_wait_ms += ms_since(tw);
                prof.mark("stream_wait");
            }
            for (int c = 0; c < nch; ++c) {
                const int T = (int) std::min<int64_t>(tmax, n - (int64_t) c * tmax);
                X = Xall + (size_t) c * tmax * kHC * N;
                pos0 = pos_tab + c;
                layer_attn(L, li, T, p0 + (int64_t) c * tmax);
                layer_ffn(L, li, T, stg);
            }
            if (!L.dense) streamer.release(si++, st);
            kvs[li] = kv_keep;
        }
        if (kv_host) {   // the VRAM regions from the host copy the prompt wrote
            first_sync(p0, p0 + n);
            ring_upto = -1;
        }
        pos0 = psave;
        head(Xall + (size_t) (n - 1) * kHC * N, 1);
        CK(cudaMemcpyAsync(Xlast, Xall + (size_t) (n - 1) * kHC * N, (size_t) kHC * N * 4, cudaMemcpyDeviceToDevice, st));
        if (m->has_mtp) {   // the MTP layer over every row whose next token is known (its KV cache), staged like the others
            const auto tw = Clock::now();
            const uint8_t* stg = streamer.take(si);
            stream_wait_ms += ms_since(tw);
            for (int c = 0; c < nch; ++c) {
                const int64_t r0 = (int64_t) c * tmax;
                const int T = (int) std::min<int64_t>(tmax, (has_next ? n : n - 1) - r0);
                if (T <= 0) break;
                mtp_forward(Xall + (size_t) r0 * kHC * N, toks + r0 + 1, T, p0 + r0, stg, false);
            }
            streamer.release(si++, st);
        }
        CK(cudaStreamSynchronize(st));
        streamer.join();
        load_rows(small);
        (void) Xsave;
        pos += n;
        const auto tr = Clock::now();
        restore_used = std::max(restore_used, used);
        if (restore) {
            restore_slots(restore_used);
            restore_used = 0;
        }
        restore_ms += ms_since(tr);
        prof.mark("restore");
        if (std::getenv("STRATA_MX_STREAM_LOG"))
            std::fprintf(stderr, "strata-mx: streamed prompt %lld tokens: %.1f ms (copy %.1f ms in the copy thread, waited "
                         "%.1f ms, slots restored in %.1f ms)\n", (long long) n, ms_since(t_all), streamer.copy_ms,
                         stream_wait_ms, restore_ms);
    }
    /// The slots' experts back into the arena after a streamed prompt borrowed it.
    void restore_slots(size_t used) {
        std::vector<uint8_t> blob;
        for (size_t li = 0; li < m->layers.size(); ++li) {
            const Layer& L = m->layers[li];
            if (L.dense || L.n_slots <= 0) continue;
            if (L.blobs >= arena + used) continue;   // past what the prompt borrowed
            const Hy& H = hy[li];
            const int S = (int) std::min<size_t>((size_t) L.n_slots, (size_t) (arena + used - L.blobs + L.stride - 1) / L.stride);
            if (!streamer.bounce) {   // --pin-experts: the parts straight from pinned memory, async
                for (int sl = 0; sl < S; ++sl) {
                    const int x = H.exp_in_slot[(size_t) sl];
                    uint8_t* d = L.blobs + (size_t) sl * L.stride;
                    CK(cudaMemcpyAsync(d, L.src_g + (size_t) x * L.src_gsz, L.src_gsz, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d + L.L.up_off, L.src_u + (size_t) x * L.src_gsz, L.src_gsz, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(d + L.L.down_off, L.src_d + (size_t) x * L.src_dsz, L.src_dsz, cudaMemcpyHostToDevice, st));
                }
                continue;
            }
            blob.resize(L.stride * 64);
            for (int s0 = 0; s0 < S; s0 += 64) {
                const int nb = std::min(64, S - s0);
                for (int k = 0; k < nb; ++k) expert_blob(L, H.exp_in_slot[(size_t) (s0 + k)], blob.data() + (size_t) k * L.stride);
                CK(cudaMemcpy(L.blobs + (size_t) s0 * L.stride, blob.data(), (size_t) nb * L.stride, cudaMemcpyHostToDevice));
            }
        }
        CK(cudaStreamSynchronize(st));
    }
    void embed(const int32_t* toks, int T) {
        embed_host(toks, T);
        CK(cudaMemcpyAsync(emb, emb_h, (size_t) T * m->n_embd * 4, cudaMemcpyHostToDevice, st));
    }
    /// The rows' embeddings (Q8_0 rows of the mapped file) into the pinned staging emb_h.
    void embed_host(const int32_t* toks, int T) {
        const int N = (int) m->n_embd;
        for (int t = 0; t < T; ++t) {
            const uint8_t* r = m->embd + (size_t) toks[t] * m->embd_row;
            float* o = emb_h + (size_t) t * N;
            for (int b = 0; b < N / 32; ++b) {
                const uint8_t* qb = r + (size_t) b * 34;
                const float d = f16_to_f32((uint16_t) (qb[0] | (qb[1] << 8)));
                for (int j = 0; j < 32; ++j) o[b * 32 + j] = d * (float) (int8_t) qb[2 + j];
            }
        }
    }
    /// The slots to the experts the routing used most (counts halved each time), at most max_swaps replacements.
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_MX_FIXED_SLOTS") != nullptr;
        if (!m->hybrid || off) return;
        const int64_t ne = m->n_expert;
        int left = max_swaps;
        std::vector<uint8_t> blob;
        for (size_t li = 0; li < hy.size() && left > 0; ++li) {
            const Layer& L = m->layers[li];
            if (L.dense) continue;
            Hy& H = hy[li];
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            std::vector<int> victims((size_t) S);
            for (int s = 0; s < S; ++s) victims[(size_t) s] = s;
            std::stable_sort(victims.begin(), victims.end(), [&](int a, int b) {
                return H.score[(size_t) H.exp_in_slot[(size_t) a]] < H.score[(size_t) H.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S && left > 0 && vi < victims.size(); ++i) {
                const int x = order[(size_t) i];
                if (H.slot_of[(size_t) x] >= 0) continue;
                const int s = victims[vi];
                const int y = H.exp_in_slot[(size_t) s];
                if (H.score[(size_t) y] >= H.score[(size_t) x]) break;
                ++vi;
                H.slot_of[(size_t) y] = -1;
                H.slot_of[(size_t) x] = s;
                H.exp_in_slot[(size_t) s] = x;
                blob.assign(L.L.bytes, 0);
                expert_blob(L, x, blob.data());
                CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice, st));
                CK(cudaStreamSynchronize(st));
                --left;
                ++hy_swaps;
                changed = true;
            }
            if (changed)
                CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            for (double& v : H.score) v *= 0.5;
        }
        CK(cudaStreamSynchronize(st));
    }

    /// T rows (host `toks`) at positions pos .. pos + T - 1; `heads` rows from the end get logits (and picks).
    // ---- decode graphs: a decode forward (1 or 2 rows) as CUDA graph segments, one per stretch between the host's CPU
    // expert parts, captured once per key (rows, dense / sparse attention) and replayed (STRATA_MX_NO_GRAPH=1: eager)
    bool use_graphs = std::getenv("STRATA_MX_NO_GRAPH") == nullptr;
    bool capturing = false, graph_mode = false;
    std::map<int, std::vector<cudaGraphExec_t>> gx;
    std::vector<cudaGraphExec_t>* cap_list = nullptr;
    void seg_begin() { CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal)); }
    void seg_end() {
        cudaGraph_t g = nullptr;
        CK(cudaStreamEndCapture(st, &g));
        cudaGraphExec_t e = nullptr;
        CK(cudaGraphInstantiate(&e, g, 0));
        CK(cudaGraphDestroy(g));
        cap_list->push_back(e);
        CK(cudaGraphLaunch(e, st));
    }
    void forward_graph(const int32_t* toks, int T, int heads) {
        const int64_t p0 = pos;
        const bool sparse = !force_dense && p0 + T > kDenseUpTo;
        const int key = T * 4 + heads * 2 + (sparse ? 1 : 0);
        ring_sync(p0);
        hpin[0] = (int32_t) p0;
        embed_host(toks, T);
        auto it = gx.find(key);
        if (it == gx.end()) {
            cap_list = &gx[key];
            capturing = graph_mode = true;
            seg_begin();
            forward_body(T, heads, p0);
            seg_end();
            capturing = graph_mode = false;
            cap_list = nullptr;
        } else {
            const std::vector<cudaGraphExec_t>& segs = it->second;
            size_t si = 0;
            CK(cudaGraphLaunch(segs[si++], st));
            for (size_t li = 0; li < (size_t) m->n_layers; ++li) {
                const Layer& L = m->layers[li];
                if (L.dense || L.n_slots >= (int) m->n_expert) continue;
                moe_cpu(li, T);
                CK(cudaGraphLaunch(segs[si++], st));
            }
        }
        pos += T;
        if (kv_host) ring_upto = pos - 1;
        if (m->hybrid && T <= 4 && pos / kRebalanceEvery != (pos - T) / kRebalanceEvery) rebalance(8);
    }
    /// The GPU work of a decode forward (the host has filled hpin[0] and emb_h).
    void forward_body(int T, int heads, int64_t p0) {
        const int N = (int) m->n_embd;
        CK(cudaMemcpyAsync(pos0, hpin, 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(emb, emb_h, (size_t) T * N * 4, cudaMemcpyHostToDevice, st));
        k_hc_init<<<grid1((int64_t) T * N), kThreads, 0, st>>>(emb, X, N, (int64_t) T * N);
        for (size_t li = 0; li < (size_t) m->n_layers; ++li) {
            layer_attn(m->layers[li], li, T, p0);
            layer_ffn(m->layers[li], li, T, nullptr);
        }
        if (heads > 0) head(X + (size_t) (T - heads) * kHC * N, heads);
    }
    void forward(const int32_t* toks, int T, int heads) {
        if (use_graphs && T <= 4 && heads > 0 && !prof.on && dump_layers == nullptr && inject.empty() && dump_attn_layer < 0) {
            forward_graph(toks, T, heads);
            return;
        }
        const int N = (int) m->n_embd;
        const int64_t p0 = pos;
        ring_sync(p0);
        prof.begin();
        hpin[0] = (int32_t) p0;
        CK(cudaMemcpyAsync(pos0, hpin, 4, cudaMemcpyHostToDevice, st));
        embed(toks, T);
        k_hc_init<<<grid1((int64_t) T * N), kThreads, 0, st>>>(emb, X, N, (int64_t) T * N);
        prof.mark("embed");
        const size_t n_main = (size_t) m->n_layers;
        for (size_t li = 0; li < n_main; ++li) {
            const Layer& L = m->layers[li];
            if (!inject.empty() && li > 0) {
                const size_t W = (size_t) kHC * N, nl = n_main;
                for (int t = 0; t < T; ++t) {
                    const size_t o = ((size_t) (p0 + t) * nl + li - 1) * W;
                    if (o + W <= inject.size())
                        CK(cudaMemcpyAsync(X + (size_t) t * W, inject.data() + o, W * 4, cudaMemcpyHostToDevice, st));
                }
                CK(cudaStreamSynchronize(st));
            }
            layer_attn(L, li, T, p0);
            layer_ffn(L, li, T, nullptr);
            if (dump_layers) {
                const size_t W = (size_t) kHC * N, nl = n_main;
                dump_buf.resize((size_t) T * nl * W);
                CK(cudaStreamSynchronize(st));
                for (int t = 0; t < T; ++t)
                    CK(cudaMemcpy(dump_buf.data() + ((size_t) t * nl + li) * W, X + (size_t) t * W, W * 4, cudaMemcpyDeviceToHost));
                if (li + 1 == nl) std::fwrite(dump_buf.data(), 4, dump_buf.size(), dump_layers);
            }
        }
        if (heads > 0) head(X + (size_t) (T - heads) * kHC * N, heads);
        pos += T;
        if (kv_host) ring_upto = pos - 1;
        if (m->hybrid && T <= 4 && pos / kRebalanceEvery != (pos - T) / kRebalanceEvery) {
            rebalance(8);
            prof.mark("rebalance");
        }
    }
    /// A prompt of n tokens in chunks of tmax; the last row's logits.
    void process(const int32_t* toks, int64_t n) {
        if (dump_layers == nullptr && inject.empty() && (!m->has_mtp || n >= 2)) {
            if (const int tb = stream_chunk(n)) {
                process_stream(toks, n, tb);
                return;
            }
            // too long for the arena at once: windows that fit, each streaming the experts once (layer-major inside)
            int64_t W = 0;
            for (int64_t w = (n + 1) / 2; w >= 1024; w = (w + 1) / 2)
                if (stream_chunk(w)) { W = w; break; }
            if (W > 0) {
                for (int64_t c = 0; c < n; c += W) {
                    const int64_t nw = std::min(W, n - c);
                    process_stream(toks + c, nw, stream_chunk(nw), c + nw < n, c + nw == n);
                }
                return;
            }
        }
        const int N = (int) m->n_embd;
        const int64_t p_start = pos;
        for (int64_t c = 0; c < n; c += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - c);
            forward(toks + c, T, c + T == n ? 1 : 0);
            if (c + T == n) CK(cudaMemcpyAsync(Xlast, X + (size_t) (T - 1) * kHC * N, (size_t) kHC * N * 4, cudaMemcpyDeviceToDevice, st));
            if (m->has_mtp) {   // the MTP layer's KV over the rows whose next token is known
                const int R = c + T == n ? T - 1 : T;
                if (R > 0) {
                    CK(cudaMemcpyAsync(hpin + 12, pick, 4, cudaMemcpyDeviceToHost, st));   // keep the head's pick
                    mtp_forward(X, toks + c + 1, R, p_start + c, nullptr, false);
                }
            }
        }
    }
    void greedyN(int n, int* out) {
        CK(cudaMemcpyAsync(hpin + 8, pick, (size_t) n * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < n; ++i) out[i] = hpin[8 + i];
    }
    void greedy2(int& a, int& b) {
        CK(cudaMemcpyAsync(hpin + 8, pick, 8, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        a = hpin[8];
        b = hpin[9];
    }
    int greedy() {
        CK(cudaMemcpyAsync(hpin + 8, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[8];
    }
    void host_logits_row(int r, std::vector<float>& out) {
        out.resize((size_t) m->n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * m->n_vocab, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int slots, std::string& err) {
    const auto t0 = Clock::now();
    const int64_t ne = m.n_expert;
    size_t total = 0, per_slot = 0;
    int n_moe = 0;
    for (Layer& L : m.layers) {
        if (L.dense) continue;
        total += L.stride * (size_t) ne + kTail;
        per_slot += L.stride;
        ++n_moe;
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) (std::getenv("STRATA_MX_RESERVE_MB") ? std::atoi(std::getenv("STRATA_MX_RESERVE_MB")) : 320) << 20;
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30)) : (fr > reserve ? fr - reserve : 0);
    std::vector<uint8_t> blob;
    const bool all = slots <= 0 && total <= budget;
    // the hybrid mode's host buffers before the slots (they take VRAM too)
    const size_t Tn = (size_t) E.tmax, Nn = (size_t) m.n_embd, Kn = (size_t) m.top_k;
    const size_t extra = all ? 0 : Tn * Kn * Nn * 4 + Tn * Kn * 4 + (size_t) n_moe * ne * 4 + (size_t) n_moe * kTail;
    int S = all ? (int) ne
                : (slots > 0 ? (int) std::min<int64_t>(ne, slots)
                             : (int) std::min<size_t>((size_t) ne, budget > extra ? (budget - extra) / per_slot : 0));
    m.hybrid = !all;
    E.hy.resize(m.layers.size());
    {   // every layer's slots in one arena (a streamed prompt borrows it whole)
        size_t ab = 0;
        for (Layer& L : m.layers)
            if (!L.dense) ab += L.stride * (size_t) S + kTail;
        CK(cudaMalloc(&E.arena, ab));
        CK(cudaMemset(E.arena, 0, ab));
        E.arena_bytes = ab;
        size_t off = 0;
        for (Layer& L : m.layers)
            if (!L.dense) {
                L.blobs = E.arena + off;
                off += L.stride * (size_t) S + kTail;
            }
    }
    for (Layer& L : m.layers) {
        if (L.dense) continue;
        const size_t li = (size_t) (&L - m.layers.data());
        L.n_slots = S;
        // the initial slots: experts 0 .. S - 1 (the routing counts move them)
        blob.assign(L.stride * (size_t) std::min<int64_t>(S, 64), 0);
        for (int e0 = 0; e0 < S; e0 += 64) {
            const int nb = std::min(64, S - e0);
            for (int k = 0; k < nb; ++k) expert_blob(L, e0 + k, blob.data() + (size_t) k * L.stride);
            CK(cudaMemcpy(L.blobs + (size_t) e0 * L.stride, blob.data(), (size_t) nb * L.stride, cudaMemcpyHostToDevice));
        }
        // the CPU layout per expert: gate / up / down rows read in place from the mapping
        L.efmt.resize((size_t) ne);
        for (int64_t e = 0; e < ne; ++e) {
            C::NativeFmt& F = L.efmt[(size_t) e];
            if (!C::native_fmt(L.L.gu_type, L.L.d_type, m.n_embd, m.n_ff_exp, F, err)) return false;
            const uint8_t* gb = L.src_g + (size_t) e * L.src_gsz;
            F.up_off = (size_t) ((L.src_u + (size_t) e * L.src_gsz) - gb);
            F.down_off = (size_t) ((L.src_d + (size_t) e * L.src_dsz) - gb);
        }
        if (m.hybrid) {
            Engine::Hy& H = E.hy[li];
            H.slot_of.assign((size_t) ne, -1);
            H.exp_in_slot.assign((size_t) S, 0);
            for (int s = 0; s < S; ++s) {
                H.slot_of[(size_t) s] = s;
                H.exp_in_slot[(size_t) s] = s;
            }
            H.score.assign((size_t) ne, 0.0);
            CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
            CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        }
    }
    if (m.hybrid) {
        CK(cudaHostAlloc((void**) &E.h_h, Tn * Nn * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &E.cpu_out, Tn * Kn * Nn * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &E.h_ids, Tn * Kn * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &E.h_miss, Tn * Kn * 4, cudaHostAllocDefault));
        CK(cudaMalloc(&E.ecpu, Tn * Kn * Nn * 4));
        CK(cudaMalloc(&E.d_miss, Tn * Kn * 4));
        E.acts = (uint8_t*) std::malloc(Tn * C::kNativeActBytes + 64);
        CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming));
        E.pool = std::make_unique<C::ExpertPool>(threads);
        // the streamed prompt: MMQ for the layers' types, its routing buffers, and the stage size
        const Layer* L0 = nullptr;
        for (const Layer& L : m.layers)
            if (!L.dense) { L0 = &L; break; }
        const bool mmq_ok = MMQ::built() && MMQ::supported(L0->L.gu_type) && MMQ::supported(L0->L.d_type) &&
                            MMQ::fits(L0->L.gu_type, m.n_ff_exp) && MMQ::fits(L0->L.d_type, m.n_embd);
        E.use_stream = mmq_ok && std::getenv("STRATA_MX_NO_STREAM") == nullptr;
        if (!mmq_ok)
            std::fprintf(stderr, "strata-mx: no streamed prompt: MMQ built %d, supports %d/%d, fits %d/%d\n", (int) MMQ::built(),
                         (int) MMQ::supported(L0->L.gu_type), (int) MMQ::supported(L0->L.d_type),
                         (int) MMQ::fits(L0->L.gu_type, m.n_ff_exp), (int) MMQ::fits(L0->L.d_type, m.n_embd));
        if (const char* v = std::getenv("STRATA_MX_STREAM_MIN")) E.stream_min = std::atoi(v);
        if (E.use_stream) {
            const size_t T = (size_t) E.tmax, ne_ = T * Kn;
            E.streamer.m = &m;
            E.streamer.gsz_all = L0->src_gsz * (size_t) ne;
            E.streamer.dsz_all = L0->src_dsz * (size_t) ne;
            E.stage_bytes = 2 * E.streamer.gsz_all + E.streamer.dsz_all + kTail;
            CK(cudaStreamCreateWithFlags(&E.streamer.cst, cudaStreamNonBlocking));
            for (int b = 0; b < 2; ++b) CK(cudaEventCreateWithFlags(&E.streamer.used[b], cudaEventDisableTiming));
            if (const char* v = std::getenv("STRATA_MX_PIECES")) E.streamer.kPieces = std::max(2, std::min(16, std::atoi(v)));
            if (const char* v = std::getenv("STRATA_MX_COPY_THREADS")) E.streamer.kCopyThreads = std::max(1, std::min(16, std::atoi(v)));
            if (const char* v = std::getenv("STRATA_MX_PIECE_MB")) E.streamer.kPiece = (size_t) std::max(1, std::atoi(v)) << 20;
            for (int k = 0; k < E.streamer.kPieces; ++k) {
                CK(cudaHostAlloc((void**) &E.streamer.pin[k], E.streamer.kPiece, cudaHostAllocDefault));
                CK(cudaEventCreateWithFlags(&E.streamer.pin_ev[k], cudaEventDisableTiming));
                CK(cudaEventRecord(E.streamer.pin_ev[k], E.streamer.cst));
            }
            E.mctx = std::make_unique<MMQ::Context>();
            E.rcnt = dalloc_t<int32_t>((size_t) ne);
            E.rcur = dalloc_t<int32_t>((size_t) ne);
            E.rbounds = dalloc_t<int32_t>((size_t) ne + 1);
            E.rmax = dalloc_t<int32_t>(1);
            E.pos_tab = dalloc_t<int32_t>((size_t) (E.ctx / E.tsmall + 2));
            CK(cudaDeviceSynchronize());
            const int64_t max_tok = (int64_t) ((E.arena_bytes > 2 * E.stage_al() ? E.arena_bytes - 2 * E.stage_al() : 0) /
                                               ((size_t) kHC * m.n_embd * 4));
            std::fprintf(stderr, "strata-mx: prompts of >= %d tokens streamed layer by layer (2 x %.2f GiB staging in the "
                         "slots' VRAM, up to %lld tokens)\n", E.stream_min, (double) E.stage_bytes / (1 << 30),
                         (long long) max_tok);
        }
    }
    CK(cudaDeviceSynchronize());
    CK(cudaMemGetInfo(&fr, &tot));
    if (all)
        std::fprintf(stderr, "strata-mx: every routed expert in VRAM (%.2f GiB) in %.1f s\n", (double) total / (1 << 30),
                     ms_since(t0) / 1000.0);
    else
        std::fprintf(stderr, "strata-mx: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), the rest read in place from "
                     "the mapped file on %d CPU threads; %.1f s, %.2f GiB of VRAM left\n", S, (long long) ne,
                     (double) S * per_slot / (1 << 30), E.pool->workers() + (E.pool->host_works() ? 1 : 0),
                     ms_since(t0) / 1000.0, (double) fr / (1 << 30));
    return true;
}

/// --pin-experts: every MoE layer's three expert tensors copied once into pinned host memory (the layout of the file),
/// which the CPU pool and the streamed prompt's DMA then read: the prompt's copies need no bounce through pinned
/// pieces (one RAM read per byte instead of three: 14 GB/s instead of ~8 on PCIe 4.0 x8).  Locks the experts' size of
/// RAM (the mapped file's pages stay reclaimable page cache).  Before place_experts (the CPU layouts point at it).
bool pin_experts(Model& m) {
    const auto t0 = Clock::now();
    size_t total = 0;
    const size_t ne = (size_t) m.n_expert;
    for (Layer& L : m.layers) {
        if (L.dense) continue;
        const size_t gb = L.src_gsz * ne, db = L.src_dsz * ne;
        uint8_t* a = nullptr;
        if (cudaHostAlloc((void**) &a, 2 * gb + db, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata-mx: --pin-experts: no pinned host memory for %.2f GiB more (%.2f GiB pinned so far); "
                         "the rest stays mapped\n", (double) (2 * gb + db) / (1 << 30), (double) total / (1 << 30));
            return false;
        }
        auto par_copy = [](uint8_t* dst, const uint8_t* src, size_t n) {
            std::thread th[4];
            const size_t part = (n / 4 + 4095) / 4096 * 4096;
            for (int i = 0; i < 4; ++i) {
                const size_t a0 = std::min(n, (size_t) i * part), b0 = std::min(n, a0 + part);
                th[i] = std::thread([=] { if (b0 > a0) std::memcpy(dst + a0, src + a0, b0 - a0); });
            }
            for (auto& t : th) t.join();
        };
        par_copy(a, L.src_g, gb);
        par_copy(a + gb, L.src_u, gb);
        par_copy(a + 2 * gb, L.src_d, db);
        L.src_g = a;
        L.src_u = a + gb;
        L.src_d = a + 2 * gb;
        total += 2 * gb + db;
    }
    std::fprintf(stderr, "strata-mx: --pin-experts: %.2f GiB of experts in pinned host memory in %.1f s\n",
                 (double) total / (1 << 30), ms_since(t0) / 1000.0);
    return true;
}

// ------------------------------------------------------------------------------------------------ serve
std::vector<int32_t> parse_ids(const char* s);
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 64;
    unsigned long long seed = 0;
    bool greedy() const { return temperature <= 0.f && repeat == 1.f && freq == 0.f && present == 0.f; }
};
/// A token from one row's logits (penalties over the last last_n tokens, then temperature / top-k / top-p / min-p).
int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> pr(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (pr[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (pr[i] / sum < s.min_p * (pr[0] / sum)) { keep = i; break; }
        cum += pr[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += pr[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= pr[i]) <= 0) return idx[i];
    return idx[keep - 1];
}

// ---- the held conversation on disk (--kv-file): written at exit (stdin closed, QUIT, SIGTERM) and on SAVE, read
// at start, so an engine stopped while idle continues the conversation without reading it again
struct KvHeader {
    char magic[8] = {'M', 'X', 'K', 'V', '0', '0', '0', '1'};
    int32_t n_layers = 0, ctx = 0, kv8 = 0, hkv = 0;
    int64_t n_live = 0, n_vocab = 0, model_bytes = 0;
};
bool kv_save(Engine& E, const Model& m, const std::string& path, const std::vector<int32_t>& live) {
    const auto t0 = Clock::now();
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    KvHeader hd;
    hd.n_layers = (int32_t) m.layers.size();
    hd.ctx = E.ctx;
    hd.kv8 = E.kv8;
    hd.hkv = (int32_t) m.n_head_kv;
    hd.n_live = (int64_t) live.size();
    hd.n_vocab = m.n_vocab;
    hd.model_bytes = (int64_t) m.file->shard(0).file_size();
    std::fwrite(&hd, sizeof(hd), 1, f);
    std::fwrite(live.data(), 4, live.size(), f);
    const size_t n = live.size(), HKV = (size_t) m.n_head_kv;
    std::vector<uint8_t> buf;
    auto put = [&](const void* dev, size_t bytes) {
        buf.resize(bytes);
        if (bytes) CK(cudaMemcpy(buf.data(), dev, bytes, cudaMemcpyDeviceToHost));
        std::fwrite(buf.data(), 1, bytes, f);
    };
    CK(cudaDeviceSynchronize());
    for (size_t li = 0; li < m.layers.size(); ++li) {
        const KVP& kv = E.kvs[li];
        const bool host = E.kv_host && li < (size_t) m.n_layers;
        for (size_t g = 0; g < HKV; ++g) {
            const size_t row = g * (size_t) E.ctx;
            if (host) {   // straight from the host copy (pinned memory)
                const uint8_t* hb = E.kvh_base[li];
                const size_t C = (size_t) E.ctx;
                std::fwrite(hb + row * kHD, 1, n * kHD, f);
                std::fwrite(hb + HKV * C * kHD + row * kHD, 1, n * kHD, f);
                std::fwrite(hb + 2 * HKV * C * kHD + row * 4, 1, n * 4, f);
                std::fwrite(hb + 2 * HKV * C * kHD + HKV * C * 4 + row * 4, 1, n * 4, f);
            } else if (E.kv8) {
                put(kv.k8 + row * kHD, n * kHD);
                put(kv.v8 + row * kHD, n * kHD);
                put(kv.ks + row, n * 4);
                put(kv.vs + row, n * 4);
            } else {
                put(kv.k + row * kHD, n * kHD * 2);
                put(kv.v + row * kHD, n * kHD * 2);
            }
        }
        put(E.ikc[li], n * kIdxDim * 2);
    }
    const bool ok = std::fclose(f) == 0 && std::rename(tmp.c_str(), path.c_str()) == 0;
    std::fprintf(stderr, "strata-mx: KV of %zu tokens %s %s in %.1f s\n", n, ok ? "saved to" : "NOT saved to", path.c_str(),
                 ms_since(t0) / 1000.0);
    return ok;
}
bool kv_load(Engine& E, const Model& m, const std::string& path, std::vector<int32_t>& live) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const auto t0 = Clock::now();
    KvHeader hd, want;
    want.n_layers = (int32_t) m.layers.size();
    want.ctx = E.ctx;
    want.kv8 = E.kv8;
    want.hkv = (int32_t) m.n_head_kv;
    want.n_vocab = m.n_vocab;
    want.model_bytes = (int64_t) m.file->shard(0).file_size();
    const bool hdr_ok = std::fread(&hd, sizeof(hd), 1, f) == 1 && std::memcmp(hd.magic, want.magic, 8) == 0 &&
                        hd.n_layers == want.n_layers && hd.kv8 == want.kv8 && hd.hkv == want.hkv && hd.n_vocab == want.n_vocab &&
                        hd.model_bytes == want.model_bytes && hd.n_live >= 0 && hd.n_live <= E.ctx;
    if (!hdr_ok) {
        std::fclose(f);
        std::fprintf(stderr, "strata-mx: %s is from another model, context, layer count or KV mode: not used\n", path.c_str());
        return false;
    }
    const size_t n = (size_t) hd.n_live, HKV = (size_t) m.n_head_kv, old_ctx = (size_t) hd.ctx;
    (void) old_ctx;
    live.resize(n);
    bool ok = std::fread(live.data(), 4, n, f) == n;
    std::vector<uint8_t> buf;
    auto get = [&](void* dev, size_t bytes) {
        buf.resize(bytes);
        ok = ok && std::fread(buf.data(), 1, bytes, f) == bytes;
        if (ok && bytes) CK(cudaMemcpy(dev, buf.data(), bytes, cudaMemcpyHostToDevice));
    };
    for (size_t li = 0; li < m.layers.size() && ok; ++li) {
        const KVP& kv = E.kvs[li];
        const bool host = E.kv_host && li < (size_t) m.n_layers;
        for (size_t g = 0; g < HKV; ++g) {
            const size_t row = g * (size_t) E.ctx;   // the cells at their positions in this context's layout
            if (host) {
                uint8_t* hb = E.kvh_base[li];
                const size_t C = (size_t) E.ctx;
                ok = ok && std::fread(hb + row * kHD, 1, n * kHD, f) == n * kHD;
                ok = ok && std::fread(hb + HKV * C * kHD + row * kHD, 1, n * kHD, f) == n * kHD;
                ok = ok && std::fread(hb + 2 * HKV * C * kHD + row * 4, 1, n * 4, f) == n * 4;
                ok = ok && std::fread(hb + 2 * HKV * C * kHD + HKV * C * 4 + row * 4, 1, n * 4, f) == n * 4;
            } else if (E.kv8) {
                get(kv.k8 + row * kHD, n * kHD);
                get(kv.v8 + row * kHD, n * kHD);
                get(kv.ks + row, n * 4);
                get(kv.vs + row, n * 4);
            } else {
                get(kv.k + row * kHD, n * kHD * 2);
                get(kv.v + row * kHD, n * kHD * 2);
            }
        }
        get(E.ikc[li], n * kIdxDim * 2);
    }
    std::fclose(f);
    if (!ok) {
        live.clear();
        std::fprintf(stderr, "strata-mx: %s is truncated: not used\n", path.c_str());
        return false;
    }
    E.pos = (int64_t) n;
    E.first_sync(0, (int64_t) n);
    E.ring_upto = -1;
    std::fprintf(stderr, "strata-mx: KV of %zu tokens restored from %s in %.1f s\n", n, path.c_str(), ms_since(t0) / 1000.0);
    return true;
}

volatile std::sig_atomic_t g_term = 0;
void on_term(int) { g_term = 1; }

/// --serve: the engine protocol of Strata's engines on stdin / stdout.  `GEN <max_new> [key=value ...] <id,id,...>`
/// (keys: temperature top_p top_k min_p penalty_last_n penalty_repeat penalty_freq penalty_present seed) -> `PP` once
/// the prompt is read, `T <id>` per token, `DONE <produced> <prompt> <prompt_ms> <decode_ms> <finish> <accepted>
/// <offered> <reused> <read>`; `STOP` ends the running request; `SAVE` writes the KV file; `QUIT` (or the end of stdin)
/// writes it and exits.  The held conversation's longest common prefix with a prompt is reused; MTP drafts are verified
/// against the main model's own choice at each row (greedy: its pick; sampled: a sample from its distribution), so the
/// output follows the main model exactly.
int serve(Engine& E, const Model& m, const std::string& kv_file, int mtp_draft, float mtp_min_p) {
    std::signal(SIGTERM, on_term);
    std::signal(SIGINT, on_term);
    std::vector<int32_t> live;   // the tokens whose KV the engine holds (cells 0 .. size - 1)
    {   // warm-up before READY (kernel modules, cuBLAS, the decode graphs, a streamed prompt), then a fresh state: the
        // first request does not pay for them.  Before the KV file is read, so the held conversation is not touched.
        const auto tw = Clock::now();
        std::vector<int32_t> toks(320);
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = (int32_t) (1000 + (i * 7919) % 50000);
        E.reset();
        E.process(toks.data(), 16);
        E.reset();
        E.process(toks.data(), (int64_t) toks.size());   // the streamed path when it streams (>= stream_min)
        int32_t w[4] = {E.greedy(), 202, 202, 202};
        for (int T = 1; T <= std::min(4, (m.has_mtp ? mtp_draft : 0) + 1); ++T) E.forward(w, T, T);
        if (m.has_mtp) {
            E.mtp_forward(E.Xlast, w, 1, E.pos - 1, nullptr, true);
            E.mtp_forward(E.X, w, 2, E.pos - 2, nullptr, true);
        }
        E.greedy();
        E.reset();
        std::fprintf(stderr, "strata-mx: warm-up in %.1f s\n", ms_since(tw) / 1000.0);
    }
    if (!kv_file.empty()) kv_load(E, m, kv_file, live);
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::thread rd([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "STOP") {
                stop.store(true);
                continue;
            }
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(line);
            cv.notify_all();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_all();
    });
    rd.detach();
    const int N = (int) m.n_embd;
    std::printf("INFO engine=strata-mx family=maxwell-moe kv=%s mtp=%d restored=%zu expert_slots=%d\n", E.kv8 ? "q8" : "f16",
                m.has_mtp ? mtp_draft : 0, live.size(), m.layers[(size_t) m.n_dense].n_slots);
    std::printf("READY %d stop\n", E.ctx);
    std::fflush(stdout);
    std::vector<float> lg;
    auto is_eos = [](int t) { return t == 200001 || t == 2; };
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::milliseconds(200), [&] { return !q.empty() || eof; });
            if (g_term) break;
            if (q.empty()) {
                if (eof) break;
                continue;
            }
            line = q.front();
            q.pop_front();
        }
        if (line == "QUIT") break;
        if (line == "SAVE") {
            if (!kv_file.empty()) kv_save(E, m, kv_file, live);
            std::printf("SAVED %zu\n", live.size());
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* p = const_cast<char*>(line.c_str()) + 4;
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* s0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(s0, (size_t) (p - s0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(s0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        bool bad = max_new < 1 || ids.empty();
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) m.n_vocab;
        if (bad || (int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR bad request: max_new, the ids, or prompt (%zu) + max_new (%lld) over the context (%d)\n", ids.size(),
                        max_new, E.ctx);
            std::fflush(stdout);
            continue;
        }
        stop.store(false);
        // the held conversation's common prefix (at least the last prompt token is read again: its logits)
        size_t reuse = 0;
        while (reuse < live.size() && reuse < ids.size() && live[reuse] == ids[reuse]) ++reuse;
        if (reuse >= ids.size()) reuse = ids.size() - 1;
        live.resize(reuse);
        E.pos = (int64_t) reuse;
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        E.process(ids.data() + reuse, fresh);
        live.insert(live.end(), ids.begin() + (long) reuse, ids.end());
        const double prompt_ms = ms_since(t0);
        std::printf("PP %lld %lld %.0f %.1f\n", (long long) fresh, (long long) fresh, prompt_ms, fresh * 1000.0 / std::max(1e-9, prompt_ms));
        std::fflush(stdout);
        std::mt19937_64 rng(smp.seed ? smp.seed : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count());
        std::vector<int32_t> hist = ids;
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        const auto d0 = Clock::now();
        auto emit = [&](int t) {   // false: the answer ends here
            std::printf("T %d\n", t);
            std::fflush(stdout);
            ++produced;
            hist.push_back(t);
            if (is_eos(t)) { finish = "stop"; return false; }
            if (produced >= max_new) return false;
            if (stop.load() || g_term) { finish = "cancel"; return false; }
            return true;
        };
        auto pick_row = [&](int r, const int* greedy_picks) {
            if (smp.greedy()) return greedy_picks[r];
            E.host_logits_row(r, lg);
            return sample(lg, smp, hist, rng);
        };
        int gp0[1];
        if (smp.greedy()) gp0[0] = E.greedy();
        int x1 = pick_row(0, gp0);
        if (m.has_mtp) {
            const float* min = E.Xlast;
            int32_t mtok[4] = {x1, 0, 0, 0};
            int R = 1;
            int64_t t = (int64_t) ids.size() - 1;
            if (emit(x1))
                for (;;) {
                    if (E.pos + mtp_draft + 1 > E.ctx) break;
                    int32_t drafts[3];
                    int nd = 0;
                    E.mtp_forward(min, mtok, R, t - R + 1, nullptr, true);
                    int d = E.greedy();
                    bool sure = mtp_min_p <= 0.f || E.draft_prob() >= mtp_min_p;
                    int rows_out = R;
                    while (sure) {
                        drafts[nd++] = d;
                        if (nd >= mtp_draft) break;
                        const float* cin = E.Xm + (size_t) (rows_out - 1) * kHC * N;
                        int32_t ctok = d;
                        E.mtp_forward(cin, &ctok, 1, t + nd, nullptr, true);
                        rows_out = 1;
                        d = E.greedy();
                        sure = mtp_min_p <= 0.f || E.draft_prob() >= mtp_min_p;
                    }
                    const int T = nd + 1;
                    int32_t win[4] = {x1, 0, 0, 0};
                    for (int i = 0; i < nd; ++i) win[i + 1] = drafts[i];
                    E.forward(win, T, T);
                    int gp[4] = {0, 0, 0, 0};
                    if (smp.greedy()) E.greedyN(T, gp);
                    // row r's choice, sampled after the drafts before it are in the history (their verify is exact)
                    int acc = 0, nxt = 0;
                    for (;;) {
                        nxt = pick_row(acc, gp);
                        if (acc < nd && nxt == drafts[acc]) {
                            hist.push_back(drafts[acc]);
                            ++acc;
                            continue;
                        }
                        break;
                    }
                    for (int i = 0; i < acc; ++i) hist.pop_back();   // emit() pushes them again
                    live.push_back(x1);
                    for (int i = 0; i < acc; ++i) live.push_back(drafts[i]);
                    E.pos -= nd - acc;
                    offered += nd;
                    accepted += acc;
                    bool more = true;
                    for (int i = 0; i < acc && more; ++i) more = emit(drafts[i]);
                    if (more) more = emit(nxt);
                    if (!more) break;
                    for (int j = 0; j <= acc; ++j) mtok[j] = j < acc ? drafts[j] : nxt;
                    R = acc + 1;
                    min = E.X;
                    t += acc + 1;
                    x1 = nxt;
                }
        } else {
            int x = x1;
            while (emit(x) && E.pos + 1 <= E.ctx) {
                E.forward(&x, 1, 1);
                live.push_back(x);
                int g1[1];
                if (smp.greedy()) g1[0] = E.greedy();
                x = pick_row(0, g1);
            }
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu %lld\n", produced, ids.size(), prompt_ms, decode_ms, finish, accepted,
                    offered, reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-mx serve: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), %lld generated in "
                     "%.0f ms (%.1f tok/s), drafts accepted %lld of %lld (%s)\n", ids.size(), reuse, (long long) fresh, prompt_ms,
                     fresh * 1000.0 / std::max(1e-9, prompt_ms), produced, decode_ms, produced * 1000.0 / std::max(1e-9, decode_ms),
                     accepted, offered, finish);
        if (g_term) break;
    }
    if (!kv_file.empty()) kv_save(E, m, kv_file, live);
    std::fflush(stdout);
    std::_Exit(0);   // the reader thread may still block on stdin
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        v.push_back((int32_t) std::atoi(s));
        while (*s && *s != ',' && *s != ' ') ++s;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    setenv("STRATA_MMQ_KQ", "1", 0);
    setenv("STRATA_IQ_MT_MIN", "1", 0);   // the CPU experts round the same for 1 token as in a group (MTP windows)   // the Q4_K down experts' prompt kernels (libstrata_kq), unless set to 0
    std::string model, dump_full, dump_layers, mtp_path;
    std::vector<int32_t> prompt, forced;
    bool pin = false;
    bool kv_host = false;    // --kv-host: the main layers' KV in pinned host memory (8-bit; automatic from a 64K context)
    std::string kv_mode = "q8";   // --kv q8 (default) | f16
    bool serve_mode = false;
    std::string kv_file;     // --kv-file F: the held conversation's KV written at exit and read at start (--serve)
    int mtp_draft = 1;       // --mtp-draft N: chained MTP drafts per verify window (1-3)
    float mtp_min_p = 0.f;   // draft only when the MTP head's top probability is at least this (0: always)   // f16 | q8 | auto (q8 from a 16K context)
    int n_gen = 32, ctx = 4096, chunk = 256, threads = 0, slots = 0, bench_pp = 0, bench_tg = 0;
    double expert_vram = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-mx: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--tokens") prompt = parse_ids(next());
        else if (a == "--tokens-file") {
            FILE* tf = std::fopen(next(), "rb");
            std::string buf;
            if (tf) {
                char tmp[65536];
                size_t r;
                while ((r = std::fread(tmp, 1, sizeof(tmp), tf)) > 0) buf.append(tmp, r);
                std::fclose(tf);
            }
            prompt = parse_ids(buf.c_str());
        }
        else if (a == "--force") forced = parse_ids(next());
        else if (a == "--n") n_gen = std::atoi(next());
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next());
        else if (a == "--chunk" || a == "--prefill") chunk = std::atoi(next());
        else if (a == "--cpu-threads") threads = std::atoi(next());
        else if (a == "--expert-slots") slots = std::atoi(next());
        else if (a == "--expert-vram") expert_vram = std::atof(next());
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--dump-layers") dump_layers = next();
        else if (a == "--mtp") mtp_path = next();
        else if (a == "--pin-experts") pin = true;
        else if (a == "--kv") kv_mode = next();
        else if (a == "--kv-host") kv_host = true;
        else if (a == "--mtp-min-p") mtp_min_p = (float) std::atof(next());
        else if (a == "--mtp-draft") mtp_draft = std::max(1, std::min(3, std::atoi(next())));
        else if (a == "--serve") serve_mode = true;
        else if (a == "--kv-file") kv_file = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next());
        else if (a == "--bench-tg") bench_tg = std::atoi(next());
        else {
            std::fprintf(stderr, "strata-mx: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: strata-mx --model GGUF (--tokens IDS [--n N] [--force IDS] [--dump-full-logits F] | "
                     "--bench-pp N [--bench-tg N]) [--ctx N] [--chunk T] [--expert-vram GiB] [--expert-slots S] "
                     "[--cpu-threads N]\n");
        return 2;
    }
    auto vram = [](const char* what) {
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        if (std::getenv("STRATA_MX_VRAM_LOG")) std::fprintf(stderr, "strata-mx VRAM: %-28s %.0f MiB used\n", what, (tot - fr) / 1048576.0);
    };
    CK(cudaFree(nullptr));
    vram("context");
    Model m;
    std::string err;
    if (!load(model, m, err)) {
        std::fprintf(stderr, "strata-mx: %s\n", err.c_str());
        return 1;
    }
    if (!mtp_path.empty() && !load_mtp(mtp_path, m, err)) {
        std::fprintf(stderr, "strata-mx: %s\n", err.c_str());
        return 1;
    }
    vram("weights");
    const bool pinned = pin && pin_experts(m);
    Engine E;
    if (pinned) E.streamer.bounce = false;
    E.want_top_p = mtp_min_p > 0.f;
    E.kv8 = kv_mode != "f16";
    E.kv_host = E.kv8 && (kv_host || ctx >= 65536);   // q8 by default: the same KL against the reference, about half the KV VRAM (more slots)
    E.init(m, ctx, chunk);
    vram("engine buffers + KV");
    if (!place_experts(m, E, expert_vram, threads, slots, err)) {
        std::fprintf(stderr, "strata-mx: %s\n", err.c_str());
        return 1;
    }
    vram("expert slots");
    if (serve_mode) return serve(E, m, kv_file, mtp_draft, mtp_min_p);
    if (bench_pp > 0 || bench_tg > 0) {
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = (int32_t) (1000 + (i * 7919) % 50000);
        if (bench_pp > 0) {
            E.reset();
            E.process(toks.data(), std::min<int64_t>(bench_pp, 300));   // warm-up (a streamed prompt when it streams)
            E.prof.ms.clear();
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), bench_pp);
            E.greedy();
            const double ms = ms_since(t0);
            std::fprintf(stderr, "strata-mx: pp%d: %.1f ms, %.1f tok/s\n", bench_pp, ms, bench_pp * 1000.0 / ms);
            E.prof.report("prompt", bench_pp, "token");
        }
        if (bench_tg > 0) {
            if (bench_pp <= 0) {
                E.reset();
                E.process(toks.data(), 1);
            }
            int tok = E.greedy();
            for (int i = 0; i < 8; ++i) {   // warm-up
                E.forward(&tok, 1, 1);
                tok = E.greedy();
            }
            E.prof.ms.clear();
            const long long c0 = E.hy_cpu, g0 = E.hy_gpu;
            E.t_wait_route = E.t_pool = 0;
            const auto t0 = Clock::now();
            for (int i = 0; i < bench_tg; ++i) {
                E.forward(&tok, 1, 1);
                tok = E.greedy();
            }
            const double ms = ms_since(t0);
            std::fprintf(stderr, "strata-mx: tg%d: %.1f ms, %.2f tok/s (expert entries: %.1f%% on the GPU, %lld slot swaps)\n",
                         bench_tg, ms, bench_tg * 1000.0 / ms,
                         100.0 * (double) (E.hy_gpu - g0) / std::max<long long>(1, (E.hy_gpu - g0) + (E.hy_cpu - c0)),
                         E.hy_swaps);
            E.prof.report("decode", bench_tg, "token");
            std::fprintf(stderr, "strata-mx: decode per token: %.2f ms waiting for the GPU up to each router, %.2f ms in the CPU pool "
                         "(incl. its activation quantization), %.2f ms rest\n", E.t_wait_route / bench_tg, E.t_pool / bench_tg,
                         ms / bench_tg - (E.t_wait_route + E.t_pool) / bench_tg);
            if (E.route_stats)
                std::fprintf(stderr, "strata-mx: route stats: %.2f of a token's %lld experts per layer were the previous token's; "
                             "4 consecutive tokens use %.1f distinct experts per layer (%.1f if independent)\n",
                             (double) E.rs_shared / std::max(1LL, E.rs_total) * (double) m.top_k, (long long) m.top_k,
                             (double) E.rs_win4 / std::max(1LL, E.rs_win4_n),
                             (double) m.n_expert * (1.0 - std::pow(1.0 - (double) m.top_k / m.n_expert, 4)));
        }
        return 0;
    }
    if (prompt.empty()) {
        std::fprintf(stderr, "strata-mx: --tokens is required\n");
        return 2;
    }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    if (!dump_layers.empty()) E.dump_layers = std::fopen(dump_layers.c_str(), "wb");
    if (const char* inj = std::getenv("STRATA_MX_INJECT")) {
        FILE* fi = std::fopen(inj, "rb");
        if (fi) {
            std::fseek(fi, 0, SEEK_END);
            E.inject.resize((size_t) std::ftell(fi) / 4);
            std::fseek(fi, 0, SEEK_SET);
            if (std::fread(E.inject.data(), 4, E.inject.size(), fi) != E.inject.size()) E.inject.clear();
            std::fclose(fi);
        }
    }
    std::printf("prompt:");
    for (int t : prompt) std::printf(" %d", t);
    std::printf("\n");
    const auto t0 = Clock::now();
    E.reset();
    E.process(prompt.data(), (int64_t) prompt.size());
    const double pp_ms = ms_since(t0);
    E.prof.report("prompt", (double) prompt.size(), "token");
    std::vector<int32_t> gen;
    std::vector<float> lg;
    E.t_wait_route = E.t_pool = E.t_jobs = E.t_run = 0;
    const auto t1 = Clock::now();
    if (m.has_mtp && forced.empty() && !ff) {
        // MTP drafting (the vendor's MaxwellMoeMTPLayer, chained as its mtp_multistream output allows): draft d1 = token
        // t + 2 from (main streams at t, token t + 1); chained, d2 from (the MTP layer's own streams at t, d1) at t + 1, ...;
        // verify [t + 1, d1 .. dn] in one window of n + 1 rows; keep the accepted prefix and the main model's next token
        // (the plain greedy text), roll the main cache back past it.  The next MTP pass rewrites the MTP cache of the kept
        // positions from the main streams.
        int x1 = E.greedy();
        gen.push_back(x1);
        const float* min = E.Xlast;
        int32_t mtok[4] = {x1, 0, 0, 0};
        int R = 1;
        int64_t t = (int64_t) prompt.size() - 1;
        const int N = (int) m.n_embd;
        while ((int) gen.size() < n_gen) {
            const auto tm = Clock::now();
            int32_t drafts[3];
            int nd = 0;
            E.mtp_forward(min, mtok, R, t - R + 1, nullptr, true);
            int d = E.greedy();
            bool sure = mtp_min_p <= 0.f || E.draft_prob() >= mtp_min_p;
            int rows_out = R;
            while (sure) {
                drafts[nd++] = d;
                if (nd >= mtp_draft) break;
                // chain: the MTP layer's output streams of its last row and the draft, one position on
                const float* cin = E.Xm + (size_t) (rows_out - 1) * kHC * N;
                int32_t ctok = d;
                E.mtp_forward(cin, &ctok, 1, t + nd, nullptr, true);
                rows_out = 1;
                d = E.greedy();
                sure = mtp_min_p <= 0.f || E.draft_prob() >= mtp_min_p;
            }
            E.t_mtp += ms_since(tm);
            const int T = nd + 1;
            int32_t win[4] = {x1, 0, 0, 0};
            for (int i = 0; i < nd; ++i) win[i + 1] = drafts[i];
            E.forward(win, T, T);
            int p[4];
            E.greedyN(T, p);
            int acc = 0;
            while (acc < nd && p[acc] == drafts[acc]) ++acc;
            for (int i = 0; i < acc; ++i) gen.push_back(drafts[i]);
            gen.push_back(p[acc]);
            E.pos -= nd - acc;   // the rows past the kept ones are stale (overwritten by later windows)
            E.mtp_drafts += nd;
            E.mtp_accepted += acc;
            if (nd == 0) ++E.mtp_skipped;
            ++E.mtp_passes;
            for (int j = 0; j <= acc; ++j) mtok[j] = j < acc ? drafts[j] : p[acc];
            R = acc + 1;
            min = E.X;
            t += acc + 1;
            x1 = p[acc];
        }
        gen.resize((size_t) n_gen);
    } else
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        int32_t t = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.forward(&t, 1, 1);
    }
    const double tg_ms = ms_since(t1);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    if (ff) std::fclose(ff);
    if (E.dump_layers) std::fclose(E.dump_layers);
    if (E.mtp_passes)
        std::fprintf(stderr, "strata-mx: MTP (%d draft%s): %lld of %lld drafts accepted (%.1f%%), %lld passes without a draft, "
                     "%.2f tokens per main pass\n", mtp_draft, mtp_draft > 1 ? "s chained" : "", E.mtp_accepted, E.mtp_drafts,
                     100.0 * E.mtp_accepted / std::max(1LL, E.mtp_drafts), E.mtp_skipped, (double) gen.size() / E.mtp_passes);
    std::fprintf(stderr, "strata-mx: decode: %.1f ms waiting for the GPU up to the routers, %.1f ms in the CPU pool (%.1f building "
                 "jobs + quantizing, %.1f running), %.1f ms in MTP forwards, of %.1f ms\n",
                 E.t_wait_route, E.t_pool, E.t_jobs, E.t_run, E.t_mtp, tg_ms);
    std::fprintf(stderr, "strata-mx: prompt %zu tokens in %.1f ms (%.1f tok/s), %d steps in %.1f ms (%.2f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() * 1000.0 / pp_ms, n_gen - 1, tg_ms, (n_gen - 1) * 1000.0 / tg_ms);
    E.prof.report("all", 1, "run");
    return 0;
}
