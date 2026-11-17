"""tools/maxwell/convert_mtp.py - the vendor's MTP layer (mtp_layers.0.*, maxwell-model/Maxwell-V1-Flash safetensors)
as a small GGUF for strata-mx --mtp, quantized with ggml's own quantizer to the formats of the main file's layers:
routed gate/up IQ4_XS, down Q4_K, the dense projections Q8_0; norms, the router, the expert bias and mHC in F32.

    python convert_mtp.py VENDOR_DIR LIBGGML_BASE.so OUT.gguf
"""
import ctypes, json, os, sys
import numpy as np
import torch
from safetensors import safe_open

vendor, libpath, out = sys.argv[1:4]
sys.path.insert(0, os.environ.get("GGUF_PY", "gguf-py"))   # a llama.cpp checkout's gguf-py with maxwell-moe
import gguf
Q = gguf.GGMLQuantizationType

lib = ctypes.CDLL(libpath)
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_void_p]
lib.ggml_quantize_init.argtypes = [ctypes.c_int]

def quant(a: np.ndarray, qt) -> np.ndarray:
    """a [..., n_per_row] f32 -> bytes [..., row_bytes] in ggml type qt."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n_per_row = a.shape[-1]
    nrows = a.size // n_per_row
    bs, ts = gguf.GGML_QUANT_SIZES[qt]
    row_bytes = n_per_row // bs * ts
    dst = np.empty(nrows * row_bytes, dtype=np.uint8)
    lib.ggml_quantize_init(int(qt))
    n = lib.ggml_quantize_chunk(int(qt), a.ctypes.data, dst.ctypes.data, 0, nrows, n_per_row, None)
    assert n == dst.size, (n, dst.size)
    return dst.reshape(*a.shape[:-1], row_bytes)

idx = json.load(open(f"{vendor}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{vendor}/config.json"))
names = [k for k in idx if k.startswith("mtp_layers.0.")]
files = {}
def get(name) -> np.ndarray:
    fn = idx[name]
    if fn not in files: files[fn] = safe_open(f"{vendor}/{fn}", "pt")
    return files[fn].get_tensor(name).float().numpy()

w = gguf.GGUFWriter(out, "maxwell-moe-mtp")
w.add_name("Maxwell-V1-Flash MTP layer")
w.add_uint32("maxwell-moe-mtp.layers", 1)
P = "mtp_layers.0."
D = P + "decoder_layer."
def add(name, arr, qt=None):
    if qt is None:
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
    else:
        w.add_tensor(name, quant(arr, qt), raw_dtype=qt)
    print(f"  {name} {tuple(arr.shape)} {qt.name if qt else 'F32'}", flush=True)

add("mtp.enorm.weight", get(P + "enorm.weight"))
add("mtp.hnorm.weight", get(P + "hnorm.weight"))
add("mtp.e_proj.weight", get(P + "e_proj.weight"), Q.Q8_0)
add("mtp.h_proj.weight", get(P + "h_proj.weight"), Q.Q8_0)
add("mtp.final_norm.weight", get(P + "final_layernorm.weight"))
add("mtp.hc_head_fn.weight", get(P + "hc_head.hc_head_fn"))
add("mtp.hc_head_base.weight", get(P + "hc_head.hc_head_base"))
add("mtp.hc_head_scale.weight", get(P + "hc_head.hc_head_scale"))
B = "blk.0."
add(B + "attn_norm.weight", get(D + "input_layernorm.weight"))
add(B + "ffn_norm.weight", get(D + "post_attention_layernorm.weight"))
for a, b in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
    add(B + b + ".weight", get(D + f"self_attn.{a}.weight"), Q.Q8_0)
add(B + "attn_q_norm.weight", get(D + "self_attn.q_layernorm.weight"))
add(B + "attn_k_norm.weight", get(D + "self_attn.k_layernorm.weight"))
for hc, tag in (("self_attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
    add(B + tag + "_fn.weight", get(D + hc + ".mapping_proj.weight"))
    add(B + tag + "_base.weight", get(D + hc + ".bias"))
    add(B + tag + "_scale.weight", np.concatenate([get(D + hc + ".alpha_pre"), get(D + hc + ".alpha_post"),
                                                    get(D + hc + ".alpha_res")]).reshape(3))
M = D + "block_sparse_moe."
add(B + "ffn_gate_inp.weight", get(M + "gate.weight"))
add(B + "exp_probs_b.bias", get(M + "expert_bias"))
ne = cfg["num_experts"]
for wid, tag, qt in (("w1", "ffn_gate_exps", Q.IQ4_XS), ("w3", "ffn_up_exps", Q.IQ4_XS), ("w2", "ffn_down_exps", Q.Q4_K)):
    add(B + tag + ".weight", np.stack([get(M + f"experts.{x}.{wid}.weight") for x in range(ne)]), qt)
for wid, tag in (("w1", "ffn_gate_shexp"), ("w3", "ffn_up_shexp"), ("w2", "ffn_down_shexp")):
    add(B + tag + ".weight", get(M + f"shared_expert.{wid}.weight"), Q.Q8_0)
used = {n for n in names}
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
