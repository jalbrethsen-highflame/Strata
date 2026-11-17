"""tools/maxwell/check_mbsa.py - strata-mx's block-sparse attention against the vendor's own code
(modeling_maxwell_moe.MaxwellMoeBlockSparseGQAAttention / ...Indexer.project_qk), run on the tensors the engine dumped
(STRATA_MX_DUMP_ATTN=L): the indexer projection from the layer input, the route, and the attention output.

    python check_mbsa.py VENDOR_DIR GGUF DUMP
"""
import importlib.util, os, sys, types
import numpy as np
import torch

vendor, gguf_path, dump = sys.argv[1:4]
sys.path.insert(0, vendor)
spec = importlib.util.spec_from_file_location("mx_vendor", f"{vendor}/modeling_maxwell_moe.py",
                                              submodule_search_locations=[vendor])
mod = importlib.util.module_from_spec(spec)
sys.modules["mx_vendor"] = mod
import configuration_maxwell_moe as cfgmod
sys.modules["mx_vendor.configuration_maxwell_moe"] = cfgmod
spec.loader.exec_module(mod)
cfg = cfgmod.MaxwellMoeConfig.from_json_file(f"{vendor}/config.json")
cfg._attn_implementation = "eager"

sys.path.insert(0, os.environ.get("GGUF_PY", "gguf-py"))   # a llama.cpp checkout's gguf-py with maxwell-moe
import gguf
r = gguf.GGUFReader(gguf_path)
T = {t.name: t for t in r.tensors}
def bf16(name):
    t = T[name]
    a = np.array(t.data).view(np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(a.view(np.float32).reshape([int(x) for x in reversed(t.shape)]).copy())
def f32(name):
    return torch.from_numpy(np.array(T[name].data, dtype=np.float32).copy())

f = open(dump, "rb")
N, HD, IH = 2048, 128, 4
step = 0
while True:
    hdr = np.fromfile(f, np.int32, 4)
    if len(hdr) < 4: break
    li, P, H, HKV = map(int, hdr)
    h = torch.from_numpy(np.fromfile(f, np.float32, N))
    iq = torch.from_numpy(np.fromfile(f, np.float32, IH * HD)).view(IH, HD)
    ik = torch.from_numpy(np.fromfile(f, np.float16, P * HD).astype(np.float32)).view(P, HD)
    q = torch.from_numpy(np.fromfile(f, np.float32, H * HD)).view(H, HD)
    k = torch.from_numpy(np.fromfile(f, np.float16, HKV * P * HD).astype(np.float32)).view(HKV, P, HD)
    v = torch.from_numpy(np.fromfile(f, np.float16, HKV * P * HD).astype(np.float32)).view(HKV, P, HD)
    route = torch.from_numpy(np.fromfile(f, np.int32, IH * 32).astype(np.int64)).view(IH, 32)
    att = torch.from_numpy(np.fromfile(f, np.float32, H * HD)).view(H, HD)
    pos = P - 1
    attn = mod.MaxwellMoeBlockSparseGQAAttention(cfg, layer_idx=li).float()
    # (a) the indexer projection, the vendor's project_qk with the GGUF's BF16 weights, from the dumped layer input
    ix = attn.indexer
    with torch.no_grad():
        ix.linear_q_proj.weight.copy_(bf16(f"blk.{li}.indexer.q_proj.weight"))
        ix.linear_k_proj.weight.copy_(bf16(f"blk.{li}.indexer.k_proj.weight"))
        ix.q_layernorm.weight.copy_(f32(f"blk.{li}.indexer.q_norm.weight"))
        ix.k_layernorm.weight.copy_(f32(f"blk.{li}.indexer.k_norm.weight"))
        rot = mod.MaxwellMoeRotaryEmbedding(config=cfg)
        pid = torch.tensor([[pos]])
        cos, sin = rot(h.view(1, 1, N), pid)
        vq, vk = ix.project_qk(h.view(1, 1, N), cos, sin)   # [1, 4, 1, 128], [1, 1, 128]
    eq = (vq[0, :, 0] - iq).norm() / vq.norm()
    ek = (vk[0, 0] - ik[pos]).norm() / vk.norm()
    # (b) the route from the engine's own indexer q / k (the vendor's _select_blocks + _build_block_aligned_route)
    qpos = torch.tensor([[pos]])
    allowed = attn._allowed_keys(query_positions=qpos, key_len=P, attention_mask=None)
    with torch.no_grad():
        sel = attn._select_blocks(iq.view(1, IH, 1, HD), ik.view(1, P, HD), qpos, allowed)
        vroute = attn._build_block_aligned_route(sel, qpos)[0, 0]   # [groups, 32]
    same_route = all(set(vroute[g].tolist()) == set(route[g].tolist()) for g in range(IH))
    exact_order = torch.equal(vroute, route)
    # (c) the attention output over the engine's route (and over the vendor's, if they differ)
    with torch.no_grad():
        vatt = attn._sparse_attention(q.view(1, H, 1, HD), k.view(1, HKV, P, HD), v.view(1, HKV, P, HD),
                                      route.view(1, 1, IH, 32), allowed)[0, :, 0]
        vatt2 = attn._sparse_attention(q.view(1, H, 1, HD), k.view(1, HKV, P, HD), v.view(1, HKV, P, HD),
                                       vroute.view(1, 1, IH, 32), allowed)[0, :, 0]
        dense = torch.softmax((q.view(HKV, H // HKV, HD) @ k.transpose(1, 2)) * HD ** -0.5, -1) @ v
    ea = (vatt - att).norm() / vatt.norm()
    ea2 = (vatt2 - att).norm() / vatt2.norm()
    ed = (dense.reshape(H, HD) - att).norm() / dense.norm()
    nblk = (P + 127) // 128
    print(f"step {step} layer {li} pos {pos} ({nblk} blocks): indexer q rel err {eq:.2e}, k {ek:.2e}; route same set {same_route}"
          f" (same order {exact_order}); attention rel err {ea:.2e} (vendor's own route {ea2:.2e}); vs dense {ed:.3f}")
    if not same_route:
        for g in range(IH):
            print("  g", g, "engine", sorted(route[g].tolist()), "\n      vendor", sorted(vroute[g].tolist()))
    step += 1
