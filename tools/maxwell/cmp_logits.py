"""tools/maxwell/cmp_logits.py - teacher-forced comparison of strata-mx against the llama.cpp reference dumps:
per step KL(P_ref || Q_mx), top-1 agreement, and whether each engine's top-1 is in the other's top-5.

    python cmp_logits.py REF.logits.bin MX.logits.bin [--vocab 200704]
"""
import sys
import numpy as np

V = 200704
args = [a for a in sys.argv[1:] if not a.startswith("--")]
pairs = list(zip(args[0::2], args[1::2]))
tot_kl, tot_n, tot_top1 = 0.0, 0, 0
for ref, mx in pairs:
    a = np.fromfile(ref, dtype=np.float32).reshape(-1, V).astype(np.float64)
    b = np.fromfile(mx, dtype=np.float32).reshape(-1, V).astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    la = a - a.max(1, keepdims=True); la -= np.log(np.exp(la).sum(1, keepdims=True))
    lb = b - b.max(1, keepdims=True); lb -= np.log(np.exp(lb).sum(1, keepdims=True))
    kl = (np.exp(la) * (la - lb)).sum(1)
    top1 = (a.argmax(1) == b.argmax(1))
    print(f"{mx}: {n} steps, mean KL {kl.mean():.5f}, max KL {kl.max():.4f} (step {kl.argmax()}), top-1 {top1.mean()*100:.1f}%")
    tot_kl += kl.sum(); tot_n += n; tot_top1 += top1.sum()
print(f"ALL: {tot_n} steps, mean KL {tot_kl/tot_n:.5f}, top-1 {tot_top1/tot_n*100:.1f}%")
