"""Public smoke tests for the S3-Merge release.

SOFTWARE TESTS ONLY: synthetic tensors verify shapes and invariants; nothing here
is scientific evidence or a benchmark. Run: python tests/test_public.py
Requires torch (+ CUDA optional) and timm; no dataset or checkpoint downloads.
"""
import os
import pathlib
import py_compile
import re
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASS = []


def check(name, ok, detail=""):
    PASS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")


# 1. import test ------------------------------------------------------------
try:
    from models import sr_tome, sr_tome_downstream  # noqa: F401
    from models.adaptive_k import (PackGeometry, allocate_slot_auction,
                                   allocate_uniform, window_scores,
                                   adaptive_merge_pack, unpack_restore)
    from models.safemode import lcb95_speedup
    check("import", True)
except Exception as e:  # pragma: no cover
    check("import", False, repr(e)); sys.exit(1)

# 2. syntax/compile test ----------------------------------------------------
tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.py"],
                         capture_output=True, text=True).stdout.split()
errs = []
for f in tracked:
    try:
        py_compile.compile(str(ROOT / f), doraise=True)
    except Exception as e:
        errs.append((f, repr(e)[:80]))
check("syntax-compile", not errs, str(errs[:3]))

# shared synthetic grid ------------------------------------------------------
torch.manual_seed(0)
B, H, W, C, ws = 2, 16, 16, 64, 8
x = torch.randn(B, H, W, C, device=DEV)
geom = PackGeometry.get(H, W, ws, DEV)
win = (x.view(B, geom.nWh, ws, geom.nWw, ws, C)
       .permute(0, 1, 3, 2, 4, 5).reshape(B, geom.n_windows, geom.tpw, C))
_, s_w = window_scores(win)

# 3. keep-all identity -------------------------------------------------------
K_all = torch.full((B, geom.n_windows), geom.tpw, dtype=torch.long, device=DEV)
packed_all, _, restore_all, _ = adaptive_merge_pack(x, geom, K_all)
full = unpack_restore(packed_all, restore_all, H, W)
check("keep-all-identity", torch.equal(full, x))

# 4. Static-K shape test -----------------------------------------------------
K_st = torch.full((B, geom.n_windows), 48, dtype=torch.long, device=DEV)
p_st, coords, r_st, stats = adaptive_merge_pack(x, geom, K_st)
check("static-k-shapes", p_st.shape == (B, geom.n_windows * 48, C)
      and coords.shape == (B, geom.n_windows * 48, 2))

# 5. deterministic variance-selection repeatability --------------------------
p2, _, r2, _ = adaptive_merge_pack(x, geom, K_st)
check("variance-selection-repeatable", torch.equal(p_st, p2) and torch.equal(r_st, r2))

# 6. Adaptive-K exact budget -------------------------------------------------
T_g = geom.n_windows * 48
K_w = allocate_slot_auction(s_w, K_min=32, K_max=64, T_g=T_g)
check("adaptive-exact-budget", bool((K_w.sum(dim=1) == T_g).all())
      and bool((K_w >= 32).all()) and bool((K_w <= 64).all()))

# 7. packed no-padding -------------------------------------------------------
p_ad, _, r_ad, st_ad = adaptive_merge_pack(x, geom, K_w)
check("packed-no-padding", p_ad.shape[1] == T_g and st_ad["padding_rate"] == 0.0)

# 8. restore-map validity ----------------------------------------------------
hit = torch.zeros(B, T_g, device=DEV)
hit.scatter_add_(1, r_ad, torch.ones_like(r_ad, dtype=hit.dtype))
check("restore-map-valid", int(r_ad.min()) >= 0 and int(r_ad.max()) < T_g
      and bool((hit >= 1).all()))

# 9. PA-disabled and PA-enabled forwards ------------------------------------
try:
    import timm
    m0 = timm.create_model("deit_small_patch16_224", pretrained=False).to(DEV).eval()
    from models.sr_tome import apply_patch
    xb = torch.randn(2, 3, 224, 224, device=DEV)
    m_no = timm.create_model("deit_small_patch16_224", pretrained=False).to(DEV).eval()
    m_no.load_state_dict(m0.state_dict())
    apply_patch(m_no, prop_attn=False)
    m_pa = timm.create_model("deit_small_patch16_224", pretrained=False).to(DEV).eval()
    m_pa.load_state_dict(m0.state_dict())
    apply_patch(m_pa, prop_attn=True)
    with torch.no_grad():
        torch.manual_seed(1); y_no = m_no(xb)
        torch.manual_seed(1); y_pa = m_pa(xb)
    check("pa-off-on-forward", y_no.shape == (2, 1000) and y_pa.shape == (2, 1000)
          and torch.isfinite(y_no).all() and torch.isfinite(y_pa).all()
          and not torch.allclose(y_no, y_pa))
except Exception as e:
    check("pa-off-on-forward", False, repr(e)[:120])

# 10. Safe-Mode baseline fallback rule ---------------------------------------
slow = [2.0 + 0.01 * i for i in range(50)]          # merged clearly slower
fast = [1.0 + 0.01 * i for i in range(50)]
lcb_bad = lcb95_speedup(fast, slow)                  # speedup < 1 -> stay baseline
lcb_good = lcb95_speedup(slow, fast)                 # speedup ~2 -> activate
if isinstance(lcb_bad, tuple): lcb_bad = lcb_bad[0]
if isinstance(lcb_good, tuple): lcb_good = lcb_good[0]
check("safemode-fallback-rule", lcb_bad < 1.02 < lcb_good,
      f"lcb_bad={lcb_bad:.3f} lcb_good={lcb_good:.3f}")

# 11. no private-path scan ---------------------------------------------------
# Generic patterns only; maintainers can add site-specific ones through the environment variable
# S3MERGE_PRIVATE_PATTERNS (a regular expression that is never committed).
_pats = [r"/home/[a-z]", r"/Users/[A-Za-z]", r"[A-Za-z]:\\Users\\"]
if os.environ.get("S3MERGE_PRIVATE_PATTERNS"):
    _pats.append(os.environ["S3MERGE_PRIVATE_PATTERNS"])
pat = re.compile("|".join(_pats))
hits = []
_listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, text=True).stdout.split()
if not _listed:  # not a git checkout: scan the files on disk
    _listed = [str(q.relative_to(ROOT)) for q in ROOT.rglob("*") if q.is_file() and ".git" not in q.parts]
for f in _listed:
    p = ROOT / f
    if p.suffix in (".png", ".pdf"):
        continue
    if pat.search(p.read_text(errors="ignore")):
        hits.append(f)
check("no-private-paths", not hits, str(hits[:5]))

# 12. no credential/secret scan ---------------------------------------------
sec = re.compile(r"(ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|BEGIN [A-Z ]*PRIVATE KEY|password\s*=\s*['\"][^'\"]+)")
hits = []
for f in subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                        capture_output=True, text=True).stdout.split():
    p = ROOT / f
    if p.suffix in (".png", ".pdf"):
        continue
    if sec.search(p.read_text(errors="ignore")):
        hits.append(f)
check("no-secrets", not hits, str(hits[:5]))

n_ok = sum(ok for _, ok in PASS)
print(f"\n{n_ok}/{len(PASS)} tests passed")
sys.exit(0 if n_ok == len(PASS) else 1)
