#!/usr/bin/env python3
"""Analytical backbone GMAC and token trajectories per arm (paper Table 1 GMAC/Tok. columns; Supplementary Table S20).

Convention: per block, with n_attn tokens entering attention and n_mlp tokens entering the MLP (both incl. [CLS]),
C = 384:  QKV 3*n_attn*C^2 + attention scores/values 2*n_attn^2*C + projection n_attn*C^2 + MLP 8*n_mlp*C^2,
plus patch embedding 196*768*384 and head 384*1000. Merge-side operations (similarity matrices, aggregation) are
EXCLUDED. For S3-Merge (merge after a block) n_mlp = n_attn; official ToMe merges between attention and MLP inside
every block, so n_mlp = n_attn - r.
Self-check: reproduces 4.5989 (unmerged) and 3.9316 (9 blocks at 197 tokens, then 3 at 85).
"""
import sys
sys.dont_write_bytecode = True
import argparse, json  # noqa: E401,E402
from pathlib import Path  # noqa: E402

C = 384
SELF_CHECK = {"baseline": 4.5989, "s3_acc": 3.9316}


def gmac(attn_tokens, mlp_tokens):
    macs = 196 * 768 * C + C * 1000
    for na, nm in zip(attn_tokens, mlp_tokens):
        macs += 3 * na * C * C + 2 * na * na * C + na * C * C + 8 * nm * C * C
    return macs / 1e9


def trajectories():
    t = {}
    t["baseline"] = ([197] * 12, [197] * 12)
    for r in (10, 12):
        a = [197 - r * i for i in range(12)]
        t[f"tome_r{r}"] = (a, [n - r for n in a])
    t["s3_acc"] = ([197] * 9 + [85] * 3,) * 2          # late_start=8: blocks 0..8 at 197, merge after block 8
    t["s3_speed"] = ([197] * 7 + [85] * 5,) * 2        # late_start=6: blocks 0..6 at 197, merge after block 6
    return t


def table():
    out = {k: {"attn_tokens": a, "mlp_tokens": m, "final_tokens": m[-1], "gmac": round(gmac(a, m), 4)}
           for k, (a, m) in trajectories().items()}
    for k, v in SELF_CHECK.items():
        assert out[k]["gmac"] == v, (k, out[k]["gmac"], v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analytical GMAC / token trajectories of the Table 1 arms.")
    ap.add_argument("--output-dir", default=None, help="if given, also write gmac.json there (never overwritten)")
    a = ap.parse_args(argv)
    print(json.dumps({"resolved_config": {"script": "gmac.py", "output_dir": a.output_dir, "C": C,
                                          "self_check": SELF_CHECK}}, indent=1))
    out = {"arms": table(), "env": {"python": sys.version.split()[0]}}
    if a.output_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import unified_common as U
        d = Path(a.output_dir)
        d.mkdir(parents=True, exist_ok=True)
        U.write_atomic_json(d / "gmac.json", out)
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
