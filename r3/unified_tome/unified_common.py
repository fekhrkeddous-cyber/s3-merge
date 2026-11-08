"""Shared code for the unified S3-Merge vs. official ToMe comparison on ImageNet-1K DeiT-S/16
(paper Table 1 and Fig. 5; Supplementary Tables S20-S22).

* Arms (the five Table 1 rows), all built from ONE DeiT-S/16 checkpoint (timm ``deit_small_patch16_224.fb_in1k``):
    baseline   unmerged timm DeiT-S/16
    tome_r10   official ToMe (unmodified release, commit af95e4b1) via ``tome.patch.timm``; model.r = 10; native PA
    tome_r12   same with model.r = 12
    s3_acc     S3-Merge-Acc   = models.sr_tome.apply_patch(target_tokens=84, late_start=8, window_size=7, PA on)
    s3_speed   S3-Merge-Speed = same call with late_start=6
  ``late_start`` is the 0-indexed block whose output is merged (paper L_s = late_start + 1).
* Official ToMe is imported by path or from an installed package; its source files are optionally hash-verified
  against the pinned release BEFORE import, so the exact compared implementation is known.
* ImageNet-1K validation is opened read-only from the Hugging Face arrow shards of ``ILSVRC/imagenet-1k`` (split
  ``validation``) with ``datasets.Dataset.from_file`` (memory-mapped), in shard order, then concatenated. This avoids
  ``datasets.load_dataset``, which touches a lock file in the cache directory. Decoding uses the HF Image feature:
  x['image'].convert('RGB') -> Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet). datasets 4.5.0
  (paper environment) applies ImageOps.exif_transpose when an EXIF Orientation tag is present.
* Timing inputs are synthetic: torch.randn float32 [B, 3, 224, 224] from a CPU torch.Generator seeded 20260718,
  drawn for B = 1, 8, 32 in that order; identical for every arm and every invocation.
"""
import sys
sys.dont_write_bytecode = True

import hashlib, importlib.util, json, os, random, re, socket, subprocess, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# ------------------------------------------------------------------------------------------------ constants
MODEL_NAME = "deit_small_patch16_224"            # timm; its only (default) pretrained tag is fb_in1k
CHECKPOINT_TAG = "deit_small_patch16_224.fb_in1k"
# sha256 of the checkpoint file (model.safetensors of timm/deit_small_patch16_224.fb_in1k) used for the paper
CHECKPOINT_FILE_SHA256 = "1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b"
# state_sha() of the loaded FP32 state dict (key-sorted; see state_sha) for that checkpoint
STATE_SHA256 = "a17d00754ef1e5401cf33cf61af23094c48b8f4a4f5f8898d8ae36b9d81925e1"

TOME_REPO = "https://github.com/facebookresearch/ToMe"
TOME_COMMIT = "af95e4b1befa172dadccd8c81e223b10090f9579"
TOME_FILES = {   # sha256 of the official release files at TOME_COMMIT (paths relative to the checkout root)
    "tome/patch/timm.py": "a26a57a6c4326793e3ad1fc3ffe9a7ea5297f8b8d1a641ac13a719d922202837",
    "tome/merge.py": "eb74b0de74329e991b49d7e2cd0c494b924eb245b8d51f9948dacc8ed8c14e2e",
    "tome/utils.py": "9f32a1fa06743c822f60f7d130807202b56475c2121f08850aed5a9dc9b9dfce",
    "tome/__init__.py": "046b3c02fc00324c8648b4aa2ac766eefa8e1497c5b707035e29d9b9fcb51b89",
    "tome/patch/__init__.py": "f779f85351c8227fdb567764528ae20a1d2deaf225da80850b13720e5c4d08f3",
    "LICENSE": "517acfe79046fca3d989a5a6f8b55b8c3919bb442609bd9ed0e81af6e25deb3a",
}

# Method registry. S3 calls are models/sr_tome.py apply_patch with every argument explicit.
METHODS = {
    "baseline": {"family": "unmerged"},
    "tome_r10": {"family": "official_tome", "r": 10, "prop_attn": True},
    "tome_r12": {"family": "official_tome", "r": 12, "prop_attn": True},
    "s3_acc": {"family": "s3_canonical", "apply_patch": dict(target_tokens=84, late_start=8, window_size=7,
               weighted_merge=False, tau=0.3, content_adaptive=False, prop_attn=True, grid_size=14)},
    "s3_speed": {"family": "s3_canonical", "apply_patch": dict(target_tokens=84, late_start=6, window_size=7,
                 weighted_merge=False, tau=0.3, content_adaptive=False, prop_attn=True, grid_size=14)},
}
ARMS = ("baseline", "s3_acc", "s3_speed", "tome_r10", "tome_r12")   # row order of the generated tables
BATCH_SIZES = (1, 8, 32)

# Per-arm description (Supplementary Table S20); used in the generated tables.
DISCLOSURE = {
    "baseline": {"label": "Unmerged DeiT-S/16", "family": "unmerged", "PA": "n/a", "schedule": "none", "scope": "none",
                 "merge_location": "none", "w": "-", "K_or_r": "-", "final_patch": 196, "final_total": 197,
                 "merge_calls": 0, "gmac_key": "baseline"},
    "s3_acc": {"label": "S3-Merge-Acc (PA on)", "family": "s3_canonical", "PA": "on (log-size mask, blocks 9-11)",
               "schedule": "single-shot", "scope": "windowed 7x7",
               "merge_location": "after block 8 (code late_start=8; 9 blocks at 197)", "w": 7, "K_or_r": "K=21/window",
               "final_patch": 84, "final_total": 85, "merge_calls": 1, "gmac_key": "s3_acc"},
    "s3_speed": {"label": "S3-Merge-Speed (PA on)", "family": "s3_canonical", "PA": "on (log-size mask, blocks 7-11)",
                 "schedule": "single-shot", "scope": "windowed 7x7",
                 "merge_location": "after block 6 (code late_start=6; 7 blocks at 197)", "w": 7, "K_or_r": "K=21/window",
                 "final_patch": 84, "final_total": 85, "merge_calls": 1, "gmac_key": "s3_speed"},
    "tome_r10": {"label": "Official ToMe r=10 (native PA)", "family": "official_tome", "PA": "on (native prop_attn, blocks 1-11)",
                 "schedule": "progressive, every block (between attention and MLP)", "scope": "global bipartite (alternating split)",
                 "merge_location": "blocks 0-11", "w": "-", "K_or_r": "r=10/block", "final_patch": 76, "final_total": 77,
                 "merge_calls": 12, "gmac_key": "tome_r10"},
    "tome_r12": {"label": "Official ToMe r=12 (native PA)", "family": "official_tome", "PA": "on (native prop_attn, blocks 1-11)",
                 "schedule": "progressive, every block (between attention and MLP)", "scope": "global bipartite (alternating split)",
                 "merge_location": "blocks 0-11", "w": "-", "K_or_r": "r=12/block", "final_patch": 52, "final_total": 53,
                 "merge_calls": 12, "gmac_key": "tome_r12"},
}

# ---- timing protocol constants
WARMUP, TIMED, PROFILED_FORWARDS = 30, 200, 20
N_LATENCY_INVOCATIONS, N_PROFILE_INVOCATIONS = 10, 3
SELECT_SEED = 1000                 # torch.manual_seed before every (arm, B) cell
INPUT_SEED = 20260718              # CPU torch.Generator seed of the synthetic timing inputs
INPUT_DRAW_ORDER = (1, 8, 32)      # the inputs are drawn from one generator in this order
INPUT_TENSOR_SHA256 = {            # sha256 of the float32 bytes of the inputs used for the paper
    1: "49167eef0df80828c3ea073066eaaebdc5aca87c3e2c4a3f7587509abd542a74",
    8: "4c455aee2a20fbd51ab739aaf18f7f35ab8c29ade756a9b0f8d02cc87c9f206d",
    32: "11967863a78b6535a7619104329d21e6640d19dd5a1c720d3148c2db1afbe86d",
}
# Randomized arm order per invocation: random.Random(ORDER_SEED); latency invocations 0..9, then profile 0..2, each a
# random.sample of ORDER_KEYS. The paper campaign also timed a sixth configuration ("SW": the PA-off factorial S3 core,
# used for a separate supplementary contrast, not a Table 1 row and not part of this port); it stays in ORDER_KEYS so
# that the draws, and hence the relative order of the five Table 1 arms in every invocation, are exactly those of the
# paper campaign. It is filtered out of every order.
ORDER_SEED = 20261104
ORDER_KEYS = ("baseline", "SW", "s3_acc", "s3_speed", "tome_r10", "tome_r12")
PAIRED_RATIO_PAIRS = (("s3_acc", "tome_r10"), ("s3_acc", "tome_r12"), ("s3_speed", "tome_r10"),
                      ("s3_speed", "tome_r12"), ("s3_speed", "s3_acc"))

# ---- accuracy protocol constants
N_VAL = 50000
N_SHARDS = 12
VAL_SHARD_NAME = "imagenet-1k-validation-{i:05d}-of-{n:05d}.arrow"
EVAL_BATCH_SIZE, EVAL_NUM_WORKERS = 64, 8
# Accuracy run list. S3 points use random targets (stochastic): seeds 0..4. Official ToMe is deterministic: one
# primary run (seed 0) plus a determinism replicate (reported, never selected). The baseline is deterministic: one
# run (seed 0; seeding is still applied).
ACCURACY_RUNS = (
    [{"run_id": "baseline_s0", "arm": "baseline", "seed": 0, "role": "deterministic"}]
    + [x for r in (10, 12) for x in (
        {"run_id": f"tome_r{r}_s0", "arm": f"tome_r{r}", "seed": 0, "role": "primary"},
        {"run_id": f"tome_r{r}_s0_rep", "arm": f"tome_r{r}", "seed": 0, "role": "determinism_replicate",
         "primary": f"tome_r{r}_s0"})]
    + [{"run_id": f"{a}_s{s}", "arm": a, "seed": s, "role": "seed"} for a in ("s3_acc", "s3_speed") for s in range(5)]
)
ACCURACY_RUN_INDEX = {r["run_id"]: r for r in ACCURACY_RUNS}

SMOKE_BANNER = "SMOKE MODE: software test only -- outputs are NOT paper measurements."


# --------------------------------------------------------------------------------------------- utilities
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def print_config(cfg):
    print(json.dumps({"resolved_config": cfg}, indent=1, default=str), flush=True)
    if cfg.get("smoke"):
        print(SMOKE_BANNER, flush=True)


def write_atomic_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "xb") as f:          # 'x': never overwrite
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        raise SystemExit(f"OUTPUT COLLISION {path}")
    os.rename(tmp, path)


def write_atomic_json(path, obj):
    write_atomic_bytes(path, json.dumps(obj, indent=1, default=str).encode())


def set_backends():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False      # TF32 matrix multiplication disabled (paper protocol)
    torch.backends.cudnn.allow_tf32 = True             # PyTorch default; affects the patch-embedding conv only


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def env_record(device="cuda"):
    dev = torch.device(device)
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "torchvision": _version("torchvision"), "timm": _version("timm"),
            "datasets": _version("datasets"), "pillow": _version("PIL"), "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" and torch.cuda.is_available() else None,
            "device": str(dev),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled(), "autocast_enabled": torch.is_autocast_enabled(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}


def _nvidia_smi(args, timeout=20):
    return subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=timeout).stdout.strip()


def gpu_snapshot():
    """nvidia-smi telemetry (recorded before and after every timing cell, outside the timed region)."""
    q = ("timestamp,index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,"
         "clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total")
    out = {"query": q}
    try:
        out["gpu"] = _nvidia_smi([f"--query-gpu={q}", "--format=csv,noheader"])
        out["compute_apps"] = _nvidia_smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    except Exception as e:  # noqa: BLE001  (no nvidia-smi, e.g. CPU smoke test)
        out["error"] = repr(e)
    return out


def compute_apps():
    try:
        return _nvidia_smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e!r}"


def gpu_idle_check(samples=60, interval_s=1.0, util_max=5.0):
    """OPTIONAL, WARN-ONLY: samples GPU utilization / thermal-slowdown flags and lists compute processes, so that a
    timing campaign is not started on a busy GPU. Never touches any process."""
    q = "utilization.gpu,temperature.gpu,power.draw,memory.free,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown"
    rec = {"utc": utc(), "query": q, "samples": [], "warnings": []}
    try:
        rec["compute_apps"] = _nvidia_smi(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        for _ in range(samples):
            rec["samples"].append([x.strip() for x in _nvidia_smi([f"--query-gpu={q}", "--format=csv,noheader,nounits"]).splitlines()[0].split(",")])
            time.sleep(interval_s)
    except Exception as e:  # noqa: BLE001
        rec["warnings"].append(f"nvidia-smi unavailable: {e!r}")
        return rec
    if rec["compute_apps"]:
        rec["warnings"].append(f"other compute processes present on the GPU: {rec['compute_apps']}")
    try:
        if any(float(s[0]) > util_max for s in rec["samples"]):
            rec["warnings"].append(f"GPU utilization above {util_max}% in at least one sample")
        if any(s[5] != "Not Active" or s[6] != "Not Active" for s in rec["samples"]):
            rec["warnings"].append("thermal slowdown reported in at least one sample")
    except (ValueError, IndexError):
        rec["warnings"].append("could not parse nvidia-smi samples")
    return rec


# ------------------------------------------------------------------------------------------ CLI helpers
def add_model_args(ap):
    g = ap.add_argument_group("model / environment")
    g.add_argument("--device", default="cuda", help="torch device (paper: a single NVIDIA H100 NVL, 'cuda'); "
                   "non-CUDA devices are only accepted with --smoke")
    g.add_argument("--checkpoint", default=None, help="local DeiT-S/16 checkpoint file (model.safetensors of "
                   f"timm/{CHECKPOINT_TAG}); default: let timm resolve {CHECKPOINT_TAG} through the Hugging Face hub/cache")
    g.add_argument("--expected-sha256", default=CHECKPOINT_FILE_SHA256,
                   help="required sha256 of the --checkpoint file (default: the paper checkpoint); '' disables the check")
    g.add_argument("--expected-state-sha256", default=STATE_SHA256,
                   help="required state_sha of the loaded weights before and after patching (default: the paper "
                        "checkpoint); '' only checks that patching leaves the weights unchanged")
    g.add_argument("--no-pretrained", action="store_true", help="random weights; only allowed with --smoke")
    g.add_argument("--tome-path", default=None, help="local checkout of the official ToMe repository to prepend to "
                   "sys.path (default: an installed 'tome' package)")
    g.add_argument("--no-verify-tome", action="store_true", help=f"skip the sha256 check of the official ToMe source "
                   f"files against commit {TOME_COMMIT[:8]}")
    return g


def validate_run_args(a):
    dev = torch.device(a.device)
    if a.no_pretrained and not a.smoke:
        raise SystemExit("--no-pretrained is only allowed with --smoke")
    if not a.smoke and dev.type != "cuda":
        raise SystemExit("paper measurements run on a CUDA device; use --smoke for CPU software tests")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {a.device} requested but CUDA is not available")
        if dev.index is not None:
            torch.cuda.set_device(dev)   # so that torch.cuda.synchronize() / get_rng_state() address this device
    return dev


def check_checkpoint(a):
    """Verifies the --checkpoint file sha256 (when a file is given)."""
    if a.no_pretrained:
        return {"pretrained": False, "note": "random weights (smoke test)"}
    if a.checkpoint is None:
        return {"pretrained": True, "source": f"timm hub/cache {CHECKPOINT_TAG}", "file_sha256": None,
                "note": "file hash not checked (no --checkpoint); weights verified through --expected-state-sha256"}
    p = Path(a.checkpoint)
    if not p.is_file():
        raise SystemExit(f"checkpoint file not found: {p}")
    h = sha256_file(p)
    if a.expected_sha256 and h != a.expected_sha256:
        raise SystemExit(f"CHECKPOINT SHA256 MISMATCH {p}: {h} != expected {a.expected_sha256} "
                         f"(the paper used {CHECKPOINT_TAG}, sha256 {CHECKPOINT_FILE_SHA256}); pass "
                         f"--expected-sha256 '' to run with a different file (results then not comparable)")
    return {"pretrained": True, "source": str(p), "file_sha256": h, "expected_sha256": a.expected_sha256 or None}


# ----------------------------------------------------------------------------------------- official ToMe
_TOME = {}


def load_tome(tome_path=None, verify=True):
    """Imports the official ToMe package (unmodified). With verify=True the pinned release files are hash-checked
    BEFORE import. Returns (tome package, the tome.patch.timm MODULE, {file: sha256})."""
    if _TOME:
        return _TOME["pkg"], _TOME["mod"], _TOME["hashes"]
    root = None
    if tome_path:
        root = Path(tome_path).resolve()
        if not (root / "tome" / "__init__.py").is_file():
            raise SystemExit(f"--tome-path {root} does not contain tome/__init__.py (expected a checkout of {TOME_REPO})")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("tome")
    if spec is None or spec.origin is None:
        raise SystemExit(f"official ToMe not found: pass --tome-path <checkout of {TOME_REPO} @ {TOME_COMMIT[:8]}> "
                         f"or install that commit")
    pkg_dir = Path(spec.origin).resolve().parent
    base = pkg_dir.parent
    hashes = {}
    for rel, exp in TOME_FILES.items():
        p = base / rel
        if not p.is_file():
            if rel == "LICENSE":           # not shipped inside an installed package
                hashes[rel] = None
                continue
            raise SystemExit(f"official ToMe file missing: {p}")
        h = sha256_file(p)
        hashes[rel] = h
        if verify and h != exp:
            raise SystemExit(f"OFFICIAL TOME SOURCE HASH MISMATCH {p}: {h} != {exp}. The paper used the unmodified "
                             f"release at commit {TOME_COMMIT}; check out that commit, or pass --no-verify-tome to run "
                             f"another version (results then not comparable).")
    import tome  # noqa: E402  (official, unmodified)
    import tome.patch.timm  # noqa: F401,E402
    if Path(tome.__file__).resolve().parent != pkg_dir:
        raise SystemExit(f"imported tome from {tome.__file__}, expected {pkg_dir}")
    # NB: tome/patch/__init__.py rebinds the attribute tome.patch.timm to the apply_patch FUNCTION, so the module object
    # (whose globals ToMeBlock.forward reads: bipartite_soft_matching, merge_wavg) must be taken from sys.modules.
    mod = sys.modules["tome.patch.timm"]
    assert callable(getattr(mod, "bipartite_soft_matching")) and hasattr(mod, "ToMeBlock")
    _TOME.update(pkg=tome, mod=mod, hashes={"verified": bool(verify), "root": str(base), "files": hashes})
    return _TOME["pkg"], _TOME["mod"], _TOME["hashes"]


# ---------------------------------------------------------------------------------------------- models
def state_sha(sd_items):
    """sha256 over (key, FP32 tensor bytes) pairs in key order."""
    h = hashlib.sha256()
    for k, v in sorted(sd_items, key=lambda kv: kv[0]):
        h.update(k.encode())
        h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def normalized_state_items(model):
    """State items under the original timm key names: strips the '.block.' level of the S3 wrappers
    (SRToMeBlock / PropAttnBlock) and skips the PropAttnBlock.merge_block REFERENCE entries ('.merge_block.'), which
    re-list the tensors of block late_start."""
    out = []
    for k, v in model.state_dict().items():
        if ".merge_block." in k:
            continue
        k2 = re.sub(r"^model\.", "", k)
        k2 = re.sub(r"blocks\.(\d+)\.block\.", r"blocks.\1.", k2)
        out.append((k2, v))
    return out


class ArmBuilder:
    """Builds every arm from a fresh timm model. Never modifies any source; only wraps / class-swaps the in-memory
    model exactly as the respective official (ToMe) or repository (S3-Merge) API does."""

    def __init__(self, device, checkpoint=None, pretrained=True, expected_state_sha256=STATE_SHA256, init_seed=None):
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.pretrained = pretrained
        self.expected_state_sha256 = expected_state_sha256 if pretrained else ""
        self.init_seed = init_seed      # smoke tests with random weights only: identical init for repeated builds

    def build_base(self):
        import timm
        if self.init_seed is not None:
            torch.manual_seed(self.init_seed)
        kw = {"pretrained_cfg_overlay": dict(file=str(self.checkpoint))} if (self.pretrained and self.checkpoint) else {}
        m = timm.create_model(MODEL_NAME, pretrained=self.pretrained, **kw)
        return m.to(self.device).eval().float()

    def build(self, name):
        """Returns dict(name, model, inner, info). ``inner`` = the 12 original timm Blocks (for hooks)."""
        from models.sr_tome import apply_patch, SRToMeBlock, PropAttnBlock
        m = METHODS[name]
        base = self.build_base()
        sha_pre = state_sha(list(base.state_dict().items()))
        inner = list(base.blocks)
        info = {"method": name, "config": m}
        if m["family"] == "unmerged":
            model = base
        elif m["family"] == "official_tome":
            tome, _, _ = load_tome()     # already loaded and verified by the entry point
            tome.patch.timm(base, trace_source=False, prop_attn=m["prop_attn"])
            base.r = m["r"]
            model = base
            info["tome_r"] = base.r
            info["tome_info"] = {k: (v if not torch.is_tensor(v) else "tensor") for k, v in base._tome_info.items()}
        else:
            ap = m["apply_patch"]
            model = apply_patch(base, **ap)
            info["s3merge_config"] = dict(model.s3merge_config)
            info["pa_blocks"] = sum(isinstance(b, PropAttnBlock) for b in model.blocks)
            info["merge_block_index"] = [i for i, b in enumerate(model.blocks) if isinstance(b, SRToMeBlock)]
            sc = model.s3merge_config
            if not (sc["total_tokens_out"] == 1 + ap["target_tokens"] and sc["L_s"] == ap["late_start"] + 1
                    and sc["prop_attn"] is True and sc["merges"] is True and info["merge_block_index"] == [ap["late_start"]]
                    and info["pa_blocks"] == 11 - ap["late_start"]):
                raise SystemExit(f"S3-Merge configuration of {name} differs from the paper operating point: {sc}")
        sha_post = state_sha(normalized_state_items(model))
        if sha_pre != sha_post:
            raise SystemExit(f"STATE SHA CHANGED BY PATCHING {name}: {sha_pre} / {sha_post}")
        if self.expected_state_sha256 and sha_post != self.expected_state_sha256:
            raise SystemExit(f"CHECKPOINT STATE SHA MISMATCH {name}: {sha_post} != expected {self.expected_state_sha256} "
                             f"(pass --expected-state-sha256 '' to run with different weights; results then not comparable)")
        head = model.head
        assert isinstance(head, nn.Linear) and head.out_features == 1000 and head.in_features == 384
        assert all(p.dtype == torch.float32 for p in model.parameters())
        info["state_sha"] = sha_post
        return dict(name=name, model=model, inner=inner, info=info)


def expected_block_inputs(name):
    """Tokens (incl. [CLS]) entering each of the 12 blocks."""
    m = METHODS[name]
    if m["family"] == "unmerged":
        return [197] * 12
    if m["family"] == "official_tome":
        return [197 - m["r"] * i for i in range(12)]
    ls = m["apply_patch"]["late_start"]
    return [197] * (ls + 1) + [1 + m["apply_patch"]["target_tokens"]] * (11 - ls)


def expected_final_tokens(name):
    m = METHODS[name]
    if m["family"] == "official_tome":
        return 197 - 12 * m["r"]
    if m["family"] == "unmerged":
        return 197
    return 1 + m["apply_patch"]["target_tokens"]


def expected_attn_mask_blocks(name):
    """Blocks that receive an attn_mask kwarg (the S3 proportional-attention bias)."""
    m = METHODS[name]
    if m["family"] == "s3_canonical":
        return list(range(m["apply_patch"]["late_start"] + 1, 12))
    return []


class Hooks:
    """Per forward: input tokens of each of the 12 inner timm Blocks, whether attn_mask was passed, and the input
    tokens of the final norm."""

    def __init__(self, inner_blocks, final_norm):
        self.block_in, self.mask = [], []
        self.final = None
        self.h = [b.register_forward_pre_hook(self._pre, with_kwargs=True) for b in inner_blocks]
        self.h.append(final_norm.register_forward_pre_hook(self._norm))

    def _pre(self, mod, args, kwargs):
        self.block_in.append(int(args[0].shape[1]))
        self.mask.append(kwargs.get("attn_mask", None) is not None)

    def _norm(self, mod, args):
        self.final = int(args[0].shape[1])

    def reset(self):
        self.block_in, self.mask, self.final = [], [], None

    def remove(self):
        for h in self.h:
            h.remove()


def final_norm_of(model):
    return model.norm


# ------------------------------------------------------------------------------------------ timing inputs
def invocation_orders(seed=ORDER_SEED):
    """{'latency': {0..9: order}, 'profile': {0..2: order}} over ORDER_KEYS (see the note at ORDER_KEYS)."""
    rng = random.Random(seed)
    keys = list(ORDER_KEYS)
    lat = {i: rng.sample(keys, len(keys)) for i in range(N_LATENCY_INVOCATIONS)}
    prof = {i: rng.sample(keys, len(keys)) for i in range(N_PROFILE_INVOCATIONS)}
    return {"latency": lat, "profile": prof}


def order_for(mode, invocation, arms, seed=ORDER_SEED, explicit=None):
    if explicit:
        order = [s.strip() for s in explicit.split(",") if s.strip()]
        if sorted(order) != sorted(arms):
            raise SystemExit(f"--order {order} must be a permutation of --arms {list(arms)}")
        return order
    orders = invocation_orders(seed)[mode]
    if invocation not in orders:
        raise SystemExit(f"randomized arm orders are defined for {mode} invocations {min(orders)}..{max(orders)} "
                         f"(seed {seed}); pass --order explicitly for other invocation indices")
    return [k for k in orders[invocation] if k in arms]


def make_inputs(batch_sizes, seed=INPUT_SEED):
    """Synthetic timing inputs (CPU). B = 1, 8, 32 are always drawn first, in that order, so their values never depend
    on which batch sizes are requested; any other batch size is drawn afterwards."""
    g = torch.Generator().manual_seed(seed)
    xs = {}
    for b in list(INPUT_DRAW_ORDER) + [b for b in batch_sizes if b not in INPUT_DRAW_ORDER]:
        if b not in xs:
            xs[b] = torch.randn(b, 3, 224, 224, generator=g, dtype=torch.float32)
    meta = {}
    for b in batch_sizes:
        h = hashlib.sha256(xs[b].numpy().tobytes()).hexdigest()
        exp = INPUT_TENSOR_SHA256.get(b) if seed == INPUT_SEED else None
        meta[str(b)] = {"sha256": h, "matches_paper_inputs": (h == exp) if exp else None}
    return {b: xs[b] for b in batch_sizes}, meta


# ------------------------------------------------------------------------------------------------- data
def resolve_val_shards(data_dir, n_shards=N_SHARDS):
    """Paths of the HF arrow shards of the ImageNet-1K validation split, in shard order. ``data_dir`` is the directory
    that holds them, or any parent of it (searched recursively for the first shard)."""
    d = Path(data_dir)
    first = VAL_SHARD_NAME.format(i=0, n=n_shards)
    if not (d / first).is_file():
        hits = sorted(d.rglob(first)) if d.is_dir() else []
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one '{first}' under {d}, found {len(hits)}: {[str(h) for h in hits][:5]}")
        d = hits[0].parent
    paths = [d / VAL_SHARD_NAME.format(i=i, n=n_shards) for i in range(n_shards)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing validation shards: {missing}")
    return paths


def open_val(shard_paths, columns=None):
    from datasets import Dataset, concatenate_datasets
    ds = concatenate_datasets([Dataset.from_file(str(p)) for p in shard_paths])
    assert len(ds) == N_VAL, len(ds)
    if columns is not None:
        ds = ds.select_columns(columns)
    return ds


def make_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ImageNetVal(torch.utils.data.Dataset):
    """ImageNet-1K validation in HF index order 0..49999. Returns (tensor, label, hf_index)."""

    def __init__(self, shard_paths, with_labels=True):
        self.shards = [str(p) for p in shard_paths]
        self.indices = list(range(N_VAL))
        self.with_labels = with_labels
        self.tfm = make_transform()
        self.ds = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        if self.ds is None:   # opened lazily inside each worker (memory-mapped, read-only)
            self.ds = open_val(self.shards, columns=None if self.with_labels else ["image"])
        i = self.indices[j]
        rec = self.ds[i]
        x = self.tfm(rec["image"].convert("RGB"))
        y = int(rec["label"]) if self.with_labels else -1
        return x, y, i


class SyntheticImages(torch.utils.data.Dataset):
    """SMOKE TESTS ONLY: random RGB images of varying size (and random labels) passed through the same preprocessing
    as ImageNetVal. Not ImageNet; nothing computed on them is an accuracy."""

    def __init__(self, n, seed=0):
        self.n, self.seed = int(n), int(seed)
        self.indices = list(range(self.n))
        self.tfm = make_transform()

    def __len__(self):
        return self.n

    def __getitem__(self, j):
        from PIL import Image
        rs = np.random.RandomState(self.seed + j)
        h, w = 224 + int(rs.randint(0, 160)), 224 + int(rs.randint(0, 160))
        img = Image.fromarray(rs.randint(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
        return self.tfm(img.convert("RGB")), int(rs.randint(0, 1000)), j


def make_loader(dataset, batch_size, num_workers, pin_memory=True):
    g = torch.Generator()
    g.manual_seed(0)      # worker base seed only; leaves the global CPU/CUDA RNGs untouched
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       pin_memory=pin_memory, drop_last=False, generator=g)
