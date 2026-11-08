"""Shared code for the controlled 2x2 factorial study (paper Table 4; Supplementary Tables S17-S19).

* Arms: ``baseline`` (unmerged DeiT-S/16), ``core`` (canonical S3-Merge Core through models/sr_tome.py
  ``apply_patch`` with PA off; a control that the factorial SW arm must reproduce, not a fifth arm) and the
  factorial arms ``SW``, ``SG``, ``PW``, ``PG`` (factorial_merge.FactorialModel, PA off).
* ImageNet-1K validation is opened read-only from the Hugging Face arrow shards of ``ILSVRC/imagenet-1k``
  (split ``validation``) with ``datasets.Dataset.from_file`` (memory-mapped), in shard order, then concatenated.
  This avoids ``datasets.load_dataset``, which touches a lock file in the cache directory. Decoding uses the HF
  Image feature exactly as the paper's evaluation loader: x['image'].convert('RGB') -> Resize(256) ->
  CenterCrop(224) -> ToTensor -> Normalize(ImageNet). datasets 4.5.0 applies ImageOps.exif_transpose when an
  EXIF Orientation tag is present (paper environment).
* No bytecode is written (sys.dont_write_bytecode).
"""
import sys
sys.dont_write_bytecode = True

import glob, hashlib, json, os, re, socket, subprocess, time  # noqa: E401,E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from factorial_merge import FactorialModel, ARM_SPECS, window_of_coord, N_PATCH, NWIN, K_FINAL  # noqa: E402
from models.sr_tome import apply_patch, PropAttnBlock  # noqa: E402

N_VAL = 50000
FACT = ["SW", "SG", "PW", "PG"]
ARMS = ["baseline", "core"] + FACT
TIMING_ARMS = ["baseline"] + FACT
MODEL_NAME = "deit_small_patch16_224"
HF_HUB_ID = "timm/deit_small_patch16_224.fb_in1k"
HF_HUB_FILE = "model.safetensors"
CKPT_FILE_SHA = "1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b"   # model.safetensors
STATE_SHA = "a17d00754ef1e5401cf33cf61af23094c48b8f4a4f5f8898d8ae36b9d81925e1"       # state_sha() of the loaded weights
DEFAULT_SHARD_PATTERN = "imagenet-1k-validation-*-of-*.arrow"
# Canonical S3-Merge Core control: single-shot merge after block index 8 (paper L_s = 9), w = 7, T = 84, PA off.
# Every argument is passed explicitly; the values equal the paper's call apply_patch(target_tokens=84,
# late_start=8, window_size=7, prop_attn=False) with the module defaults weighted_merge=False, tau=0.3,
# content_adaptive=False and the 14x14 grid of DeiT-S/16 at 224^2.
CORE_KWARGS = dict(target_tokens=84, late_start=8, window_size=7, weighted_merge=False, tau=0.3,
                   content_adaptive=False, prop_attn=False, grid_size=14, keep_all_control=False)
EXPECT_TRAJ = {
    "baseline": [197] * 12,
    "core": [197] * 9 + [85] * 3,
    "SW": [197] * 9 + [85] * 3, "SG": [197] * 9 + [85] * 3,
    "PW": [197] * 6 + [169, 141, 113] + [85] * 3, "PG": [197] * 6 + [169, 141, 113] + [85] * 3,
}
GMAC_ANALYTICAL = {"baseline": 4.5989, "single_shot": 3.9316, "progressive": 3.5919}
SMOKE_NOTICE = ("SMOKE MODE: software test only (tiny counts, synthetic inputs). "
                "These results are NOT paper measurements.")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def source_hashes(*extra):
    """sha256 of the code that defines the run (informational; recorded in every output, never enforced)."""
    files = [HERE / "factorial_merge.py", HERE / "factorial_common.py", REPO / "models" / "sr_tome.py"]
    files += [Path(p).resolve() for p in extra]
    return {str(p.relative_to(REPO)) if REPO in p.parents else str(p): sha256_file(p) for p in files}


def print_config(cfg):
    print("RESOLVED CONFIGURATION:", flush=True)
    print(json.dumps(cfg, indent=1, default=str), flush=True)


def none_or(s):
    """'none' (any case) or '' -> None, else the string."""
    return None if s is None or str(s).strip().lower() in ("", "none") else str(s)


# ----------------------------------------------------------------------------------------------- data
def shard_paths(data_dir, pattern=DEFAULT_SHARD_PATTERN):
    """Validation shards in shard order (zero-padded names sort lexicographically: 00000, 00001, ...)."""
    files = sorted(glob.glob(os.path.join(str(data_dir), pattern)))
    if not files:
        raise SystemExit(f"no ImageNet-1K validation shards matching {pattern!r} in {data_dir}")
    totals = {m.group(1) for m in (re.search(r"-of-(\d+)\.arrow$", f) for f in files) if m}
    if len(totals) == 1 and int(totals.pop()) != len(files):
        raise SystemExit(f"incomplete shard set in {data_dir}: found {len(files)} files matching {pattern!r}")
    return files


def open_val(files, decode=True, columns=None):
    from datasets import Dataset, concatenate_datasets, Image as HFImage
    ds = concatenate_datasets([Dataset.from_file(str(p)) for p in files])
    if len(ds) != N_VAL:
        raise SystemExit(f"expected {N_VAL} validation images, found {len(ds)}")
    if columns is not None:
        ds = ds.select_columns(columns)
    if not decode:
        ds = ds.cast_column("image", HFImage(decode=False))
    return ds


def make_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ValDS(torch.utils.data.Dataset):
    """Evaluation-loader semantics. Returns (tensor, label or -1, hf_index)."""

    def __init__(self, files, indices=None, with_labels=True):
        self.files = [str(f) for f in files]
        self.indices = list(range(N_VAL)) if indices is None else [int(i) for i in indices]
        self.with_labels = with_labels
        self.tfm = make_transform()
        self.ds = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        if self.ds is None:   # opened lazily inside each worker (memory-mapped, read-only)
            self.ds = open_val(self.files, decode=True, columns=None if self.with_labels else ["image"])
        i = self.indices[j]
        rec = self.ds[i]
        x = self.tfm(rec["image"].convert("RGB"))
        y = int(rec["label"]) if self.with_labels else -1
        return x, y, i


class SyntheticDS(torch.utils.data.Dataset):
    """SMOKE ONLY: random RGB PIL images of varying size and random labels, through the same transform."""

    def __init__(self, n):
        self.n = int(n)
        self.tfm = make_transform()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from PIL import Image
        r = np.random.RandomState(1000 + i)
        h, w = int(r.randint(200, 400)), int(r.randint(200, 400))
        img = Image.fromarray(r.randint(0, 256, size=(h, w, 3), dtype=np.uint8), mode="RGB")
        return self.tfm(img.convert("RGB")), int(r.randint(0, 1000)), i


def make_loader(dataset, batch_size, num_workers, pin_memory):
    g = torch.Generator()
    g.manual_seed(0)      # worker base seed only; leaves the global CPU/CUDA RNGs untouched
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       pin_memory=pin_memory, drop_last=False, generator=g)


# ---------------------------------------------------------------------------------------------- model
def set_backends():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False      # paper protocol: FP32 matmul, no TF32
    torch.backends.cudnn.allow_tf32 = True             # PyTorch default (affects the patch-embedding conv only)


def sync_fn(device):
    """Zero-argument synchronize callable for `device`. For the default CUDA device this is exactly
    torch.cuda.synchronize (called with no argument, as in the paper's timing loops); no-op on CPU."""
    d = torch.device(device)
    if d.type != "cuda":
        return lambda: None
    if d.index is None:
        return torch.cuda.synchronize
    return lambda: torch.cuda.synchronize(d)


def sync(device):
    """torch.cuda.synchronize() on CUDA devices; no-op on CPU."""
    sync_fn(device)()


def resolve_checkpoint(checkpoint, expected_sha256, pretrained=True):
    """Returns (path or None, measured file sha256 or None). Without --checkpoint, the timm release file is
    fetched from (or found in) the Hugging Face hub cache. The file sha256 is checked unless disabled."""
    if not pretrained:
        return None, None
    if checkpoint:
        path = str(Path(checkpoint).expanduser().absolute())  # keep the file name: timm picks the loader by extension
        if not os.path.isfile(path):
            raise SystemExit(f"checkpoint file not found: {path}")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_HUB_ID, HF_HUB_FILE)
    got = sha256_file(path)
    if expected_sha256 is not None and got != expected_sha256:
        raise SystemExit(f"CHECKPOINT FILE SHA256 MISMATCH {path}: {got} != expected {expected_sha256}")
    return path, got


def build_base(ckpt_path, device, pretrained=True):
    """timm DeiT-S/16 (deit_small_patch16_224.fb_in1k weights from the given file), eval mode, FP32."""
    import timm
    if pretrained:
        m = timm.create_model(MODEL_NAME, pretrained=True, pretrained_cfg_overlay=dict(file=str(ckpt_path)))
    else:   # SMOKE ONLY: deterministic random weights; the global RNG state is left untouched
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            m = timm.create_model(MODEL_NAME, pretrained=False)
    return m.to(device).eval().float()


def state_sha(sd_items):
    """sha256 over sorted (key, float32 tensor bytes) pairs."""
    h = hashlib.sha256()
    for k, v in sorted(sd_items, key=lambda kv: kv[0]):
        h.update(k.encode())
        h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def normalized_items(model):
    """state_dict items with wrapper prefixes removed, so wrapped and unwrapped models hash identically."""
    out = []
    for k, v in model.state_dict().items():
        k2 = re.sub(r"^model\.", "", k)
        k2 = re.sub(r"blocks\.(\d+)\.block\.", r"blocks.\1.", k2)
        out.append((k2, v))
    return out


class TrajHooks:
    """Pre-hooks on the 12 inner timm Blocks: per-forward token trajectory and attn_mask presence."""

    def __init__(self, inner_blocks):
        self.traj, self.masks = [], []
        self.handles = [blk.register_forward_pre_hook(self._pre(i), with_kwargs=True) for i, blk in enumerate(inner_blocks)]

    def _pre(self, i):
        def f(mod, args, kwargs):
            self.traj.append(int(args[0].shape[1]))
            self.masks.append(kwargs.get("attn_mask", None) is not None)
        return f

    def reset(self):
        self.traj, self.masks = [], []


def build_model(arm, ckpt_path, device, pretrained=True, expected_state_sha=STATE_SHA):
    """Accuracy-path builder: arm wrapper + trajectory hooks + configuration checks. The factorial arms use
    record=False (the record flag only adds bookkeeping)."""
    if arm not in ARMS:
        raise ValueError(arm)
    base = build_base(ckpt_path, device, pretrained)
    sha_pre = state_sha(list(base.state_dict().items()))
    inner = list(base.blocks)
    if arm == "baseline":
        model = base
    elif arm == "core":
        model = apply_patch(base, **CORE_KWARGS)
    else:
        model = FactorialModel(base, arm, record=False)
    sha_post = state_sha(normalized_items(model))
    if sha_pre != sha_post or (expected_state_sha is not None and sha_post != expected_state_sha):
        raise SystemExit(f"CHECKPOINT STATE SHA MISMATCH for {arm}: {sha_pre} / {sha_post} (expected {expected_state_sha})")
    has_pa = any(isinstance(m, PropAttnBlock) for m in model.modules())
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    # Every timm module must be in eval mode. The parameter-free wrappers (FactorialModel, FactorialMergeBlock,
    # SRToMeBlock) keep nn.Module's default training flag; none of them reads self.training.
    wrapper_names = {"FactorialModel", "FactorialMergeBlock", "SRToMeBlock"}
    train_modules = sorted({type(mm).__name__ for mm in model.modules() if mm.training})
    eval_ok = set(train_modules) <= wrapper_names
    if has_pa or dtypes != ["torch.float32"] or not eval_ok:
        raise SystemExit(f"CONFIG VIOLATION for {arm}: PA={has_pa} dtypes={dtypes} train-mode modules={train_modules}")
    hooks = TrajHooks(inner)
    return dict(arm=arm, model=model, hooks=hooks, state_sha=sha_post, has_propattn=has_pa, param_dtypes=dtypes,
                train_flag_wrappers=train_modules, s3merge_config=getattr(model, "s3merge_config", None))


def forward_fn(model, x):
    """THE single forward boundary used for every arm: no_grad, synchronize before and after model(x)."""
    with torch.no_grad():
        sync(x.device)
        out = model(x)
        sync(x.device)
    return out


def forward(m, x):
    """Accuracy-path forward with per-forward trajectory / attn_mask / logits checks."""
    m["hooks"].reset()
    logits = forward_fn(m["model"], x)
    if m["hooks"].traj != EXPECT_TRAJ[m["arm"]] or any(m["hooks"].masks):
        raise SystemExit(f"TRAJECTORY/MASK VIOLATION {m['arm']}: {m['hooks'].traj} masks={m['hooks'].masks}")
    if logits.dtype != torch.float32 or logits.shape[1] != 1000 or not bool(torch.isfinite(logits).all()):
        raise SystemExit(f"LOGITS VIOLATION {m['arm']}: {logits.dtype} {tuple(logits.shape)}")
    return logits


_WIN = None


def factorial_state_checks(m, B):
    """Invariants from the post-forward FactorialState (available with record=False):
    final targets (canonical topk order), masses, membership. Returns (rep_u8 [B,84] cpu, ok dict, stats dict)."""
    global _WIN
    st = m["model"].state
    rep, size, owner = st.rep, st.size, st.owner
    dev = rep.device
    if _WIN is None or _WIN.device != dev:
        _WIN = window_of_coord(dev)
    ok = {}
    ok["final_84"] = tuple(rep.shape) == (B, NWIN * K_FINAL) and tuple(size.shape) == (B, NWIN * K_FINAL)
    ok["stage_calls"] = list(st.stage_calls) == list(ARM_SPECS[m["arm"]][0])
    ok["mass_196"] = bool((size.sum(1) == float(N_PATCH)).all())
    slot = torch.full((B, N_PATCH), -1, device=dev, dtype=torch.long)
    slot.scatter_(1, rep, torch.arange(NWIN * K_FINAL, device=dev).unsqueeze(0).expand(B, -1))
    restore = slot.gather(1, owner)
    ok["owner_in_targets"] = bool((restore >= 0).all())
    ok["targets_own_themselves"] = bool(torch.equal(owner.gather(1, rep), rep))
    cnt = torch.zeros(B, NWIN * K_FINAL, device=dev, dtype=torch.float32)
    cnt.scatter_add_(1, restore.clamp_min(0), torch.ones(B, N_PATCH, device=dev, dtype=torch.float32))
    ok["counts_equal_mass"] = bool(torch.equal(cnt, size))
    ok["targets_in_original_window"] = bool(torch.equal(
        _WIN[rep], torch.arange(NWIN, device=dev).repeat_interleave(K_FINAL).unsqueeze(0).expand(B, -1)))
    cross = (_WIN[owner] != _WIN[torch.arange(N_PATCH, device=dev)].unsqueeze(0))     # [B,196]
    n_cross = int(cross.sum())
    if ARM_SPECS[m["arm"]][2] == "windowed":
        ok["windowed_zero_cross_window_membership"] = n_cross == 0
    hist = torch.bincount(size.round().long().flatten(), minlength=N_PATCH + 1).cpu().numpy()
    stats = {"n_cross_window_members": n_cross, "size_hist": hist}
    return rep.to(torch.uint8).cpu(), ok, stats


# ------------------------------------------------------------------------------------------ environment
def gpu_snapshot():
    """nvidia-smi telemetry (never inside a timed region). Returns an 'error' entry when nvidia-smi is absent."""
    q = ("timestamp,index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,"
         "clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total")
    out = {"query": q}
    try:
        out["gpu"] = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                                    text=True, timeout=20).stdout.strip()
        out["compute_apps"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                              "--format=csv,noheader"], capture_output=True, text=True,
                                             timeout=20).stdout.strip()
    except Exception as e:  # pragma: no cover
        out["error"] = repr(e)
    return out


def compute_apps():
    """PIDs of compute processes on the GPU(s), or None when nvidia-smi is unavailable."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def env_record(device):
    import timm, torchvision, PIL
    try:
        import datasets
        datasets_version = datasets.__version__
    except Exception:  # datasets is only needed for the ImageNet path
        datasets_version = None
    dev = torch.device(device)
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "torchvision": torchvision.__version__, "timm": timm.__version__, "datasets": datasets_version,
            "pillow": PIL.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
            "device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled(), "autocast_enabled": torch.is_autocast_enabled(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "hostname": socket.gethostname(), "uid": os.getuid(), "pid": os.getpid(),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}


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
