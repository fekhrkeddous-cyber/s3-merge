# Provenance: `r3/factorial/`

These scripts are cleaned ports of the validated scripts that produced paper Table 4 and Supplementary Tables S17–S19. The SHA-256 values identify the archived source files.

| File | Source file(s) (SHA-256) |
|---|---|
| `factorial_merge.py` | `3d9398f34cffce42d101e24562f3a2a3419570e6471d9c27fbefa23c0fefdc73`. Only comments and docstrings changed: the syntax tree without docstrings is identical |
| `factorial_common.py` | `3e9a3165602ea157aa1aab8b72df4d0f1e214098aa257a2e781ff2a041c2d5ac`, `56f7015186c87518761f95d4eb064e8a13b7d40aac4975dc876cd40e6b374dfd` (helpers only) |
| `accuracy.py` | `722402bc3f819292baeda9cbdb56c496ddd7ef509a7c7890d6fcd0e3bdbea3d2` |
| `timing.py` | `87b0019f4b21e4afd34fb17a4047d52baace4086ee9211fb512a62b81177b595`, `d60ab6d3bdafbb0e2659ebba2c8988d15047a1212e18226187d03f2ac3a26127` (input and order rules) |
| `campaign.py` | `eaf4294135252bdeda8f737530f1b1129f8de5fe5ef6b4690443ccce89e3c775`, `2558f29a8b3840351a81d83c55dd9502a88caf0dd71271cfbdef9cf41aa8d3cb` |
| `analyze_accuracy.py` | `c6c6fb4e4948102d7d98628e612fd346bd424db3f51463a895fc76eec095302b` |
| `analyze_timing.py` | `882a166f3a877d0c36ae969c8d0fa18e62aaffa9f0f7f2799c7f16e1ce4c94a9` |
| `test_factorial.py` | new (CPU tests of the properties checked on GPU for the paper campaign) |

The `core` anchor arm uses `models/sr_tome.py` of this repository, with PA off. On synthetic inputs, its merge computation is bit-identical to the module used for the paper's campaign.

## Unchanged protocol

The following are as in the paper campaign:
- the arms SW/SG/PW/PG (84 patch tokens; no PA), the baseline and the `core` anchor;
- seeds 0–4 and the seeding calls;
- data order and preprocessing;
- FP32 with TF32 matmul off;
- batch 64 with 8 workers for accuracy;
- B ∈ {1, 8, 32}, 30 warm-up and 200 timed forwards with synchronization around every forward;
- ten latency invocations with the paper's arm orders;
- synthetic inputs (verified equal by SHA-256);
- the analysis margins (±0.10 points; ±1% latency) and the CI methods.

Re-applied to the paper's raw outputs, the analysis scripts reproduce the original analysis exactly.

## Differences from the paper campaign

1. **Environment-specific checks removed.** The resource, scheduling and approval checks of the original compute environment are gone. Other GPU processes at start now produce a warning instead of stopping the timing run. None of these checks changed any computation.
2. **Courtesy pause removed.** A pause between accuracy batches was used only to share the GPU. It affected only bookkeeping timings, not results.
3. **Inputs and orders regenerated.** Synthetic inputs and arm orders are generated in-process rather than read from files; both are verified to be identical.
4. **Historical consistency check replaced.** It is now a comparison with the published Table S17 counts and the paper baseline's per-batch checksum.
5. **Interface changes.**
   - Paths are command-line arguments.
   - `--smoke` replaces the original reduced test modes.
   - Run identifiers and output file names are renamed.
   - The timing campaign no longer runs the analysis automatically.
