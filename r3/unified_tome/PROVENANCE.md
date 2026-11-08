# Provenance: `r3/unified_tome/`

These scripts are cleaned ports of the validated scripts that produced paper Table 1, Fig. 5 and Supplementary Tables S20–S22. The SHA-256 values identify the archived source files.

| File | Source file(s) (SHA-256) |
|---|---|
| `unified_common.py` | `123222a0493c0a5d88a188a7d6258548dcffa6a5b9fe0dafba8ed9e5da81a999`, `3e9a3165602ea157aa1aab8b72df4d0f1e214098aa257a2e781ff2a041c2d5ac`, `56f7015186c87518761f95d4eb064e8a13b7d40aac4975dc876cd40e6b374dfd`, `d60ab6d3bdafbb0e2659ebba2c8988d15047a1212e18226187d03f2ac3a26127`; campaign configuration and manifests `2b64da60…`, `3a4a5d8b…`, `4a628594…` |
| `timing_unified.py` | `2004039b1efeaee048a2a4c56ed253271de1ebbdcf25e17b3479dffde69c2834` |
| `timing_campaign.py` | `1764f32c94296d4961834e6ef8395d18207b1f82d1694e051bb91f6e2a26cdae`, `08a3b32c7255b5bd9d79de12ffd2754ac76a9aa875831a4d478c08ca623ca942` |
| `timing_analysis.py` | `cd374cbdf7e9b630826a1725d5fcd35e5f439b17fdfd6d65f67d96ed4cb3c498` |
| `accuracy_eval.py` | `918f910860a4bd57f7725461f00169bde81a30c9ec8a73fab520acb40709be00`, `722402bc3f819292baeda9cbdb56c496ddd7ef509a7c7890d6fcd0e3bdbea3d2` |
| `accuracy_campaign.py` | `ac435147ee85f25831cc2ad0dd08e89945e940e28147efc50ca9db27815ab6f7` |
| `accuracy_check.py` | `1062c0af1ea22f352f8906ece7d6d23113ac44fe3328342331d1a20f682869f7` |
| `accuracy_analysis.py` | `6a225ae7debf8bef8846e2e85a47b5906cce20f871af4bc8aa34709b5dfac5e2` |
| `gmac.py` | `c5b18863bd9c530db1a5f58bd398917cb7cab5b150487e16cb178550ea4774de` |

The S³-Merge arms use `models/sr_tome.py` of this repository. On synthetic inputs, its merge computation is bit-identical to the module used for the paper's campaign (see `REPRODUCIBILITY.md`).

## Unchanged protocol

The following are as in the paper campaign:
- arms and their configurations;
- FP32 and backend settings;
- batch sizes;
- 30 warm-up and 200 timed forwards with synchronization around every forward;
- ten latency invocations with the paper's randomized arm orders;
- synthetic inputs (the regenerated tensors equal the paper's by SHA-256);
- seeds, data order and preprocessing;
- the aggregation and confidence-interval methods.

## Differences from the paper campaign

1. **Sixth configuration not timed.** The paper campaign also timed a sixth configuration that is not a Table 1 row (the single-shot/windowed SW arm of the factorial study, PA off). It is no longer timed. Its position is still drawn, so the five Table 1 arms keep the paper's relative order.
2. **Baseline accuracy is computed.** The baseline is evaluated as run `baseline_s0`, with the same pipeline, instead of reusing an earlier run.
3. **One process per invocation.** Each invocation is a fresh Python process; the paper used a fresh container of the same image.
4. **Inputs and orders regenerated.** Synthetic inputs and arm orders are generated in-process rather than read from files; both are verified to be identical.
5. **Environment-specific checks removed.** The resource and scheduling checks specific to the original compute environment are gone; an optional, warn-only GPU-idle check remains. None of these checks changed any computation.
6. **Interface changes.**
   - Paths are command-line arguments.
   - The checkpoint hash is also checked in timing runs.
   - `--smoke` replaces the original reduced test modes.
   - Run identifiers and output file names are renamed.
   - The timing analysis no longer requires the accuracy analysis; the join is optional.
   - Extra fields (resolved configuration, input hashes) are recorded.
7. **Fidelity checks not included.** The fidelity checks of official ToMe and of the S³-Merge operating points are not part of this directory.
