# DataMIL replacement: paper settings, 2026-11-04

The user explicitly stopped trajectory-level / 10-supplied-trial policy training.
Old checkpoints, results, and logs were preserved. They are NOT paper-matched results.
AMF jobs were not stopped.

## Configuration and provenance

Paper: https://arxiv.org/html/2505.09603 (v2), Appendix F.2, D.1 and Table 4.

| Setting | Replacement |
| --- | --- |
| Prior data | Existing 4500 LIBERO-90 demonstrations / 669043 frames |
| Target | Existing fixed 5 Book-Caddy demonstrations; author-exact demo IDs unverified |
| Selection units | Horizon15, released-builder short-tail convention (minimum8) |
| Selection iterations/trials | 30 NEW iterations; old trajectory influences incompatible |
| Learner / differentiated steps per iteration | 2100 / last100 (paper Table4) |
| Learner LR schedule | Cosine; code release hardcoded linear, explicitly corrected |
| Proxy | Paper pre-grasp weighting2, other states1; 20-step lookback from released helper |
| Selection budget | Top10% of segment IDs, not top10% of whole episodes |
| Final policy | Octo-small; batch128; 10000 steps; target/prior1:1; action chunk8 |
| Policy seeds | ONE seed0 first, explicitly requested diagnostic scope, not paper5-seed mean |
| Evaluation | 50 Book-Caddy rollouts, same released evaluation task |

Values not stated in paper retain released-code defaults, notably effective datamodel
batch512 and LR3e-4. Runtime microbatch128 (actual replay tiles64), loader4 workers,
small model-init batch8 limit memory on one H100 while AMF remains active. These do
not reduce effective batch size. They are not a claim of bitwise reproduction.
Common prior action/proprio statistics are used across prior/target, as instructed
by the release README. Other low-level optimizer defaults remain from the release;
this is not evidence that every unpublished author setting has been recovered.

## Data preparation

Isolated data: `/mnt/data/thanhpv/datamil_data/paper_h15_t30`.
The cached prior RLDS records are reindexed into **46705** segment IDs while
preserving the original encoded images/actions/language. No PNG re-encoding or
change of physical demonstrations. Completed TFRecord shards can be reused.
MDS datasets are regenerated for segment grouping and shared normalization.
Target MDS also carries proxy weights, which were commented out in the release.
Data preparation is CPU-only; AMF continues using GPU.

## Tests/fixes

- CPU: segment coverage/short tails, exact selected-frame filtering, preservation
  of original future-action context, legacy trajectory-filter compatibility.
- CPU: atomic optimizer-array checkpoint roundtrip; shape/dtype/count validation.
- CPU: repeated cluster indices accumulate all metagradients (`np.add.at`); NumPy
  fancy-index `+=` in the release loses duplicate contributions.
- GPU: JAX/CuPy replay transfer passed on H100.
- GPU: actual metagradient forward state saved at step2, then restored into a
  new process and reused successfully. Backward completion is checked separately.
- Actual proxy-gradient GPU smoke found `CUDNN_STATUS_EXECUTION_FAILED` while
  the existing environment loaded CUDA11/12/13 and cuDNN8/9 together. This is
  an observed environment conflict, not proof of the sole cause. A clone
  a lightweight runtime `datamil-paper-h15-cpu-loader` inherits the existing
  Conda packages and overrides only CPU Torch/torchvision for the streaming
  loader; JAX retains CUDA. Original `datamil` is not changed. A slow full-clone
  attempt was stopped; its partial directory `envs/datamil-paper-h15` is preserved
  but unused. The wrapper switches only paper-H15 children to the new runtime
  once `environment.ready.json` exists.
- Cleaning up CUDA imports alone did not fix the cuDNN proxy-gradient failure.
  `XLA_FLAGS=--xla_gpu_autotune_level=0` selects a compatible kernel path; the
  actual GPU proxy gradient then passed with finite loss. Algorithm parameters
  are unchanged. Full GPU VJP subsequently PASSED: 4 learner steps, last2
  differentiated, effective batch512, finite influences on69 candidate clusters.
  Timings (including first compilation): forward46.75s, backward402.38s.
  This is a fixture smoke, not a paper performance result. Numerical proof:
  `logs/datamil_paper_h15/gpu_smoke_verified.json`.
- `zstandard` in the isolated runtime avoids the multi-minute zlib compression
  of large JAX cache executables observed in a Python stack trace.
- Production DataMIL uses one bounded JAX pool (0.80 of VRAM, preallocated) to
  leave room for AMF and approach the requested ~90% total occupancy. Occupancy
  includes allocator/cache reservation and is NOT live tensor usage. GPU smoke
  remains non-preallocated. No scientific batch or step count was increased.
- Actual small-fixture GPU VJP smoke has separate data/run ID and is never counted
  as a paper experiment. See `logs/datamil_paper_h15/gpu_smoke_console.log`.
- Partial validation batches count actual sample sizes in paper mode.

## Launch / resume

Screen: `ldva-datamil-paper-h15`; source `scratchpad/datamil_paper_h15_pipeline.py`.
Do not launch a duplicate while its process is active.

```bash
cd /mnt/data/thanhpv/ldva
SCREENDIR=/mnt/data/thanhpv/ldva/logs/screen screen -ls
cat logs/datamil_paper_h15/status.json
# After the previous allocation/processes have ended, on a GPU allocation:
SCREENDIR=/mnt/data/thanhpv/ldva/logs/screen screen -dmS ldva-datamil-paper-h15 bash scratchpad/run_datamil_paper_h15.sh
```

Resume units: completed RLDS shard; completed MDS conversion stage; completed
datamodel iteration; datamodel forward optimizer state every100 steps; policy
state every1000 steps. An interrupted backward phase reuses its saved forward
phase but recomputes the backward work. Array checkpoint roundtrip is tested;
bitwise full dataloader replay for a datamodel interruption is NOT yet verified.
An interrupted MDS conversion currently starts a fresh build attempt and preserves
the partial attempt instead of deleting it.

W&B pipeline: https://wandb.ai/hust_edu_vn/paper-reproduction/runs/datamil-paper-h15-t30-pipeline
Durable status/logs: `logs/datamil_paper_h15/`; errors also append to
`logs/reproduction_errors.txt`. Failure stops the affected stage, not AMF.
Report watcher updates `scratchpad/reproduction_overnight_report.md` every20min.
Slurm allocation still ends 2026-11-04 05:51:18 UTC; detaching does not extend it.
