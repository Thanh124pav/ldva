# CPU-only reproduction status (2026-11-03)

## Scope

- Workspace: `/mnt/data/thanhpv/ldva`
- Workspace branch: `acquisition`
- GPU work is intentionally paused until GPU access is restored.
- Existing user changes in the workspace were preserved.

## Source revisions

- DataMIL: `f8fc4abccacd1e2b070e68128a7f2bce98f4b303`
- AMF: `5f76b5ec6d834d9a77e2ccce8545504a10523293`
- MetaWorld compatibility checkout: `cca35cff0ec62f1a18b11440de6b09e2d10a1380`
- Diffusion Policy: `5ba07ac6661db573af695b419a7947ecb704690f`

## DataMIL

- Reused `/mnt/data/thanhpv/datamil_data` (about 165 GB).
- Found 10 completed Book-Caddy datamodel trials, each with 2,000,000 finite
  metagradient values.
- Re-ran the released aggregation/selection stage on CPU.
- Output: 450 unique trajectories selected from 4,500 LIBERO-90 trajectories
  (top 10%).
- The selected set spans 86 of 90 source tasks and is concentrated on
  Book-Caddy-like tasks, consistent with the qualitative result in the paper.
- Found five existing Octo checkpoints (seeds 0-4), each trained for 10,000
  steps. No policy evaluation result files were present.
- Full datamodel training, Octo training, and LIBERO rollout evaluation are
  waiting for GPU access.
- The official MetaWorld data link on `utexas.box.com` is unreachable from this
  machine (TCP/443 connection timeout), so that dataset could not be downloaded.

## AMF data

### Robomimic

- Downloaded the four official v1.5 low-dimensional PH datasets: Lift, Can,
  Square, and Tool Hang.
- Converted all four to absolute actions with 8 CPU workers after adding a
  compatibility adapter for the Robosuite 1.5 composite-controller API.
- Validation: 800 episodes, 158,989 transitions, 149,813 training sequences;
  all sampled tensors are finite.

### Kitchen

- Downloaded all three seeds for Micro, Left Door, Knob 2 Off, and Knob 1 On.
- Processed the seed-1 files required by the released AMF script.
- Validation: four available tasks, 25 demonstrations per task, 50 transitions
  per demonstration (5,000 transitions total), all finite.
- Missing: `FK1_SdoorOpenRandom_v2d-v4/seed1/`
  `FK1_SdoorOpenRandom_v2d-v420230402-210417_trace.h5`.
- The public Google Drive file failed both folder and direct-file downloads due
  to Drive access/quota behavior.

### MetaWorld

- Generated the released low-dimensional dataset on CPU.
- Validation: four tasks, 40 demonstrations per task, 60 transitions per
  demonstration (9,600 transitions total), all finite.
- Mean return sums of generated demonstrations:
  - cup_left: 11.525
  - cup_right: 8.525
  - faucet_open: 5.15
  - faucet_close: 3.4

## AMF environment and CPU checks

- Main env: `/mnt/data/thanhpv/miniconda3/envs/amf`
- Conversion/MetaWorld env: `/mnt/data/thanhpv/miniconda3/envs/amf-convert39`
- A separate conversion env was necessary because AMF pins Robosuite 1.2.0,
  while the currently published Robomimic datasets declare Robosuite 1.5.1.
- PyTorch3D 0.7.2 was built successfully for CPU.
- PyAV 10 could not build without system FFmpeg headers; PyAV 12.3.0 is used.
- Generated 384-dimensional MiniLM task embeddings for every released task.
- AMF Robomimic CPU training smoke test passed for one batch:
  `train_loss=1.2102320194244385`, `lr=0.0001`.
- AMF MetaWorld MLP CPU training smoke test passed for one batch:
  `train_loss=0.2930033802986145`, `lr=0.0001`.
- AMF Kitchen MLP CPU training smoke test passed on the four available tasks:
  `train_loss=4.83595609664917`, `lr=0.0001`.
- The AMF-GP job finished all 12 prior-support settings and 10 seeds for both
  methods, then failed in notebook plotting because LaTeX `type1ec.sty` is
  missing. The wrapper printed its summary only after plotting and did not
  checkpoint metrics; numerical metrics from this run were lost when it exited.
  This is an incomplete reproduction, not a successful result. The full error
  and progress log are preserved in `logs/amf_gp_reproduction.log`.
- No GP job remains running. The user requested a session reset before further
  experiments. Do not start GPU work until the user confirms GPU restoration.
- Before rerunning GP, save metrics after every seed and before plotting, use
  Matplotlib text without external LaTeX, and propagate pipeline failures.
- Remediation implemented in `external/amf/reproduce_gp.py`: atomic checkpoint
  after each Active/Uniform run, JSON and Markdown summaries immediately,
  exact NumPy RNG recovery, configuration mismatch checks, and plotting only
  after saving with external LaTeX disabled. A simulated interruption between
  Active and Uniform resumed bit-for-bit identically to uninterrupted results.
  A real one-round GP smoke test wrote all three files under
  `external/amf/outputs/gp_save_smoke/`. These are smoke-test results only.
- Full GP rerun has NOT been started, pending the requested session reset.
  To rerun after reset from `external/amf`, use:
  `MPLCONFIGDIR=/mnt/data/thanhpv/tmp/matplotlib-amf /mnt/data/thanhpv/miniconda3/envs/acvcl/bin/python -u reproduce_gp.py --no-plots`.
  Run this directly, or use `set -o pipefail` if piping to tee. Restarting the
  same command resumes `outputs/gp/checkpoint.pkl`.
- A local benchmark measured 29.24 seconds for one full AMF-GP active run at
  budget 40. With 240 active/uniform runs in the notebook, expected wall time
  on this host is roughly 80-100 minutes.
- MuJoCo headless processes emit `free(): invalid pointer` during interpreter
  teardown on this host. Data generation finishes and files validate before the
  teardown error; this does not affect the generated HDF5 files.

## Logs

- `logs/amf_gp_reproduction.log`
- `logs/robomimic_convert_{lift,can,square,tool_hang}.log`
- `logs/kitchen_collect.log`
- `logs/metaworld_collect.log`
- `logs/robohive_init.log`
