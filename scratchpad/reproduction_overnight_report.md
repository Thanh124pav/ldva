# Reproduction overnight report

Updated UTC: 2026-11-05T08:52:25.840588+00:00

## Scope and interpretation

- Branch: acquisition. Smoke tests prove execution, not agreement with the paper.
- AMF MetaWorld: 10 seeds × matched/mismatched priors × AMF / uniform+prior / uniform = 60 runs.
- Protocol: 200 pretraining epochs; 20 acquisition rounds; 3000 gradient steps/round; batch 256; LR 1e-4; 2×256 MLP; 50 evaluation episodes/task; weighted adaptive prior.
- DataMIL old trajectory/10-trial policy training was STOPPED by user request. Old checkpoints and evaluations below are archived results, not paper-matched runs.
- Replacement DataMIL pipeline: horizon15 clusters, 30 NEW datamodel trials, 2100 learner steps/trial (last100 differentiated), common prior normalization, paper weighted proxy; ONE policy seed, batch128, 10k steps, target/prior 1:1, 50 evaluation rollouts.
- AMF GP: both released notebook (20 shooting candidates) and appendix shooting budget (100 candidates) run in separate output folders.

## Resources

- Latest GPU utilization: 0.0%; VRAM: 65569.0 MiB (80.39%).
- Four small MLP workers overlap rollout and training. Small VRAM occupancy is expected; throughput and GPU activity determine useful utilization.
- Resource CPU percentage is HOST-wide, not normalized to this job allocation (16 available CPUs).

## Running and queued

- Detached paper queue: Slurm 64276; supervisor PID 1777803; heartbeat 1793868733.6587362. Status: logs/paper_queue/status.json. Completed GP and old DataMIL training are not relaunched.
  - datamil_h15_t30: PID 1777805; exit=None.
  - resources: PID 1777806; exit=None.
  - report20min: PID 1777807; exit=None.
- DataMIL paper H15/T30: datamodel_trial16; PID 1778300; exit=None; status timestamp=1793868737.602769; log=/mnt/data/thanhpv/ldva/logs/datamil_paper_h15/datamodel_trial16.log.
- DataMIL paper datamodel trials completed: 16/30. Train one policy seed only after selection completes.
- DataMIL environment: ready_actual_gpu_vjp_smoke_passed; /mnt/data/thanhpv/miniconda3/envs/datamil-paper-h15-cpu-loader; original environment preserved.
- status_worker0: complete; next: None; exit: running.
- status_worker1: complete; next: None; exit: running.
- status_worker2: complete; next: None; exit: running.
- status_worker3: complete; next: None; exit: running.

### AMF MetaWorld data-fix validation

- Success-only collector saves a separate seeded dataset: external/amf/data/metaworld_success_v1. Original datasets and completed runs preserved.
- CPU train/select/eval/checkpoint/W&B smoke passed on 160 successful demonstrations. This is an execution test, NOT a new paper result.
- Diagnosis: scratchpad/amf_metaworld_diagnosis_20261104.md; tests/logs: logs/amf_metaworld_fix/.
- Corrected-data AMF mismatch seed0: device CPU, 2 threads; PID 1008191; exit=0; epoch=219/219; latest score=0.5; heartbeat=1793812383.3315356.
- Corrected-data full budget: pretrain200 epochs, 20×3000 fine-tuning steps, batch256, MLP2×256, beta.01, evaluation50/task. Separate outputs under external/amf/outputs/metaworld_success_v1.

## AMF MetaWorld completed results

Means and 90% bootstrap confidence intervals across completed seeds only; partial cohorts are not final comparisons.

| Regime | Method | Seeds | Final success | 90% CI |
|---|---|---:|---:|---|
| matched | posterior_grad | 10/10 | 0.4965 | [0.4630, 0.5300] |
| matched | uniform | 10/10 | 0.4820 | [0.4525, 0.5080] |
| matched | uniform_prior | 10/10 | 0.5325 | [0.4975, 0.5620] |
| mismatch | posterior_grad | 10/10 | 0.5000 | [0.4655, 0.5340] |
| mismatch | uniform | 10/10 | 0.4400 | [0.4100, 0.4700] |
| mismatch | uniform_prior | 10/10 | 0.5525 | [0.5000, 0.6055] |

Completed: 60/60. No claim of matching the paper until the seed cohort and learning curves are checked.

## DataMIL results

- seed0_libero90_book-caddy_20261021_161546: 56.00%
- seed1_libero90_book-caddy_20261021_170207: 44.00%
- seed2_libero90_book-caddy_20261021_174743: 46.00%
- seed3_libero90_book-caddy_20261021_183319: 50.00%
- seed4_libero90_book-caddy_20261021_191857: 46.00%
- Across 5/5 checkpoints: mean=48.40%, std=4.27%. Reference Book-Caddy table: 82.4 ± 1.7%; uncertainty definitions/protocol must be verified before direct comparison.

### Archived trajectory/10-trial DataMIL training (stopped)

- Verified batch-128 pilot: 0.219 s/step; peak allocator 27.370962142944336 GiB; estimated five-seed train-only time 3.04 hours, excluding startup/checkpoint/evaluation overhead and future contention.
- seed0_fresh_libero90_book-caddy_20261103_185027: 10000/10000 steps; loss=0.58570; elapsed 2258.6s.
- seed1_fresh_libero90_book-caddy_20261104_021157: 10000/10000 steps; loss=0.47486; elapsed 3828.0s.
- seed2_fresh_libero90_book-caddy_20261104_021225: 10000/10000 steps; loss=0.49948; elapsed 3813.9s.
- seed3_fresh_libero90_book-caddy_20261104_031654: 5700/10000 steps; loss=0.67190; elapsed 2233.4s.
- seed4_fresh_libero90_book-caddy_20261104_031719: 5500/10000 steps; loss=0.57125; elapsed 2182.8s.
- Fresh evaluation seed0_fresh_libero90_book-caddy_20261103_185027: 42.00%.
- Fresh evaluation seed1_fresh_libero90_book-caddy_20261104_021157: 36.00%.

## AMF GP checkpointed results

# AMF-GP results

Partial results; smoke tests are not paper reproductions.

| Prior support | Method | Completed seeds | Final return mean | Std |
|---|---|---:|---:|---:|
| 1 | active | 10 | -7.92569845 | 0.12890256 |
| 1 | uniform | 10 | -8.79994700 | 0.50886158 |
| 2 | active | 10 | -7.90162653 | 0.11825049 |
| 2 | uniform | 10 | -8.45059498 | 0.33236462 |
| 3 | active | 10 | -7.80113635 | 0.07744257 |
| 3 | uniform | 10 | -8.36025047 | 0.21365527 |
| 4 | active | 10 | -7.77217705 | 0.06666467 |
| 4 | uniform | 10 | -8.57102242 | 0.58807214 |
| 5 | active | 10 | -7.74101115 | 0.09037694 |
| 5 | uniform | 10 | -8.30236687 | 0.35846715 |
| 6 | active | 10 | -7.74149119 | 0.09393236 |
| 6 | uniform | 10 | -8.25580050 | 0.31638202 |
| 7 | active | 10 | -7.71293357 | 0.08519052 |
| 7 | uniform | 10 | -8.46984525 | 0.52715424 |
| 8 | active | 10 | -7.68852163 | 0.08898566 |
| 8 | uniform | 10 | -8.07013558 | 0.19093841 |
| 9 | active | 10 | -7.66392689 | 0.08403288 |
| 9 | uniform | 10 | -7.90427267 | 0.23332188 |
| 10 | active | 10 | -7.65410875 | 0.07256880 |
| 10 | uniform | 10 | -7.85637392 | 0.30718311 |
| 11 | active | 10 | -7.60146600 | 0.07375313 |
| 11 | uniform | 10 | -7.75812604 | 0.11882937 |
| 12 | active | 10 | -7.58455329 | 0.03853882 |
| 12 | uniform | 10 | -7.76199069 | 0.10014260 |


### Appendix 100-candidate variant

# AMF-GP results

Partial results; smoke tests are not paper reproductions.

| Prior support | Method | Completed seeds | Final return mean | Std |
|---|---|---:|---:|---:|
| 1 | active | 10 | -7.89616726 | 0.06071218 |
| 1 | uniform | 10 | -8.72497067 | 0.48820976 |
| 2 | active | 10 | -7.91589487 | 0.08616809 |
| 2 | uniform | 10 | -8.65950854 | 0.38528184 |
| 3 | active | 10 | -7.80587063 | 0.04308854 |
| 3 | uniform | 10 | -8.24732988 | 0.31054760 |
| 4 | active | 10 | -7.76738432 | 0.09257452 |
| 4 | uniform | 10 | -8.21772139 | 0.27020907 |
| 5 | active | 10 | -7.73551992 | 0.13340825 |
| 5 | uniform | 10 | -8.10853059 | 0.21316163 |
| 6 | active | 10 | -7.69132808 | 0.08742851 |
| 6 | uniform | 10 | -8.25484843 | 0.46600522 |
| 7 | active | 10 | -7.63876729 | 0.07027471 |
| 7 | uniform | 10 | -8.09345763 | 0.34482821 |
| 8 | active | 10 | -7.65725760 | 0.08291755 |
| 8 | uniform | 10 | -7.93885269 | 0.30473213 |
| 9 | active | 10 | -7.66861898 | 0.09986411 |
| 9 | uniform | 10 | -8.16412988 | 0.49234378 |
| 10 | active | 10 | -7.65641998 | 0.07747946 |
| 10 | uniform | 10 | -7.78213619 | 0.17264919 |
| 11 | active | 10 | -7.61545788 | 0.07117113 |
| 11 | uniform | 10 | -7.85762902 | 0.20268773 |
| 12 | active | 10 | -7.59280753 | 0.06772240 |
| 12 | uniform | 10 | -7.70503403 | 0.09411892 |


## AMF Robomimic diffusion

- A successful batch-8 smoke is complete. A separate batch-256/100-denoise-step pilot gates one full matched-prior seed: 200 pretraining epochs, 20×3000 finetuning steps, 20 queried demos/round, 25 evaluation episodes/task. Downloaded simulator/data v1.5 differs from the released AMF simulator v1.2.

- amf-robomimic-v15-full-matched-seed0: epoch 219; score=0.8099999999999999; peak allocated=2.986268997192383 GiB. Pilot scores are NOT paper results.
- amf-robomimic-v15-pilot-matched-seed0: epoch 1; score=0.0; peak allocated=2.9890127182006836 GiB. Pilot scores are NOT paper results.

## Data and environment status

- No active dataset download or environment installation is managed by this reporter.
- Available: existing DataMIL LIBERO data/checkpoints; all four AMF MetaWorld task datasets; four Robomimic datasets; four of five Kitchen tasks.
- Missing: Kitchen Sdoor (public Google Drive access/quota); official DataMIL MetaWorld UTBox download (connection timeout). Real-robot experiments require hardware.
- Existing environments reused: datamil (JAX/Flax), amf (PyTorch CUDA), acvcl (CPU GP). amf-convert39 isolates Robosuite-1.5 dataset conversion.

## Fixes and caveats

- GP metrics, RNG and checkpoint written atomically after every method/seed; reports generated before optional plotting.
- AMF fixes: CUDA embedding device; differentiable holdout policy for acquisition; vector-env reward retention; renderer API; durable epoch checkpoints/RNG; paper-configurable MLP layout, gradient budget and weighted mixture.
- DataMIL: TensorFlow CPU preprocessing/JAX GPU inference; disabled unrelated Torch Transformers backend after native Triton crash; explicit trusted LIBERO init-state loading for PyTorch-2.6+ compatibility.
- MetaWorld demonstrations were generated by the released script (which does not seed generation); exact paper trajectories are unavailable. This may affect numerical agreement.

## Logs and links

- W&B: https://wandb.ai/hust_edu_vn/paper-reproduction
- Current resume owner: scratchpad/resume_paper_queue.py; DataMIL: scratchpad/run_datamil_paper_h15.sh. Legacy launchers must not be used to restart old DataMIL training.
- Errors: logs/reproduction_errors.txt; per-run console.log contains complete stderr/stdout.
- Resources: logs/overnight/resources.jsonl; workers: external/amf/outputs/metaworld_paper/.
- GP: external/amf/outputs/gp_reproduction_v2/{results.md,summary.json,checkpoint.pkl}.
- Paper references: https://arxiv.org/html/2410.05026v3 and https://github.com/UT-Austin-RobIn/datamil
