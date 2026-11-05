"""Cheap CPU-only reporting; no LLM calls. Results survive session resets."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import subprocess
import sys
import numpy as np

ROOT = Path('/mnt/data/thanhpv/ldva')

def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default

def report():
    now = datetime.now(timezone.utc).isoformat()
    base = ROOT / 'external/amf/outputs/metaworld_paper'
    lines = ['# Reproduction overnight report', '', f'Updated UTC: {now}', '',
             '## Scope and interpretation', '',
             '- Branch: acquisition. Smoke tests prove execution, not agreement with the paper.',
             '- AMF MetaWorld: 10 seeds × matched/mismatched priors × AMF / uniform+prior / uniform = 60 runs.',
             '- Protocol: 200 pretraining epochs; 20 acquisition rounds; 3000 gradient steps/round; batch 256; LR 1e-4; 2×256 MLP; 50 evaluation episodes/task; weighted adaptive prior.',
             '- DataMIL old trajectory/10-trial policy training was STOPPED by user request. Old checkpoints and evaluations below are archived results, not paper-matched runs.',
             '- Replacement DataMIL pipeline: horizon15 clusters, 30 NEW datamodel trials, 2100 learner steps/trial (last100 differentiated), common prior normalization, paper weighted proxy; ONE policy seed, batch128, 10k steps, target/prior 1:1, 50 evaluation rollouts.',
             '- AMF GP: both released notebook (20 shooting candidates) and appendix shooting budget (100 candidates) run in separate output folders.', '',
             '## Resources', '']
    resource = read_json(ROOT/'logs/overnight/status.json', {}).get('resources', {})
    vram_controller = read_json(ROOT/'logs/vram_target_90/status.json', {})
    if vram_controller and not (ROOT/'logs/vram_target_90/stopped_by_user.json').exists():
        lines.append(f'- VRAM-90 controller: {vram_controller.get("measured_vram_pct", 0):.2f}% measured; target 90%; includes JAX allocator/cache. Active seed pools: {vram_controller.get("active", {})}. The old sequential fresh-training launcher is intentionally paused to avoid duplicates.')
    lines += [f'- Latest GPU utilization: {resource.get("gpu_util_pct", "unknown")}%; VRAM: {resource.get("gpu_memory_mib", "unknown")} MiB ({resource.get("gpu_memory_pct", 0):.2f}%).',
              '- Four small MLP workers overlap rollout and training. Small VRAM occupancy is expected; throughput and GPU activity determine useful utilization.',
              '- Resource CPU percentage is HOST-wide, not normalized to this job allocation (16 available CPUs).', '',
              '## Running and queued', '']
    queue = read_json(ROOT/'logs/paper_queue/status.json', {})
    if queue:
        lines.append(f'- Detached paper queue: Slurm {queue.get("slurm_job_id")}; supervisor PID {queue.get("pid")}; heartbeat {queue.get("updated")}. Status: logs/paper_queue/status.json. Completed GP and old DataMIL training are not relaunched.')
        for name, job in queue.get('jobs', {}).items():
            lines.append(f'  - {name}: PID {job.get("pid")}; exit={job.get("exit_code")}.')
    paper_status = read_json(ROOT/'logs/datamil_paper_h15/status.json', {})
    if paper_status:
        lines.append(f'- DataMIL paper H15/T30: {paper_status.get("stage")}; PID {paper_status.get("child_pid")}; exit={paper_status.get("exit_code")}; status timestamp={paper_status.get("updated")}; log={paper_status.get("log")}.')
        paper_dm = Path('/mnt/data/thanhpv/datamil_data/paper_h15_t30/exp/datamodels/libero90_horizon/book-caddy')
        lines.append(f'- DataMIL paper datamodel trials completed: {len(list(paper_dm.glob("iter_*/complete.json")))}/30. Train one policy seed only after selection completes.')
        paper_environment = read_json(ROOT/'logs/datamil_paper_h15/environment_status.json', {})
        if paper_environment:
            lines.append(f'- DataMIL environment: {paper_environment.get("state")}; {paper_environment.get("destination")}; original environment preserved.')
    for path in sorted(base.glob('status_worker*.json')):
        status = read_json(path, {})
        lines.append(f'- {path.stem}: {status.get("current", "complete")}; next: {status.get("next")}; exit: {status.get("exit_code", "running")}.')
        if status.get('current'):
            metrics = base/status['current']/'metrics.jsonl'
            try:
                latest = json.loads(metrics.read_text().splitlines()[-1])
                lines.append(f'  - Latest epoch {latest["epoch"]}/219; round time {latest.get("timing/epoch_seconds", 0):.1f}s; most recent success {latest.get("mean_score", "not evaluated")}.')
            except (OSError, ValueError, IndexError):
                pass
    fixed_smoke = read_json(ROOT/'logs/amf_metaworld_fix/cpu_smoke_passed.json', {})
    if fixed_smoke:
        lines += ['', '### AMF MetaWorld data-fix validation', '',
                  '- Success-only collector saves a separate seeded dataset: external/amf/data/metaworld_success_v1. Original datasets and completed runs preserved.',
                  '- CPU train/select/eval/checkpoint/W&B smoke passed on 160 successful demonstrations. This is an execution test, NOT a new paper result.',
                  '- Diagnosis: scratchpad/amf_metaworld_diagnosis_20261104.md; tests/logs: logs/amf_metaworld_fix/.']
    clean_run = read_json(ROOT/'logs/amf_metaworld_fix/train_seed0_status.json', {})
    if clean_run:
        latest = clean_run.get('latest_metrics') or {}
        lines += [f'- Corrected-data AMF mismatch seed0: device CPU, 2 threads; PID {clean_run.get("child_pid")}; exit={clean_run.get("exit_code")}; epoch={latest.get("epoch", "initializing")}/219; latest score={latest.get("mean_score", "not evaluated")}; heartbeat={clean_run.get("updated")}.',
                  '- Corrected-data full budget: pretrain200 epochs, 20×3000 fine-tuning steps, batch256, MLP2×256, beta.01, evaluation50/task. Separate outputs under external/amf/outputs/metaworld_success_v1.']
    lines += ['', '## AMF MetaWorld completed results', '',
              'Means and 90% bootstrap confidence intervals across completed seeds only; partial cohorts are not final comparisons.', '',
              '| Regime | Method | Seeds | Final success | 90% CI |',
              '|---|---|---:|---:|---|']
    groups = defaultdict(list)
    completed = 0
    for path in sorted(base.glob('*/complete.json')):
        result = read_json(path)
        if result is None:
            continue
        name = path.parent.name.split('-')
        groups[(name[2], name[3])].append(result['mean_score'])
        completed += 1
    rng = np.random.default_rng(0)
    for (regime, method), values in sorted(groups.items()):
        means = rng.choice(values, (10000, len(values))).mean(axis=1)
        low, high = np.quantile(means, [.05, .95])
        interval = f'[{low:.4f}, {high:.4f}]' if len(values)>1 else 'n/a (one seed)'
        lines.append(f'| {regime} | {method} | {len(values)}/10 | {np.mean(values):.4f} | {interval} |')
    lines += ['', f'Completed: {completed}/60. No claim of matching the paper until the seed cohort and learning curves are checked.', '',
              '## DataMIL results', '']
    aggregate = read_json(ROOT/'logs/datamil_book_caddy_eval/aggregate.json', {})
    for name, value in aggregate.items():
        lines.append(f'- {name}: {value:.2%}')
    if aggregate:
        values = list(aggregate.values())
        lines.append(f'- Across {len(values)}/5 checkpoints: mean={np.mean(values):.2%}, std={np.std(values):.2%}. Reference Book-Caddy table: 82.4 ± 1.7%; uncertainty definitions/protocol must be verified before direct comparison.')
    else:
        lines.append('- Full evaluation has not produced a completed seed yet. See datamil_smoke_diagnostic.txt for smoke progress/errors.')
    lines += ['', '### Archived trajectory/10-trial DataMIL training (stopped)', '']
    pilot = read_json(ROOT/'logs/overnight/datamil_pilot_summary.json', {})
    if pilot:
        lines.append(f'- Verified batch-128 pilot: {pilot["median_steady_seconds_per_step"]:.3f} s/step; peak allocator {pilot.get("peak_allocator_gb")} GiB; estimated five-seed train-only time {pilot["estimated_five_seed_training_hours"]:.2f} hours, excluding startup/checkpoint/evaluation overhead and future contention.')
    for metrics in sorted((ROOT/'logs/datamil_fresh/full').glob('**/training_metrics.jsonl')):
        try:
            latest = json.loads(metrics.read_text().splitlines()[-1])
            config_path = metrics.parent/'finetune_config.json'
            stale = config_path.exists() and metrics.stat().st_mtime < config_path.stat().st_mtime
            warning = 'OLD metrics before resume; awaiting a new measurement. ' if stale else ''
            lines.append(f'- {metrics.parent.name}: {warning}{latest["progress/completed_steps"]}/10000 steps; loss={latest["training/loss"]:.5f}; elapsed {latest["progress/elapsed_wall_seconds"]:.1f}s.')
        except (OSError, ValueError, IndexError):
            pass
    fresh = read_json(ROOT/'logs/datamil_fresh/evaluation/aggregate.json', {})
    for name, value in fresh.items():
        lines.append(f'- Fresh evaluation {name}: {value:.2%}.')
    gp = ROOT/'external/amf/outputs/gp_reproduction_v2/results.md'
    if gp.exists():
        lines += ['', '## AMF GP checkpointed results', '', gp.read_text()]
    gp100 = ROOT/'external/amf/outputs/gp_paper100/results.md'
    if gp100.exists():
        lines += ['', '### Appendix 100-candidate variant', '', gp100.read_text()]
    lines += ['', '## AMF Robomimic diffusion', '',
              '- A successful batch-8 smoke is complete. A separate batch-256/100-denoise-step pilot gates one full matched-prior seed: 200 pretraining epochs, 20×3000 finetuning steps, 20 queried demos/round, 25 evaluation episodes/task. Downloaded simulator/data v1.5 differs from the released AMF simulator v1.2.', '']
    for directory in sorted((ROOT/'external/amf/outputs').glob('amf-robomimic-v15-*')):
        metrics = directory/'metrics.jsonl'
        try:
            latest = json.loads(metrics.read_text().splitlines()[-1])
            lines.append(f'- {directory.name}: epoch {latest["epoch"]}; score={latest.get("mean_score", "not evaluated")}; peak allocated={latest.get("gpu/peak_allocated_gb")} GiB. Pilot scores are NOT paper results.')
        except (OSError, ValueError, IndexError):
            pass
    lines += ['', '## Data and environment status', '',
              '- No active dataset download or environment installation is managed by this reporter.',
              '- Available: existing DataMIL LIBERO data/checkpoints; all four AMF MetaWorld task datasets; four Robomimic datasets; four of five Kitchen tasks.',
              '- Missing: Kitchen Sdoor (public Google Drive access/quota); official DataMIL MetaWorld UTBox download (connection timeout). Real-robot experiments require hardware.',
              '- Existing environments reused: datamil (JAX/Flax), amf (PyTorch CUDA), acvcl (CPU GP). amf-convert39 isolates Robosuite-1.5 dataset conversion.', '',
              '## Fixes and caveats', '',
              '- GP metrics, RNG and checkpoint written atomically after every method/seed; reports generated before optional plotting.',
              '- AMF fixes: CUDA embedding device; differentiable holdout policy for acquisition; vector-env reward retention; renderer API; durable epoch checkpoints/RNG; paper-configurable MLP layout, gradient budget and weighted mixture.',
              '- DataMIL: TensorFlow CPU preprocessing/JAX GPU inference; disabled unrelated Torch Transformers backend after native Triton crash; explicit trusted LIBERO init-state loading for PyTorch-2.6+ compatibility.',
              '- MetaWorld demonstrations were generated by the released script (which does not seed generation); exact paper trajectories are unavailable. This may affect numerical agreement.', '',
              '## Logs and links', '',
              '- W&B: https://wandb.ai/hust_edu_vn/paper-reproduction',
              '- Current resume owner: scratchpad/resume_paper_queue.py; DataMIL: scratchpad/run_datamil_paper_h15.sh. Legacy launchers must not be used to restart old DataMIL training.',
              '- Errors: logs/reproduction_errors.txt; per-run console.log contains complete stderr/stdout.',
              '- Resources: logs/overnight/resources.jsonl; workers: external/amf/outputs/metaworld_paper/.',
              '- GP: external/amf/outputs/gp_reproduction_v2/{results.md,summary.json,checkpoint.pkl}.',
              '- Paper references: https://arxiv.org/html/2410.05026v3 and https://github.com/UT-Austin-RobIn/datamil', '']
    target = ROOT/'scratchpad/reproduction_overnight_report.md'
    temporary = target.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(target)
    return dict(amf_completed=completed, datamil_completed=len(aggregate))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        if not args.watch:
            print(json.dumps(report()), flush=True)
            break
        # Reload report code each heartbeat so fixes are picked up safely.
        subprocess.run([sys.executable, __file__], check=False)
        time.sleep(1200)
