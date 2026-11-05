"""Validate pilot output and estimate useful steady-state throughput."""
import json
import math
from pathlib import Path
import statistics

root = Path('/mnt/data/thanhpv/ldva/logs/datamil_fresh/pilot')
paths = sorted(root.glob('**/training_metrics.jsonl'), key=lambda path: path.stat().st_mtime)
assert paths, 'No pilot metrics saved'
metrics = [json.loads(line) for line in paths[-1].read_text().splitlines()]
assert len(metrics) == 20 and metrics[-1]['progress/completed_steps'] == 20
assert (paths[-1].parent/'20').is_dir(), 'Pilot checkpoint missing'
assert all(math.isfinite(row['training/loss']) for row in metrics)
durations = [right['progress/elapsed_wall_seconds'] - left['progress/elapsed_wall_seconds']
             for left, right in zip(metrics[-11:-1], metrics[-10:])]
seconds = statistics.median(durations)
summary = dict(batch_size=128, measured_steps=20,
               median_steady_seconds_per_step=seconds,
               estimated_10k_training_hours=seconds*10000/3600,
               estimated_five_seed_training_hours=seconds*50000/3600,
               excludes='checkpoint, evaluation and per-process JIT startup overhead',
               peak_allocator_gb=metrics[-1].get('gpu/peak_bytes_in_use_gb'),
               metrics_file=str(paths[-1]))
Path('/mnt/data/thanhpv/ldva/logs/overnight/datamil_pilot_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
