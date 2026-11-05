"""Measured diffusion pilot, then one full matched-prior AMF seed."""
import json
from pathlib import Path
import subprocess
import sys
import yaml

root = Path('/mnt/data/thanhpv/ldva/external/amf')
base = yaml.safe_load((root/'reproduction_robomimic_smoke.yaml').read_text())
for phase in ['pilot', 'full']:
    name = f'amf-robomimic-v15-{phase}-matched-seed0'
    directory = root/'outputs'/name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'complete.json').exists():
        print(f'SKIP completed {name}', flush=True)
        continue
    cfg = dict(base, exp_name=name, working_dir=str(directory), batch_size=256,
               num_inference_steps_override=100, finetune_prune_trajs=0.1,
               pre_demos={task:160 if phase=='full' else 4 for task in ['lift','can','square','tool_hang']},
               pre_epochs=200 if phase=='full' else 1,
               rollout_every=200 if phase=='full' else 1,
               finetune_iters=20 if phase=='full' else 1,
               finetune_gradient_steps=3000 if phase=='full' else 2,
               eval_episodes_override=25 if phase=='full' else 1,
               num_parallel_envs=25 if phase=='full' else 1,
               repeat_demos_override=20 if phase=='full' else 1,
               wandb_group='amf-robomimic-v15-paper' if phase=='full' else 'amf-robomimic-smoke',
               protocol='Appendix M budgets; downloaded v1.5 dataset/Robosuite 1.5, not original 1.2 simulator' if phase=='full' else 'batch256 and 100 diffusion steps; shortened training/rollout pilot')
    if phase=='full':
        del cfg['eval_max_steps_override']
    config = directory/'input.yaml'
    config.write_text(yaml.safe_dump(cfg))
    with (directory/'console.log').open('a') as log:
        result = subprocess.run([sys.executable,'-u','main.py',str(config)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        with Path('/mnt/data/thanhpv/ldva/logs/reproduction_errors.txt').open('a') as errors:
            errors.write(f'\n{name} failed: exit {result.returncode}\n'+(directory/'console.log').read_text()[-12000:])
        raise RuntimeError(f'{name} failed; no downstream job started')
    rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    assert rows[-1]['epoch'] == (219 if phase=='full' else 1)
    (directory/'complete.json').write_text(json.dumps(rows[-1], indent=2))
    if phase=='pilot':
        # Reserve at least ~40 GiB for Octo and the four small MLP workers.
        assert rows[-1]['gpu/peak_allocated_gb'] < 35, 'Reduce concurrency before full diffusion training'
    print(f'COMPLETED {name}: '+json.dumps(rows[-1]), flush=True)
