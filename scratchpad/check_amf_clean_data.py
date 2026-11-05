"""Validate isolated clean datasets and run a short CPU/W&B execution smoke."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import yaml
from ml_collections import ConfigDict

ROOT = Path('/mnt/data/thanhpv/ldva')
CODE = ROOT/'external/amf'
sys.path.insert(0, str(CODE))
from scripts.collect_metaworld import successful_demo, TASKS

data = CODE/'data/metaworld_success_v1'
assert (data/'collection_manifest.json').exists(), 'Collection must finish first'
audit = {}
old_hashes = {}
for task in TASKS:
    relative = Path('metaworld/datasets')/task/'ph/low_dim_abs.hdf5'
    old = CODE/'data'/relative
    old_hashes[str(old)] = hashlib.sha256(old.read_bytes()).hexdigest()
    with h5py.File(data/relative) as f:
        assert f.attrs['success_only']
        assert len(f['data']) == 40
        for demo in f['data'].values():
            values = {k: v[:] for k, v in demo.items()}
            assert successful_demo(values)
            assert values['obs'].shape == (60,28)
            assert values['actions'].shape == (60,4)
            np.testing.assert_array_equal(values['dones'], np.r_[np.zeros(59),1.])
            assert np.unique(values['collection_seed']).size == 1
        audit[task] = dict(successful_demos=40, attempts=int(f.attrs['attempts']))
print('PASS clean HDF5 validation: '+json.dumps(audit), flush=True)

output = CODE/'outputs/amf-metaworld-success-v1-cpu-smoke-v2'
output.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((CODE/'outputs/metaworld_paper/amf-metaworld-mismatch-posterior_grad-seed0/input.yaml').read_text())
cfg.update(device='cpu', data_root=str(data), working_dir=str(output),
           require_successful_demos=True,
           exp_name=output.name, pre_epochs=1, rollout_every=1,
           finetune_iters=1, finetune_gradient_steps=2,
           eval_episodes_override=2, num_parallel_envs=1,
           eval_max_steps_override=16, sample_every=1,
           wandb_group='amf-metaworld-fix-smoke',
           protocol='CPU execution smoke ONLY; successful-demo dataset; 1 pretrain epoch, 2 finetune steps, shortened rollouts')
(output/'input.yaml').write_text(yaml.safe_dump(cfg))
os.chdir(CODE)
import main
started = time.time()
main.main(ConfigDict(cfg))
rows = [json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
assert rows[-1]['epoch'] == 1 and np.isfinite(rows[-1]['train_loss'])
for path, digest in old_hashes.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
report = dict(timestamp=time.time(), complete=True, smoke_only=True, device='cpu',
              elapsed_seconds=time.time()-started, datasets=audit,
              old_dataset_sha256=old_hashes, original_data_unchanged=True,
              latest_metrics=rows[-1])
(ROOT/'logs/amf_metaworld_fix/cpu_smoke_passed.json').write_text(json.dumps(report,indent=2))
print('PASS CPU train/select/eval/checkpoint/W&B; old datasets unchanged',flush=True)
