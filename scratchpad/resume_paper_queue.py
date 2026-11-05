"""Resume only unfinished paper work; detached owner, append-only logs, no old DataMIL."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/data/thanhpv/ldva')
OUT = ROOT / 'logs/paper_queue'
OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT / 'supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
PY = '/mnt/data/thanhpv/miniconda3/envs/amf/bin/python'
AUTH = str(ROOT / 'scratchpad/run_with_wandb.py')
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1',
           MPLCONFIGDIR='/mnt/data/thanhpv/tmp/matplotlib-amf',
           WANDB_PROJECT='paper-reproduction', XLA_PYTHON_CLIENT_PREALLOCATE='false')
jobs = [('datamil_h15_t30', ['bash', 'scratchpad/run_datamil_paper_h15.sh'], ROOT, {})]
for worker in range(4):
    if len(list((ROOT/'external/amf/outputs/metaworld_paper').glob('*/complete.json'))) == 60:
        break
    jobs.append((f'metaworld_worker{worker}',
                 [PY, '-u', AUTH, 'reproduce_nn_batch.py', '--worker', str(worker), '--workers', '4'],
                 ROOT / 'external/amf',
                 {'PYTHONPATH': str(ROOT / 'external/amf') + ':' + str(ROOT / 'external/metaworld-history')}))
if not (ROOT/'external/amf/outputs/amf-robomimic-v15-full-matched-seed0/complete.json').exists():
    jobs.append(('robomimic', ['bash', 'scratchpad/run_robomimic_verified.sh'], ROOT, {}))
if not (ROOT/'external/amf/outputs/metaworld_success_v1/amf-metaworld-mismatch-posterior_grad-seed0-cpu/complete.json').exists():
    jobs.append(('amf_clean_seed0', ['bash', 'scratchpad/run_amf_clean_seed0.sh'], ROOT, {}))
jobs.extend([
    ('resources', [PY, '-u', AUTH, 'scratchpad/reproduction_resource_watcher.py'], ROOT, {}),
    ('report20min', [PY, '-u', 'scratchpad/write_reproduction_report.py', '--watch'], ROOT, {}),
])
processes = {}
started = time.time()
for name, command, cwd, extra in jobs:
    log = (OUT / (name + '.log')).open('a', buffering=1)
    log.write('\nRESUME ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
    process = subprocess.Popen(command, cwd=cwd, env=dict(env, **extra),
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    processes[name] = (process, log)
recorded = set()
while True:
    status = dict(updated=time.time(), started=started, pid=os.getpid(), ppid=os.getppid(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  jobs={name: dict(pid=p.pid, exit_code=p.poll()) for name, (p, _) in processes.items()})
    temporary = OUT / 'status.tmp'
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(OUT / 'status.json')
    for name, (process, _) in processes.items():
        code = process.poll()
        if code is not None and name not in recorded:
            recorded.add(name)
            (OUT / (name + '.exit.json')).write_text(json.dumps(dict(timestamp=time.time(), exit_code=code)))
            if code:
                with (ROOT / 'logs/reproduction_errors.txt').open('a') as errors:
                    errors.write(f'\n{time.ctime()} paper queue {name} exit={code}; see {OUT / (name + ".log")}\n')
    time.sleep(15)
