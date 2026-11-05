"""Independent screen-owned supervisor: append-only logs, durable exit status."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/mnt/data/thanhpv/ldva')
out = root/'logs/restart_20261104'
out.mkdir(parents=True, exist_ok=True)
lock = (out/'supervisor.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           MUJOCO_GL='egl', MPLCONFIGDIR='/mnt/data/thanhpv/tmp/matplotlib-amf',
           XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
auth = str(root/'scratchpad/run_with_wandb.py')
acvcl = '/mnt/data/thanhpv/miniconda3/envs/acvcl/bin/python'
amf = '/mnt/data/thanhpv/miniconda3/envs/amf/bin/python'
jobs = [
    ('metaworld_and_existing_datamil', ['bash','scratchpad/run_reproductions.sh'], root),
    ('gp_notebook', [acvcl,'-u','reproduce_gp.py','--no-plots','--wandb-mode','offline','--output','outputs/gp_reproduction_v2'], root/'external/amf'),
    ('gp_online_mirror', [acvcl,'-u',auth,'scratchpad/watch_gp_wandb.py','external/amf/outputs/gp_reproduction_v2'], root),
    ('gp_paper100', ['bash','scratchpad/run_gp_paper.sh'], root),
    ('robomimic', ['bash','scratchpad/run_robomimic_verified.sh'], root),
    ('report', [amf,'-u','scratchpad/write_reproduction_report.py','--watch'], root),
]
processes = {}
def launch(name, command, cwd):
    log = (out/(name+'.log')).open('a', buffering=1)
    log.write('\nRESTART '+time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())+'\n')
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    processes[name] = (process, log)
for job in jobs:
    launch(*job)
while True:
    if 'fresh_datamil' not in processes and (out/'fresh_resume_ready.json').exists():
        launch('fresh_datamil',['bash','scratchpad/run_datamil_fresh.sh'],root)
    status = dict(updated=time.time(), supervisor_pid=os.getpid(), supervisor_ppid=os.getppid(),
        jobs={name:dict(pid=p.pid, exit_code=p.poll()) for name,(p,_) in processes.items()})
    temporary = out/'status.tmp'
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(out/'status.json')
    for name, (p, log) in processes.items():
        marker = out/(name+'.exit.json')
        if p.poll() is not None and not marker.exists():
            marker.write_text(json.dumps(dict(exit_code=p.returncode, timestamp=time.time())))
            if p.returncode:
                with (root/'logs/reproduction_errors.txt').open('a') as errors:
                    errors.write(f'\n{time.ctime()} restarted job {name} exit={p.returncode}; see {out/name}.log\n')
    time.sleep(15)
