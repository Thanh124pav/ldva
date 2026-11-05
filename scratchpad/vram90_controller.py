"""Run real DataMIL seeds in two bounded JAX pools, targeting ~90% NVML VRAM."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil

ROOT = Path('/mnt/data/thanhpv/ldva')
OUT = ROOT/'logs/vram_target_90'
OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT/'controller.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
PY = '/mnt/data/thanhpv/miniconda3/envs/datamil/bin/python'
AUTH = str(ROOT/'scratchpad/run_with_wandb.py')
BASE = ROOT/'logs/datamil_fresh/full/paper-reproduction/datamil-fresh-training'
OLD_PARENT, OLD_EVAL = 3298365, 3303034
TARGET = .90

def gpu():
    row = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    return tuple(float(value.strip()) for value in row)

def checkpoint(seed):
    paths = list(BASE.glob(f'seed{seed}_fresh_*'))
    if len(paths)>1:
        raise RuntimeError(f'Ambiguous seed {seed} checkpoints: {paths}')
    return paths[0] if paths else None

env = dict(os.environ, DATA_DIR='/mnt/data/thanhpv/datamil_data',
    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', USE_TORCH='0', USE_FLAX='1',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MUJOCO_GL='egl',
    LIBERO_CONFIG_PATH=str(ROOT/'scratchpad/libero_config'),
    PYTHONPATH=str(ROOT/'external/datamil')+':'+str(ROOT/'external/LIBERO'),
    MPLCONFIGDIR='/mnt/data/thanhpv/tmp/matplotlib-datamil')
task = 'study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy'
indices = '/mnt/data/thanhpv/datamil_data/exp/datamodels/libero90/book-caddy/selected_indices_topk0.1.npy'
pretrained = '/home/thanhpv/.cache/huggingface/hub/models--rail-berkeley--octo-small/snapshots/03d88976c54a58e10480d2043a8c762b35bc2611'
pending = [seed for seed in range(1,5) if not (checkpoint(seed) and (checkpoint(seed)/'10000').is_dir())]
running, completed, evaluated = {}, [], set()
eval_process = None
first_launch = None
pool_overhead_mib = 1024.

def launch(seed, fraction):
    command = [PY,'-u',AUTH,'scripts/cotraining.py', '--config',
        f'scripts/configs/reproduction_cotrain_config.py:{task},libero90,{indices}',
        f'--config.seed={seed}','--config.batch_size=128','--config.num_steps=10000',
        '--config.log_interval=100','--config.save_interval=1000',
        '--config.data_parallel_calls=1','--config.data_parallel_reads=1',
        '--config.pretrained_path='+pretrained,
        '--config.save_dir='+str(ROOT/'logs/datamil_fresh/full'),
        '--name',f'seed{seed}_fresh_libero90_book-caddy','--debug=false']
    existing = checkpoint(seed)
    if existing:
        command += ['--resume_dir',str(existing)]
    job_env = dict(env, XLA_PYTHON_CLIENT_PREALLOCATE='true',
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction), REPRO_VRAM_TARGET='0.90')
    log = (OUT/f'train_seed{seed}.log').open('a', buffering=1)
    process = subprocess.Popen(command,cwd=ROOT/'external/datamil',env=job_env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    running[seed] = dict(process=process, log=log, fraction=fraction, started=time.time())
    print(f'START seed={seed} pid={process.pid} JAX_pool_fraction={fraction:.4f}',flush=True)

while True:
    used,total,util = gpu()
    for seed, job in list(running.items()):
        code = job['process'].poll()
        if code is None:
            continue
        if code or not (checkpoint(seed)/'10000').is_dir():
            failure = dict(seed=seed,exit_code=code,log=str(OUT/f'train_seed{seed}.log'))
            (OUT/'FIX_REQUIRED.json').write_text(json.dumps(failure,indent=2))
            raise RuntimeError(f'Training failure: {failure}; sequential launcher remains paused to avoid duplicates')
        completed.append(seed)
        job['log'].close()
        del running[seed]
        print(f'COMPLETE train seed={seed}',flush=True)
        time.sleep(5)
        used,total,util = gpu()

    if pending and not running:
        seed = pending.pop(0)
        first_launch = dict(time=time.time(),baseline=used,total=total,fraction=.38)
        launch(seed,.38)
    elif pending and len(running)==1:
        # Wait for the first JAX pool to exist before sizing the second pool.
        first = next(iter(running.values()))
        warm = used >= total*first['fraction'] + 5000
        if warm and time.time()-first['started']>25:
            if first_launch:
                delta = used-first_launch['baseline']
                pool_overhead_mib = max(1024.,delta-first_launch['fraction']*total)
                first_launch = None
            fraction = (TARGET*total-used-pool_overhead_mib)/total
            if fraction>=.36:
                launch(pending.pop(0),min(.40,fraction))

    old_eval_running = psutil.pid_exists(OLD_EVAL) and psutil.Process(OLD_EVAL).status()!=psutil.STATUS_ZOMBIE
    if eval_process and eval_process.poll() is not None:
        if eval_process.returncode:
            (OUT/'EVAL_FIX_REQUIRED.json').write_text(json.dumps(dict(seed=eval_seed,exit_code=eval_process.returncode)))
            raise RuntimeError('Evaluation failed; inspect the per-seed log')
        evaluated.add(eval_seed)
        eval_process = None
    if not old_eval_running and eval_process is None:
        candidates = [seed for seed in range(1,5) if seed not in evaluated and checkpoint(seed) and
                      (checkpoint(seed)/'10000').is_dir() and seed not in running]
        if candidates:
            eval_seed = candidates[0]
            command = [PY,'-u',AUTH,'datamil/eval_libero_parallel.py','--task_name','book-caddy',
                '--ckpt',str(BASE),'--out_dir',str(ROOT/'logs/datamil_fresh/evaluation'),
                '--num_envs','5','--total_evals','50','--seeds',str(eval_seed),
                '--checkpoint_provenance','fresh training; ten supplied metagradient trials',
                '--run_group','datamil-fresh-evaluation']
            eval_log = (OUT/f'eval_seed{eval_seed}.log').open('a')
            eval_process = subprocess.Popen(command,cwd=ROOT/'external/datamil',
                env=dict(env,XLA_PYTHON_CLIENT_PREALLOCATE='false'),
                stdin=subprocess.DEVNULL,stdout=eval_log,stderr=subprocess.STDOUT,start_new_session=True)
    status = dict(updated=time.time(),pid=os.getpid(),target_vram_pct=90,
        measured_vram_pct=100*used/total,gpu_used_mib=used,gpu_total_mib=total,gpu_util_pct=util,
        includes_jax_pool_cache=True,pending=pending,completed=completed,evaluated=sorted(evaluated),
        active={seed:dict(pid=job['process'].pid,pool_fraction=job['fraction']) for seed,job in running.items()},
        sequential_launcher_paused=OLD_PARENT,evaluation_seed=eval_seed if eval_process else None)
    tmp=OUT/'status.tmp'; tmp.write_text(json.dumps(status,indent=2)); tmp.replace(OUT/'status.json')
    if not pending and not running and len(evaluated)==4 and not eval_process:
        # The old launcher now finds all training/evaluation outputs complete,
        # skips them, and exits normally, keeping its original supervisor green.
        if psutil.pid_exists(OLD_PARENT):
            os.kill(OLD_PARENT,signal.SIGCONT)
        print('All four additional seeds trained and evaluated; sequential launcher released',flush=True)
        break
    time.sleep(10)
