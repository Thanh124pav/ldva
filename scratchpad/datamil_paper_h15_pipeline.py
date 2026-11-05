"""Detached, resumable stage queue. Never reuses trajectory-level influences."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/data/thanhpv/ldva')
CODE = ROOT/'external/datamil'
OUT = ROOT/'logs/datamil_paper_h15'
DATA = Path('/mnt/data/thanhpv/datamil_data')
PAPER = DATA/'paper_h15_t30'
TASK = 'study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy'
PY = '/mnt/data/thanhpv/miniconda3/envs/datamil/bin/python'
AUTH = str(ROOT/'scratchpad/run_with_wandb.py')
PRETRAINED = '/home/thanhpv/.cache/huggingface/hub/models--rail-berkeley--octo-small/snapshots/03d88976c54a58e10480d2043a8c762b35bc2611'
OUT.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)
lock = (OUT/'pipeline.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (OUT/'FIX_REQUIRED.json').exists():
    (OUT/'FIX_REQUIRED.json').rename(OUT/f'previous_failure_{time.time_ns()}.json')
stats = list((DATA/'tf/libero90/0.1.0').glob('dataset_statistics_*.json'))
assert len(stats) == 1
env = dict(os.environ, DATA_DIR=str(DATA), DATAMIL_PAPER_TF=str(PAPER/'tf'),
    DATAMIL_PAPER_MDS=str(PAPER/'mds'), DATASET_STATISTICS=str(stats[0]),
    DATAMIL_TARGET=TASK, DATAMIL_CLUSTER_HORIZON='15', DATAMIL_CLUSTER_MIN_TAIL='8',
    DATAMIL_DM_PROGRESS='1', USE_TORCH='0', USE_FLAX='1',
    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MUJOCO_GL='egl',
    LIBERO_CONFIG_PATH=str(ROOT/'scratchpad/libero_config'),
    PYTHONPATH=str(CODE)+':'+str(ROOT/'external/LIBERO'),
    XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.80',
    MPLCONFIGDIR='/mnt/data/thanhpv/tmp/matplotlib-datamil-paper')
sys.path.insert(0, str(ROOT/'scratchpad'))
from run_with_wandb import configure_wandb
configure_wandb()
env.update({k:v for k,v in os.environ.items() if k.startswith('WANDB_')})
import wandb
run = wandb.init(project='paper-reproduction', entity='hust_edu_vn',
    id='datamil-paper-h15-t30-pipeline', resume='allow', group='datamil-paper-h15-t30',
    name='DataMIL horizon15 30-trial queue', dir=str(OUT),
    config=dict(horizon=15, minimum_tail=8, trials=30, policy_seeds=[0],
                policy_batch=128, policy_steps=10000, datamodel_batch=512,
                datamodel_microbatch=128, datamodel_steps=2100, metagradient_steps=100,
                shared_statistics=str(stats[0]), target_demos=5,
                resume_granularity='data stage / datamodel forward100steps and completed trial / policy1000steps'))

def atomic(path, data):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(data,f,indent=2); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)

def stage(name, command, cwd=CODE, extra=None):
    started=time.time()
    print('START', name, flush=True)
    jobenv=dict(env, **(extra or {}))
    logpath=OUT/(name+'.log')
    with logpath.open('a') as log:
        p=subprocess.Popen(command,cwd=cwd,env=jobenv,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            code=p.poll()
            status=dict(updated=time.time(),pipeline_pid=os.getpid(),stage=name,
                        child_pid=p.pid,elapsed_seconds=time.time()-started,
                        exit_code=code,log=str(logpath),horizon=15,trials=30,policy_seeds=[0])
            atomic(OUT/'status.json',status)
            run.log({'stage/elapsed_seconds':status['elapsed_seconds'], 'stage/name':name})
            if code is not None: break
            time.sleep(15)
    if code:
        atomic(OUT/'FIX_REQUIRED.json', status)
        with (ROOT/'logs/reproduction_errors.txt').open('a') as f:
            f.write(f'\nDataMIL paper H15 stage {name} exit {code}; {logpath}\n')
        raise RuntimeError(f'{name} failed, see {logpath}')
    atomic(OUT/(name+'.complete.json'), status)
    print('DONE',name,flush=True)

def convert(name, source):
    destination=PAPER/'mds'/name
    if (destination/'conversion.complete.json').exists(): return
    attempt=PAPER/f'mds_build_{name}_{time.time_ns()}'
    stage('mds_'+name,[PY,'-u','scripts/libero_rlds_to_mds.py','--data_path',str(source),
        '--out_path',str(attempt)],extra={'CUDA_VISIBLE_DEVICES':'',
        'DATAMIL_PAPER_WEIGHTED_PROXY':'1' if name==TASK else '0'})
    destination.parent.mkdir(parents=True,exist_ok=True)
    assert not destination.exists(), f'Unexpected incomplete destination: {destination}'
    shutil.move(str(attempt/name),str(destination))
    atomic(destination/'conversion.complete.json',dict(time=time.time(),source=str(source),horizon=15))

try:
    stage('unit_tests',[PY,'-u',str(ROOT/'scratchpad/test_datamil_h15.py')],cwd=ROOT,
          extra={'CUDA_VISIBLE_DEVICES':''})
    stage('build_rlds_h15',[PY,'-u',str(ROOT/'scratchpad/prepare_datamil_h15.py')],cwd=ROOT,
          extra={'CUDA_VISIBLE_DEVICES':''})
    convert('libero90_horizon',PAPER/'tf/libero90_horizon')
    convert(TASK,DATA/'tf'/TASK)
    stage('gpu_metagradient_smoke',['bash',str(ROOT/'scratchpad/run_datamil_h15_gpu_smoke.sh')],cwd=ROOT)
    dmroot=PAPER/'exp/datamodels'
    dm=dmroot/'libero90_horizon/book-caddy'
    for trial in range(30):
        if (dm/f'iter_{trial}/complete.json').exists(): continue
        stage(f'datamodel_trial{trial:02}',[PY,'-u',AUTH,
            'datamil/metagradients/do_libero_data_selection_parallel.py',
            '--config','scripts/configs/paper_h15_dm_config.py',
            '--config.pretrained_path='+PRETRAINED, f'--config.job_id={trial}',
            '--config.folder_name=book-caddy','--config.save_dir='+str(dmroot)])
    stage('aggregate_top10',[PY,'-u','datamil/utils/load_dm.py',
        '--config','scripts/configs/paper_h15_dm_config.py',
        '--config.pretrained_path='+PRETRAINED,'--config.folder_name=book-caddy',
        '--config.save_dir='+str(dmroot),'--config.topk=0.1'])
    policy=OUT/'policy'
    base=policy/'paper-reproduction/datamil-paper-h15-t30-policy'
    paths=list(base.glob('seed0_h15_t30_*'))
    assert len(paths)<=1
    if not paths or not (paths[0]/'10000').exists():
        command=[PY,'-u',AUTH,'scripts/cotraining.py','--config',
            f'scripts/configs/paper_h15_cotrain_config.py:{TASK},libero90_horizon,{dm}/selected_indices_topk0.1.npy',
            '--config.seed=0','--config.pretrained_path='+PRETRAINED,
            '--config.save_dir='+str(policy),'--name=seed0_h15_t30_book-caddy','--debug=false']
        if paths: command += ['--resume_dir',str(paths[0])]
        stage('policy_seed0',command)
    stage('evaluation_seed0',[PY,'-u',AUTH,'datamil/eval_libero_parallel.py',
        '--task_name','book-caddy','--ckpt',str(base),'--out_dir',str(OUT/'evaluation'),
        '--num_envs','5','--total_evals','50','--seeds','0',
        '--checkpoint_provenance','horizon15; 30 new metagradient trials; common statistics; weighted proxy',
        '--run_group','datamil-paper-h15-t30-evaluation'])
    atomic(OUT/'complete.json',dict(time=time.time(),complete=True))
finally:
    run.finish()
