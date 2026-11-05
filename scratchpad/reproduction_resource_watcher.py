"""Persist resource measurements and queue health; publish telemetry to W&B."""
import json
import os
from pathlib import Path
import subprocess
import time
import psutil
import wandb

root=Path('/mnt/data/thanhpv/ldva')
out=root/'logs/overnight'
out.mkdir(parents=True,exist_ok=True)
run=wandb.init(project='paper-reproduction',group='orchestration',name='overnight-resource-monitor',dir=str(out))
print('WANDB_RUN_URL='+run.url,flush=True)
psutil.cpu_percent(interval=None)
while True:
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory','--format=csv,noheader,nounits'],text=True).strip().split(',')
    used,total,util,bandwidth=[float(x.strip()) for x in gpu]
    row=dict(timestamp=time.time(),gpu_memory_mib=used,gpu_memory_pct=100*used/total,
             gpu_util_pct=util,gpu_bandwidth_pct=bandwidth,cpu_util_pct=psutil.cpu_percent(),
             ram_used_gb=psutil.virtual_memory().used/2**30)
    with (out/'resources.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    run.log(row)
    statuses=[]
    for path in sorted((root/'external/amf/outputs/metaworld_paper').glob('status_worker*.json')):
        try:statuses.append(json.loads(path.read_text()))
        except (OSError,json.JSONDecodeError):pass
    (out/'status.json').write_text(json.dumps(dict(resources=row,amf_workers=statuses),indent=2))
    # Lightweight on-disk heartbeat, inspected by the supervising agent.
    failures=[s for s in statuses if s.get('exit_code',0)!=0]
    if failures:
        (out/'FIX_REQUIRED.json').write_text(json.dumps(failures,indent=2))
        print('FIX_REQUIRED '+json.dumps([s.get('current') for s in failures]),flush=True)
    time.sleep(15)
