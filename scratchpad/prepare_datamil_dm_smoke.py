"""Small explicitly non-scientific fixtures exercising the actual GPU VJP code."""
from pathlib import Path
import json
import os
import shutil
import sys
import time
import numpy as np
sys.path.insert(0,'/mnt/data/thanhpv/ldva/external/datamil')
from scripts.libero_rlds_to_mds import columns
from streaming import LocalDataset,MDSWriter

data=Path('/mnt/data/thanhpv/datamil_data')
out=Path('/mnt/data/thanhpv/ldva/logs/datamil_paper_h15/gpu_smoke/mds')
target=os.environ['DATAMIL_TARGET']
for name,source,count in [('libero90_horizon','libero90',1024),(target,target,32)]:
    destination=out/name
    if (destination/'index.json').exists(): continue
    if destination.exists(): destination.rename(destination.with_name(name+f'.failed_{time.time_ns()}'))
    ds=LocalDataset(local=str(data/'mds'/source))
    writer=None; previous=None
    for i in range(count):
        item=ds[i]
        cluster=i//15
        if cluster!=previous:
            if writer: writer.finish()
            writer=MDSWriter(columns=columns,out=str(destination/f'traj_{cluster}'))
            previous=cluster
        item['index']=cluster
        item['weights']=np.array([2.0 if name==target and i%2 else 1.0],dtype=np.float32)
        writer.write({k:item[k] for k in columns})
    writer.finish()
    shards=[]
    for path in sorted(destination.glob('traj_*/index.json')):
        for shard in json.loads(path.read_text())['shards']:
            shard['raw_data']['basename']=path.parent.name+'/'+shard['raw_data']['basename']
            shards.append(shard)
    (destination/'index.json').write_text(json.dumps(dict(shards=shards,version=2)))
    shutil.copy2(data/'mds/libero90/dataset_statistics.json',destination/'dataset_statistics.json')
print('GPU smoke fixtures ready (NOT paper experiment data)',flush=True)
