"""Build the released horizon RLDS representation in an isolated data root."""
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
tfds.core.utils.gcs_utils._is_gcs_disabled = True

source = Path('/mnt/data/thanhpv/ldva/external/datamil/rlds_utils/rlds_dataset_builders/libero90_horizon/libero90_horizon.py')
sys.path.insert(0, str(source.parent.parent))
module = __import__('libero90_horizon.libero90_horizon', fromlist=['Libero90Horizon'])
assert module.HORIZON == 15 and module.MIN_H == 8
data=Path(os.environ['DATA_DIR'])
source_data=data/'tf/libero90/0.1.0'
destination=Path(os.environ['DATAMIL_PAPER_TF'])/'libero90_horizon/0.1.0'
destination.mkdir(parents=True,exist_ok=True)
if (destination/'horizon15.complete.json').exists():
    print('Reuse completed horizon15 RLDS',flush=True)
    sys.exit(0)
# Preserve already encoded images, actions, language and episode order exactly.
# Only change each frame's cluster ID. This avoids re-encoding 669k PNG pairs.
mds_index=json.loads((data/'mds/libero90/index.json').read_text())
lengths={}
for shard in mds_index['shards']:
    trajectory=int(shard['raw_data']['basename'].split('/')[0].split('_')[-1])
    lengths[trajectory]=lengths.get(trajectory,0)+shard['samples']
assert sorted(lengths)==list(range(4500))
offsets={}; cluster_count=0
for trajectory,length in sorted(lengths.items()):
    offsets[trajectory]=cluster_count
    cluster_count+=(length+14)//15
seen=set(); output_shards=[]
for path in sorted(source_data.glob('*tfrecord*')):
    new_path=destination/path.name.replace('libero90-', 'libero90_horizon-',1)
    marker=new_path.with_suffix(new_path.suffix+'.complete.json')
    if marker.exists():
        record=json.loads(marker.read_text()); seen.update(record['trajectory_ids'])
        output_shards.append(record); continue
    temporary=new_path.with_suffix(new_path.suffix+'.tmp')
    trajectories=[]; byte_count=0
    with tf.io.TFRecordWriter(str(temporary)) as writer:
        for serialized in tf.compat.v1.io.tf_record_iterator(str(path)):
            episode=tf.train.Example.FromString(serialized)
            index=episode.features.feature['steps/index'].int64_list.value
            trajectory=int(index[0]); assert len(set(index))==1
            assert len(index)==lengths[trajectory], (trajectory,len(index),lengths[trajectory])
            n,ids=module.get_chunked_idxs({'actions': __import__('numpy').empty((len(index),7))},15)
            assert n==(len(index)+14)//15
            old_image=episode.features.feature['steps/observation/image'].SerializeToString()
            index[:]=(ids+offsets[trajectory]).tolist()
            assert episode.features.feature['steps/observation/image'].SerializeToString()==old_image
            result=episode.SerializeToString(); writer.write(result); byte_count+=len(result)
            trajectories.append(trajectory)
    temporary.replace(new_path)
    record=dict(trajectory_ids=trajectories,serialized_bytes=byte_count,
                source=str(path),source_size=path.stat().st_size)
    marker.write_text(json.dumps(record))
    seen.update(trajectories);output_shards.append(record)
    print('REINDEX',len(seen),'/4500 episodes; shard',new_path.name,flush=True)
assert seen==set(range(4500))
for name in ('features.json',): shutil.copy2(source_data/name,destination/name)
info=json.loads((source_data/'dataset_info.json').read_text())
info['name']='libero90_horizon'
info['splits'][0]['numBytes']=str(sum(x['serialized_bytes'] for x in output_shards))
(destination/'dataset_info.json').write_text(json.dumps(info,indent=2))
builder=tfds.builder_from_directory(str(destination))
assert sum(builder.info.splits['train'].shard_lengths)==4500
(destination/'horizon15.complete.json').write_text(json.dumps(dict(
    horizon=15,minimum_tail=8,clusters=cluster_count,episodes=4500,
    source=str(source_data),images_actions_language='unchanged serialized features'),indent=2))
print('Horizon15 RLDS complete:',destination,'clusters',cluster_count,flush=True)
