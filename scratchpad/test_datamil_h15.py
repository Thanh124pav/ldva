"""CPU regressions for horizon assignment and exact frame-level selection."""
import copy
import importlib.util
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from datamil.data.dataset import apply_trajectory_transforms

source=Path('/mnt/data/thanhpv/ldva/external/datamil/rlds_utils/rlds_dataset_builders/libero90_horizon/libero90_horizon.py')
spec=importlib.util.spec_from_file_location('h15_test_builder',source)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert module.HORIZON==15 and module.MIN_H==8
for length in (1,7,15,16,29,30,31,45,46,150,153):
    n,ids=module.get_chunked_idxs({'actions':np.zeros((length,7))},15)
    assert ids.shape==(length,) and np.array_equal(np.unique(ids),np.arange(n))
    assert np.bincount(ids).max()<=15

class EagerDataset:
    def __init__(self, items): self.items=items
    def filter(self, fn):
        return EagerDataset([x for x in self.items if bool(fn(x))])
    def traj_map(self, fn, *args, **kwargs):
        return EagerDataset([fn(copy.deepcopy(x)) for x in self.items])

n=45
sample=dict(observation=dict(proprio=tf.zeros((n,8)),timestep=tf.range(n)),
    task=dict(language_instruction=tf.repeat('test',n)),
    action=tf.tile(tf.cast(tf.range(n)[:,None],tf.float32),[1,7]),
    index=tf.repeat(tf.range(3),15)[:,None])
result=apply_trajectory_transforms(EagerDataset([sample]),train=False,
    future_action_window_size=7,training_idxs=[1],filter_selected_frames=True,
    num_parallel_calls=1).items[0]
assert result['action'].shape==(15,8,7)
assert np.all(result['index'].numpy()==1)
np.testing.assert_array_equal(result['observation']['timestep'][:,0],np.arange(15,30))
# Last selected frame still predicts the original future, not a stitched or
# prematurely padded action from a filtered episode.
assert float(result['action'][-1,-1,0])==36
old=apply_trajectory_transforms(EagerDataset([sample]),train=False,
    training_idxs=[1],num_parallel_calls=1).items[0]
assert old['action'].shape[0]==45
print('PASS: horizon15 coverage, short tails, frame filtering, action context, legacy compatibility')

import tempfile
from collections import namedtuple
from datamil.metagradients.resume_state import save_state,restore_state
Opt=namedtuple('Opt',['count','moment'])
State=namedtuple('State',['params','opt_state'])
state=State(params=np.arange(6,dtype=np.float32).reshape(2,3),
            opt_state=Opt(np.array(100,dtype=np.int32),np.zeros((2,3),np.float32)))
with tempfile.TemporaryDirectory(prefix='datamil-h15-test-') as directory:
    path=Path(directory)/'state.pkl'
    save_state(path,state)
    restored=restore_state(path,state)
    np.testing.assert_array_equal(restored.params,state.params)
    assert int(restored.opt_state.count)==100
print('PASS: atomic replay checkpoint roundtrip (arrays, shapes, dtype, optimizer count)')

import ast
selection_source=Path('/mnt/data/thanhpv/ldva/external/datamil/datamil/metagradients/do_libero_data_selection_parallel.py')
tree=ast.parse(selection_source.read_text())
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='grad_from_store')
namespace={'np':np}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(selection_source),'exec'),namespace)
grad=namespace['grad_from_store']({'segment':{0:np.array([1.,2.,3.])}},
                                {0:np.array([0,0,1])})
np.testing.assert_array_equal(grad,np.array([3.,3.]))
print('PASS: duplicate horizon-index metagradient contributions summed')

from datamil.metagradients.core.dataloading import RoboREPLAYBatch
batch=dict(index=np.arange(32,dtype=np.int32),action=np.zeros((32,8,7),np.float32))
replay=RoboREPLAYBatch(batch,bs=32,minibs=128,sharding=None)
tiles=list(replay.get_minibatches('val'))
assert len(tiles)==1 and len(tiles[0][0])==32
print('PASS: partial validation batch smaller than replay tile')

from datamil.data.dataset import restructure
raw=dict(observation=dict(state=tf.zeros((3,8))),action=tf.zeros((3,7)),
         weights=tf.ones((3,1)))
def upweight(trajectory):
    trajectory['weights']=tf.constant([1.,2.,1.])
    return trajectory
processed=restructure(raw,name='test',standardize_fn=upweight,state_obs_keys=['state'])
np.testing.assert_array_equal(processed['weights'],[1.,2.,1.])
print('PASS: computed proxy weights survive dataset restructuring')

os.environ.setdefault('DATAMIL_PAPER_MDS','/mnt/data/thanhpv/datamil_data/paper_h15_t30/mds')
os.environ.setdefault('DATAMIL_PAPER_TF','/mnt/data/thanhpv/datamil_data/paper_h15_t30/tf')
os.environ.setdefault('DATAMIL_TARGET','study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy')
os.environ.setdefault('DATASET_STATISTICS',str(next(Path('/mnt/data/thanhpv/datamil_data/tf/libero90/0.1.0').glob('dataset_statistics_*.json'))))
from scripts.configs.paper_h15_dm_config import get_config as dm_config
from scripts.configs.paper_h15_cotrain_config import get_config as policy_config
cfg=dm_config()
assert cfg.num_steps==2100 and cfg.bob_steps==100 and cfg.optimizer.learning_rate.decay_steps==2100
assert cfg.batch_size==512 and cfg.paper_weighted_proxy
cfg=policy_config(os.environ['DATAMIL_TARGET']+',libero90_horizon,none')
assert cfg.num_steps==10000 and cfg.batch_size==128 and cfg.sample_weights==[1.,1.]
assert cfg.traj_transform_kwargs.filter_selected_frames
assert len({x['dataset_statistics'] for x in cfg.dataset_kwargs_list})==1
print('PASS: resolved paper datamodel/policy configuration and common statistics')
