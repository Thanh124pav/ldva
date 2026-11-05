"""CPU regressions: successful data, actual selector, alpha update and mixture."""
from pathlib import Path
import sys
import unittest
import tempfile
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'external/amf'))
import numpy as np
import torch
from ml_collections import ConfigDict
from scripts.collect_metaworld import successful_demo
from trainer import Trainer, AdaptiveMixture
from dataset import Dataset


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))
        self.device = torch.device('cpu')

    def predict_action(self, batch):
        value = self.value.expand_as(batch['action'])
        return dict(action=value, action_pred=value)

    def compute_loss(self, batch):
        return (self.value-batch['action']).square().mean()


class Tests(unittest.TestCase):
    def test_loader_rejects_failed_demo_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metaworld/datasets/cup_left/ph/low_dim_abs.hdf5'
            path.parent.mkdir(parents=True)
            with h5py.File(path, 'w') as f:
                demo = f.create_group('data/demo_0')
                demo['rewards'] = np.zeros(3)
            cfg = ConfigDict(dict(env_name='metaworld', horizon=1, n_action_steps=1,
                                  data_root=directory, require_successful_demos=True))
            with self.assertRaisesRegex(ValueError, 'Unsuccessful/nonfinite demo'):
                Dataset(cfg)

    def test_filter_failed_nonfinite_and_shape(self):
        demo = {k: np.zeros((3, 1)) for k in ('obs', 'actions', 'next_obs', 'rewards', 'dones')}
        self.assertFalse(successful_demo(demo))
        demo['rewards'][1] = 1
        self.assertTrue(successful_demo(demo))
        demo['actions'][0] = np.nan
        self.assertFalse(successful_demo(demo))
        demo['actions'] = np.zeros((2, 1))
        self.assertFalse(successful_demo(demo))

    def test_mixture_endpoints(self):
        batch = {'action': torch.zeros(2, 1, 4)}
        for alpha in (0., .25, 1.):
            actual = AdaptiveMixture(Constant(2), Constant(6), alpha).predict_action(batch)['action']
            torch.testing.assert_close(actual, torch.full_like(actual, alpha*2+(1-alpha)*6))

    def test_real_alpha_update_matches_autograd(self):
        # Equation: global-batch MSE + beta*sum(alpha), LR(alpha)=1 (paper).
        trainer = object.__new__(Trainer)
        trainer.cfg = ConfigDict(dict(finetune_ewc=0., finetune_l2=0., finetune_prior=True,
                                     finetune_prior_every=1, finetune_prior_lr=1.,
                                     finetune_prior_pessimism=.01, tasks=['a', 'b']))
        trainer.finetune = True
        trainer.finetune_steps = 0
        trainer.model, trainer.prior = Constant(.2), Constant(.8)
        initial = torch.tensor([.4, .6])
        trainer.prior.alpha = initial.clone()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.)
        class Scheduler:
            def step(self): pass
            def get_last_lr(self): return [0.]
        class EMA:
            def step(self, model): pass
        trainer.lr_scheduler, trainer.ema = Scheduler(), EMA()
        batch = dict(obs=torch.zeros(3,1,2), action=torch.tensor([[[.1]*4], [[.7]*4], [[.2]*4]]),
                     task_id=torch.tensor([[[0.]], [[1.]], [[0.]]]), holdout=torch.ones(3,1))
        a = initial.clone().requires_grad_()
        mix = a[batch['task_id'].long()]*.2+(1-a[batch['task_id'].long()])*.8
        loss = (mix-batch['action']).square().mean()+.01*a.sum()
        expected = (a-torch.autograd.grad(loss,a)[0]).clamp(0,1).detach()
        trainer.train_step(batch)
        torch.testing.assert_close(trainer.prior.alpha, expected)

    def test_actual_acquisition_on_orthogonal_tasks(self):
        # Equal task weights; observing the higher-variance task should win.
        trainer = object.__new__(Trainer)
        trainer.cfg = ConfigDict(dict(finetune_strategy='posterior_grad',
                                     finetune_demos_per_iter=1, finetune_repeat_demos=1,
                                     finetune_noise=.01, device='cpu', tasks=['a','b']))
        embeddings = [torch.tensor([[3.,0.]]), torch.tensor([[0.,1.]])]
        trainer.get_embeddings = lambda epoch: (embeddings, torch.eye(2))
        self.assertEqual(trainer.select_demos(0), {'a':1,'b':0})
        trainer.get_embeddings = lambda epoch: (embeddings[::-1], torch.eye(2))
        self.assertEqual(trainer.select_demos(0), {'a':0,'b':1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
