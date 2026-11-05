"""GPU regression checks followed by a short real train/select/eval run."""
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/data/thanhpv/ldva/external/amf')
import numpy as np
import torch
import yaml
from ml_collections import ConfigDict
from trainer import AdaptiveMixture, Trainer
import main


class ConstantPolicy:
    device = torch.device('cuda')
    def __init__(self, value): self.value = value
    def predict_action(self, batch): return {'action': torch.tensor([self.value], device='cuda')}


assert torch.cuda.is_available()
mix = AdaptiveMixture(ConstantPolicy(2.), ConstantPolicy(6.), .25)
assert mix.predict_action({})['action'].item() == 5.
print('PASS: adaptive prior uses the paper weighted mixture', flush=True)
# Regression: the held-out selector must use its own differentiable model,
# never a rollout-only adaptive mixture.
class Dummy:
    cfg = ConfigDict(dict(finetune_prior=True, paper_adaptive_mixture=True))
    finetune = True
    model = object()
    ema_model = object()
    holdout_model = object()

dummy = Dummy()
assert Trainer.get_policy(dummy, task=0, ema=False, force_holdout=True) is dummy.holdout_model
print('PASS: task-selection model remains differentiable', flush=True)
cfg=ConfigDict(yaml.safe_load(Path('reproduction_metaworld_smoke.yaml').read_text()))
cfg.working_dir='outputs/metaworld_gpu_smoke_v2'
cfg.exp_name='amf-metaworld-gpu-smoke-v2'
main.main(cfg)
print('PASS: GPU train, active selection, rollout, local checkpoint, W&B',flush=True)
