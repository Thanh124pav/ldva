"""GPU regression: an all-holdout minibatch must not update the holdout model."""
from types import SimpleNamespace
import torch
from trainer import Trainer

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1, device='cuda')

    def compute_loss(self, batch):
        return ((self.layer(batch['obs']) - batch['action'])**2).mean()

trainer = Trainer.__new__(Trainer)
trainer.cfg = SimpleNamespace(finetune_l2=0, finetune_ewc=0, finetune_prior=False)
trainer.model = Policy()
trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _:1)
trainer.finetune, trainer.finetune_steps = True, 0
trainer.ema = SimpleNamespace(step=lambda _:None)
# No holdout model/optimizer exists: accessing them would fail this test.
batch = dict(obs=torch.randn(8,2,device='cuda'), action=torch.randn(8,1,device='cuda'),
             holdout=torch.ones(8,1,device='cuda'))
result = trainer.train_step(batch)
assert torch.isfinite(torch.tensor(result['train_loss']))
assert trainer.finetune_steps == 1
print('PASS: finite main-model update with no empty hold-in update')
