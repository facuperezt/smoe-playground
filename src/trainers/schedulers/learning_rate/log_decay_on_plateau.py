#%%
import copy
import math
from typing import Optional, Union
from torch.optim.lr_scheduler import _LRScheduler


__all__ = [
    "LogarithmicResetLRScheduler"
]


class LogarithmicResetLRScheduler(_LRScheduler):
    def __init__(self, optimizer, factor=7e-4, patience=1, min_lr=1e-7, reset_factor=0.9, verbose=False):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.reset_factor = reset_factor
        self.verbose = verbose
        self.num_bad_epochs = 0
        self.best = None
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.start_lrs = copy.deepcopy(self.initial_lrs)
        self.decay_steps = 0
        super().__init__(optimizer)

    def step(self, metrics: Optional[Union[float, int]] = None, epoch: Optional[int] = None):
        if self._step_count != 0:
            current_lr = self.get_last_lr()[0]
        else:
            current_lr = self.initial_lrs[0]
        if metrics is None:
            metrics = torch.inf 
        if self.best is None or metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            if current_lr > self.min_lr:
                self.decay_steps += 1
                new_lr = self.calculate_lr()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Reducing learning rate to {new_lr:.6f}')
            else:
                self.decay_steps = 0
                for i, (param_group, initial_lr) in enumerate(zip(self.optimizer.param_groups, self.initial_lrs)):
                    new_lr = initial_lr * self.reset_factor
                    param_group['lr'] = self.start_lrs[i] = new_lr
                    self.reset_factor *= 0.9
                if self.verbose:
                    print(f'Resetting learning rate to {initial_lr * self.reset_factor:.6f}')
            self.num_bad_epochs = 0

        super().step()

    def calculate_lr(self):
        for start_lr in self.start_lrs:
            return max(2*start_lr - (start_lr * math.exp(self.factor * self.decay_steps)), self.min_lr)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    class DummyClass(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(torch.tensor([5.], requires_grad=True))

        def forward(self, x):
            return x * self.weights

    optim = torch.optim.SGD(DummyClass().parameters(), lr=1e-6)
    scheduler = LogarithmicResetLRScheduler(optim)
    vals = []
    for i in range(10000):
        scheduler.step(i)
        vals.append(optim.param_groups[0]["lr"])
    plt.plot(vals)
    plt.show()
# %%
