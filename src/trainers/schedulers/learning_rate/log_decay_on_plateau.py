#%%
import copy
import math
from typing import Callable, List, Optional, Union
from torch.optim.lr_scheduler import _LRScheduler


__all__ = [
    "LogarithmicResetLRScheduler"
]


class LogarithmicResetLRScheduler(_LRScheduler):
    def __init__(self, optimizer, factor=3e-3, patience=15, min_lr=1e-8, reset_factor=0.75, warmup_length: int = 5000, verbose=False, callbacks_on_reset: Optional[List[Callable]] = None):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.reset_factor = reset_factor
        self.initial_reset_factor = copy.deepcopy(reset_factor)
        self.verbose = verbose
        self.num_bad_epochs = 0
        self.best = None
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.start_lrs = copy.deepcopy(self.initial_lrs)
        self.decay_steps = 0
        self.len_warmup = max(1, warmup_length)
        self.rate_warmup = 100
        self.orig_len_warmup = copy.deepcopy(self.len_warmup)
        self.warmup_steps = 1
        self.callbacks = []
        if callbacks_on_reset is not None:
            self.callbacks = callbacks_on_reset
        super().__init__(optimizer)

    def step(self, metrics: Optional[Union[float, int]] = None, epoch: Optional[int] = None):
        if self._step_count != 0:
            current_lr = self.get_last_lr()[0]
        else:
            current_lr = self.initial_lrs[0]
        if metrics is None:
            metrics = math.inf 
        if self.best is None or metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Warmup has been shown to increase learning stability, not really necessary for shallow models
        if self.warmup_steps < self.len_warmup:
            factor = self.warmup_steps/self.len_warmup
            for param_group, lr in zip(self.optimizer.param_groups, self.start_lrs):
                # A lower rate makes it more linear, a higher rate makes it more exponential
                param_group["lr"] = lr * (self.rate_warmup**factor - 1)/(self.rate_warmup - 1)
            self.warmup_steps += 1
        elif self.num_bad_epochs > self.patience:
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
                    for callback in self.callbacks:
                        callback()
                    # Once the learning rate has reached the minimum, reset it.
                    new_lr = initial_lr * self.reset_factor
                    if new_lr > self.min_lr:
                        # The lr stays at minimum for one more epoch, since now we start the warmup process again.
                        param_group['lr'] = self.min_lr
                        # Reset warmup procedure, but make it shorter
                        self.warmup_steps = 0
                        self.len_warmup //= 2
                        # Set the goal for warmup
                        self.start_lrs[i] = new_lr
                        # Reduce the factor of the reset for the next iteration, lowers the peak.
                        self.reset_factor *= self.reset_factor
                    else:
                        param_group['lr'] = self.start_lrs[i] = self.min_lr
                        self.initial_reset_factor *= self.initial_reset_factor
                        self.reset_factor = self.initial_reset_factor
                        # Reset warmup length
                        self.len_warmup = self.orig_len_warmup // 2
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

    optim = torch.optim.SGD(DummyClass().parameters(), lr=5e-5)
    scheduler = LogarithmicResetLRScheduler(optim, warmup_length=2500, factor=1.5e-3, patience=10, min_lr=5e-8)
    vals = []
    for i in range(int(3e4)):
        scheduler.step(i)
        vals.append(optim.param_groups[0]["lr"])
    plt.plot(vals)
    plt.show()
# %%
