import math
import warnings

import numpy as np
import torch.optim.lr_scheduler as scheduler


class SinusoidalExponentialLR(scheduler.LRScheduler):
    """Sets the learning rate to match a sinusoidal path from start_lr
    to end_lr with the given multiplicative factor.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_lr (float): The learning rate we start with
        end_lr (float): The learning rate we end with
        epochs (int): The number of overall epochs
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
    """

    def __init__(
        self,
        optimizer,
        start_lr: int,
        end_lr: int,
        epochs: int,
        total_steps: int,
        gamma: float,
        jump_factor: float = None,
        mul_factor=1.8,
        last_epoch=-1,
        verbose=False,
    ):
        self.gamma = gamma
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.total_steps = total_steps
        self.mul_factor = mul_factor
        self.jump_factor = math.sqrt(epochs) if jump_factor is None else jump_factor
        self.prev_exps = None

        super().__init__(optimizer, last_epoch, verbose)

    def update_lrs(self):
        self.prev_exps = [self.gamma * lr for lr in self.prev_exps]

        x = self.last_epoch / self.total_steps  # [0,1]
        x *= self.jump_factor  # [0, jump_factor]
        p = math.cos(x * math.pi)  # [-1,1]
        diff_ratio = max(0, self.mul_factor - 1)  # [0, mul_factor-1]
        p *= diff_ratio  # [-diff_ratio, diff_ratio]

        return list((1 + p) * np.array(self.prev_exps))

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        groups = self.optimizer.param_groups

        if self.last_epoch == 0:
            self.prev_exps = [group["lr"] for group in groups]
            return self.prev_exps
        return self.update_lrs()
