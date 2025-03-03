"""
Learning rate schedulers specialized for Vortex Neural Networks.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class VortexCyclicalLR(_LRScheduler):
    """
    Learning rate scheduler with cycles based on vortex mathematics patterns.
    
    Args:
        optimizer: The optimizer to update
        base_lr: Minimum learning rate
        max_lr: Maximum learning rate
        step_size_up: Number of training iterations in the increasing half of a cycle
        cycle_pattern: List of numbers from vortex mathematics to determine the cycle pattern
    """
    def __init__(self, optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=2000, cycle_pattern=[1,2,4,8,7,5]):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.cycle_pattern = cycle_pattern
        super(VortexCyclicalLR, self).__init__(optimizer)

    def get_lr(self):
        """
        Return learning rates based on vortex cycle pattern.
        """
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size_up))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)

        # Map x [0,1] to an index in the cycle pattern
        pattern_idx = min(int(x * len(self.cycle_pattern)), len(self.cycle_pattern) - 1)

        # Use the cycle pattern value to determine where in the lr range we are
        cycle_factor = self.cycle_pattern[pattern_idx] / max(self.cycle_pattern)

        # Calculate lr
        lr = self.base_lr + (self.max_lr - self.base_lr) * cycle_factor

        return [lr for _ in self.optimizer.param_groups]
