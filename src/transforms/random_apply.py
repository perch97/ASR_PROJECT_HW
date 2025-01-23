import random
from typing import Callable

from torch import Tensor, nn

class RandomApply(nn.Module):
    def __init__(self, p: float, augmentation: Callable):
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")
        self.probability = p
        self.augmentation_function = augmentation

    def __call__(self, data: Tensor) -> Tensor:
        if random.uniform(0, 1) < self.probability:
            return self.augmentation_function(data)
        return data