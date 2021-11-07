import math
from abc import ABC, abstractmethod
from typing import List

from problems.individual import Individual


class Optimizer(ABC):
    """Abstract class representing an optimizer"""

    def __init__(self, max_steps):
        """
        Initialize the optimizer
        :param max_steps: Max number of steps
        """
        self.max_steps = max_steps
        self.step = 0

    @property
    @abstractmethod
    def best_individual(self) -> "Individual":
        """Return the best individual currently found"""
        pass

    @abstractmethod
    def iterate(self) -> List["Individual"]:
        """Run the optimization algorithm until termination, yielding the current population / individual"""
        pass

    def _print_loss(self, step, loss):
        s = int(math.log10(self.max_steps)) - int(math.log10(step))
        print(f"[{'0' * s}{step}/{self.max_steps}] Loss: {loss:.4}")
