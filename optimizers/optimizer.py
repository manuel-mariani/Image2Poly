from abc import ABC, abstractmethod
from typing import Generator, List

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
