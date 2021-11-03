from abc import ABC, abstractmethod
from typing import Generator

from problems.individual import Individual


class Optimizer(ABC):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.step = 0

    @property
    @abstractmethod
    def best_individual(self) -> "Individual":
        pass

    @abstractmethod
    def iterate(self, x, y):
        pass
