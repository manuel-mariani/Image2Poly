from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number

import numpy as np


class Individual(ABC):
    def __init__(self):
        self.loss = 0
        self.mumentum = None

    @abstractmethod
    def eval(self, x, y) -> Number:
        pass

    @abstractmethod
    def get_genome(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_encoding(self) -> "Encoding":
        pass

    @staticmethod
    @abstractmethod
    def from_genome(genome: np.ndarray, encoding: "Encoding") -> "Individual":
        pass

    @staticmethod
    @abstractmethod
    def get_mutation_weights(encoding: "Encoding") -> np.ndarray:
        pass


@dataclass
class Encoding(ABC):
    length: int

    def __post_init__(self):
        if type(self).__name__ == "Encoding":
            raise TypeError("Cannot instantiate abstract class.")
