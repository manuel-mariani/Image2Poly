from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number

import numpy as np


class Individual(ABC):
    """Abstract class representing a solution of an optimization problem"""

    def __init__(self):
        self.loss = 0
        self.mumentum = None

    @abstractmethod
    def eval(self) -> Number:
        """Evaluate the solution, returning a loss value (lower is better)"""
        pass

    @abstractmethod
    def get_genome(self) -> np.ndarray:
        """Return a flat array representing the solution in R^n"""
        pass

    @abstractmethod
    def get_encoding(self) -> "Encoding":
        """Return the encoding of the genome, used to convert "flat genome" -> individual"""
        pass

    @staticmethod
    @abstractmethod
    def from_genome(genome: np.ndarray, encoding: "Encoding") -> "Individual":
        """Create an individual using a genome and the provided encoding for conversion"""
        pass

    @staticmethod
    @abstractmethod
    def get_mutation_weights(encoding: "Encoding") -> np.ndarray:
        """
        Return the mutation weights array of the genome.
        They effect the mutation strength along that particular direction.
        """
        pass


@dataclass
class Encoding(ABC):
    """Abstract class representing the encoding of a genome. It always contains a length"""

    length: int

    def __post_init__(self):
        # Forcing abstraction
        if type(self).__name__ == "Encoding":
            raise TypeError("Cannot instantiate abstract class.")
