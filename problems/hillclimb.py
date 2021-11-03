from numbers import Number

import numpy as np

from problems.individual import Individual, Encoding


class HillClimbIndividual(Individual):


    function = None

    def __init__(self, point=None):
        super().__init__()
        self.point = point

    def eval(self) -> Number:
        val = HillClimbIndividual.function(*self.point)
        return val ** 2

    def get_genome(self) -> np.ndarray:
        return self.point

    def get_encoding(self) -> "Encoding":
        return HillClimbEncoding(self.point.size)

    @staticmethod
    def from_genome(genome: np.ndarray, encoding: "Encoding") -> "Individual":
        return HillClimbIndividual(genome)

    @staticmethod
    def initialize_population(pop_size, n_params, bounds, function):
        HillClimbIndividual.function = function
        return [
            HillClimbIndividual(p)
            for p in np.random.uniform(-bounds, bounds, (pop_size, n_params))
        ]

    @staticmethod
    def get_mutation_weights(encoding: "Encoding") -> np.ndarray:
        return np.ones(encoding.length)

class HillClimbEncoding(Encoding):
    pass
