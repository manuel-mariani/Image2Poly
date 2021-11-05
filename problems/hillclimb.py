from numbers import Number

import numpy as np

from problems.individual import Individual, Encoding


class HillClimbIndividual(Individual):
    function = None  # Function to evaluate

    def __init__(self, point: np.ndarray):
        super().__init__()
        self.point = point  # Point in R^n

    def eval(self) -> Number:
        """Evaluate the function at this particular point, returning the squared error wrt 0"""
        val = HillClimbIndividual.function(*self.point)
        return val ** 2

    def get_genome(self) -> np.ndarray:
        return self.point  # Since the point is already in R^n we just return it

    def get_encoding(self) -> "Encoding":
        return HillClimbEncoding(self.point.size)

    @staticmethod
    def from_genome(genome: np.ndarray, encoding: "Encoding") -> "Individual":
        return HillClimbIndividual(genome)

    @staticmethod
    def initialize_population(pop_size, n_params, bounds, function):
        """
        Randomly initialize a list of solutions.
        :param pop_size: Number of individuals
        :param n_params: Number of parameters in the function
        :param bounds: Bounds of the initial search space
        :param function: Function to optimize
        :return:
        """
        HillClimbIndividual.function = function
        return [
            HillClimbIndividual(p)
            for p in np.random.uniform(-bounds, bounds, (pop_size, n_params))
        ]

    @staticmethod
    def get_mutation_weights(encoding: "Encoding") -> np.ndarray:
        return np.ones(encoding.length)


class HillClimbEncoding(Encoding):
    pass  # This encoding does not need additional parameters since we are already in R^n
