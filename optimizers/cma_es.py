from functools import partial
from typing import List

import math
import numpy as np
from numpy.random import default_rng

from optimizers.optimizer import Optimizer
from problems.individual import Individual


class CmaEs(Optimizer):
    def __init__(self, max_steps, individual, sigma, pop_size=None):
        super().__init__(max_steps)
        self.individual = individual
        self.sigma = sigma
        self.pop_size = pop_size

    @property
    def best_individual(self) -> "Individual":
        return self.individual

    def iterate(self) -> List[Individual]:
        # Initialize rng and individual
        rng = default_rng()
        individual = self.individual
        individual_generator = partial(
            individual.from_genome, encoding=individual.get_encoding()
        )

        # Initialize constants
        N = individual.get_encoding().length  # Search dimension
        m = individual.get_genome()  # Mean of distribution
        sigma = self.sigma  # Step size
        _lambda = (
            int(4 + np.floor(3 * np.log(N))) if self.pop_size is None else self.pop_size
        )  # Number of individuals
        mu = math.floor(_lambda / 2)  # Effective size of selection
        weights = np.log(mu + 0.5) - np.log(
            np.arange(1, mu + 1)
        )  # Recombination weights
        weights = weights / np.sum(weights)

        mu_w = np.sum(np.power(weights, 2))  # Variance effective selection mass
        cc = 4 / (N + 4)  # Backward time horizon for path pc
        cs = 4 / (N + 4)  # Backward time horizon for path ps
        cmu = mu_w / (N ** 2)  # Learning rate for rank-mu update
        c1 = 2 / (N ** 2)  # Learning rate for rank-1 update
        d_sigma = 0.9  # Dampening for sigma
        chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))  # E||N(0, I)||

        # Initialize dynamic variables
        C = np.eye(N)  # Covariance matrix
        ps = np.zeros(N)  # Evolution path (sigma cumulation)
        pc = np.zeros(N)  # Evolution path (covariance cumulation)

        for step in range(self.max_steps):
            y = rng.multivariate_normal(np.zeros(N), C, _lambda)
            x = m + sigma * y  # x_i = m_k + sigma_k * N(0, C_k)

            # Evaluate
            pop = [individual_generator(g) for g in x]
            for p in pop:
                p.loss = p.eval()
            sort_indexes = np.argsort([i.loss for i in pop])

            # Store and yield the best individual
            self.individual = pop[sort_indexes[0]]
            print(self.individual.loss)
            yield sorted(pop, key=lambda i: i.loss)

            # Move the mean (mutation)
            m_old = m  # m' = m
            x_s = x[sort_indexes][:mu]  # w_i:lambda
            x_ws = x_s * weights[:, np.newaxis]  # w_i * x_i:lambda
            m = np.sum(x_ws, 0)  # sum_i^mu wi_i * x_i:lambda
            m_d = (m - m_old) / sigma  # displacement of m

            # Cumulative step size adaptation (path length control)
            c_sqrt_inv = np.linalg.inv(np.sqrt(C))  # C^(-1/2)
            comp = lambda i: np.sqrt(1 - (1 - i) ** 2)  # Complement function
            ps = (1 - cs) * ps + comp(cs) * np.sqrt(mu_w) * c_sqrt_inv * m_d
            indicator = np.linalg.norm(ps) < (1.5 * np.sqrt(N))
            pc = (1 - cc) * pc + indicator * comp(cc) * np.sqrt(mu_w) * m_d

            # Update covariance matrix using rank1 and rank mu
            m_ds = (x_s - m) / sigma
            rk = weights @ m_ds @ m_ds.T
            rk = np.sum(rk, 0)
            rk1 = np.outer(pc, pc)  # Rank one matrix: pc * pc.T
            C = (1 - cc - cmu + cs) * C + c1 * pc @ pc.T + cmu * rk

            _debug_rank1 = np.linalg.matrix_rank(rk1)
            _debug_symmetry = np.allclose(C, C.T, rtol=1e-5, atol=1e-08)

            # Update step size
            sigma = sigma * np.exp((cs / d_sigma) * (np.linalg.norm(ps) / chiN - 1))
