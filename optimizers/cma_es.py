from functools import partial

import math
import numpy as np
from numpy.random import default_rng

from optimizers.optimizer import Optimizer
from problems.individual import Individual


class CmaEs(Optimizer):
    def __init__(self, max_steps, individual, sigma):
        super().__init__(max_steps)
        self.individual = individual
        self.sigma = sigma

    @property
    def best_individual(self) -> "Individual":
        return self.individual

    def iterate(self, X, Y):
        # Initialize rng and individual
        rng = default_rng()
        individual = self.individual
        individual_generator = partial(
            individual.from_genome, encoding=individual.get_encoding()
        )
        N = individual.get_encoding().length
        xmean = individual.get_genome()
        sigma = self.sigma

        # Strategy parameters: selection
        _lambda = int(4 + np.floor(3 * np.log(N)))
        mu = math.floor(_lambda / 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))  # +1 for avoid log(0)
        weights = weights / np.sum(weights)
        mueff = np.sum(weights) ** 2 / np.sum(np.power(weights, 2))

        # Strategy parameters: adaptation
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        cs = (mueff + 2) / (N + mueff + 5)
        c1 = 2 / ((N + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs

        # Initialize dynamic strategy and constants
        pc = np.zeros(N)
        ps = np.zeros(N)
        B = np.eye(N)
        D = np.eye(N)
        C = np.eye(N)
        invsqrtC = B * np.reciprocal(np.diag(D)) * B.T
        eigenval = 0
        chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        for counteval in range(1, self.max_steps + 1):
            # Sample the distribution (defined by mean and covariance) to generate the population
            arx = xmean + sigma * rng.multivariate_normal(np.zeros(N), C, _lambda)

            # Evaluate
            pop = [individual_generator(g) for g in arx]
            for p in pop:
                p.loss = p.eval(X, Y)
            sort_indexes = np.argsort([i.loss for i in pop])

            # Store and yield the best individual
            self.individual = pop[sort_indexes[0]]
            print(self.individual.loss)
            yield self.individual

            # Update the mean
            xold = xmean
            xmean = weights @ arx[sort_indexes][:mu]

            # Update the evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / _lambda)) / chiN < 1.4 + 2 / (N + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma

            # Adapt the covariance matrix
            artmp = (1 / sigma) * (arx[sort_indexes][:mu] - xold).T
            C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp @ np.diag(
                weights) @ artmp.T

            # Adapt the step size
            sigma = sigma * np.exp((cs/damps)*(np.linalg.norm(ps)/chiN -1))

            # Update B and D from C (performed not on every step to achieve O(n^2))
            if counteval - eigenval > _lambda / (c1+cmu) / N / 10:
                eigenval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                B, D = np.linalg.eig(C)
                D = np.sqrt(np.diag(D))
                invsqrtC = B * np.reciprocal(np.diag(D)) * B.T
