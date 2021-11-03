from dataclasses import dataclass

import numpy as np

from optimizers.optimizer import Optimizer
from problems.individual import Individual


@dataclass
class SimulatedAnnealing(Optimizer):
    def __init__(
        self, individual: Individual, max_steps, exploration_factor, tau_ini, tau_end
    ):
        super().__init__(max_steps)
        self.individual = individual
        self.exploration_factor = exploration_factor
        self.tau_ini = tau_ini
        self.tau_end = tau_end
        self.tau_k = (self.tau_end - self.tau_ini) / self.max_steps

    def iterate(self):
        self.individual.loss = self.individual.eval()
        for _ in range(self.max_steps):
            self.step += 1
            yield self.individual
            print(self.individual.loss)

            candidate = self.generate_candidate(self.individual)
            candidate.loss = candidate.eval()

            diff = candidate.loss - self.individual.loss
            tau = self.step * self.tau_k + self.tau_ini
            # m = np.exp(-diff / tau)  # Metropolis Acceptance
            m = diff * tau
            r = np.random.random_sample()

            if diff < 0 or r < m:
                self.individual = candidate

    def generate_candidate(self, original_individual: Individual):
        encoding = original_individual.get_encoding()
        mutation_weights = original_individual.get_mutation_weights(encoding)
        genome = original_individual.get_genome()

        genome = self.mutate(genome, mutation_weights)
        return self.individual.from_genome(genome, encoding)

    def mutate(self, genome: np.ndarray, mutation_weights: np.ndarray) -> np.ndarray:
        mutation = np.random.normal(0, self.exploration_factor, len(mutation_weights))
        genome += mutation_weights * mutation
        return genome

    @property
    def best_individual(self) -> "Individual":
        return self.individual
