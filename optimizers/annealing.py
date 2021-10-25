from dataclasses import dataclass

import numpy as np

from optimizers.individual import Individual


@dataclass
class SimulatedAnnealing:
    max_steps: int
    max_restarts: int
    individual: Individual
    exploration_factor: float
    tau_ini: float = 1
    tau_end: float = 0.2

    def __post_init__(self):
        self.steps = 1
        self.restarts = 0
        self.tau_k = (self.tau_end - self.tau_ini) / self.max_steps

    def iterate(self, x, y):
        self.individual.loss = self.individual.eval(x, y)
        for _ in range(self.max_steps):
            yield self.individual

            candidate = self.generate_candidate(self.individual)
            candidate.loss = candidate.eval(x, y)

            diff = candidate.loss - self.individual.loss
            tau = self.steps * self.tau_k + self.tau_ini
            m = np.exp(-diff / tau)  # Metropolis Acceptance
            r = np.random.random_sample()
            if diff > 0:
                print("m", m, "r", r, "tau", tau, "diff/m", diff / m)
            if diff < 0 or r < m:
                # print(diff)
                # if diff < 0:
                # if np.random.random(1) < m:
                self.individual = candidate
            self.steps += 1

        self.restarts += 1

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
