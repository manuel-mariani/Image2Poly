from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np

from optimizers.optimizer import Optimizer
from problems.individual import Individual


def evaluate(individual):
    return individual.eval()


@dataclass
class GeneticAlgorithm(Optimizer):
    pop_size: int
    max_steps: int
    population: List[Individual]
    mutation_rate: float = 0.05
    mutation_strength: float = 0.1
    elitism: int = 1
    tau_ini: float = 5.0
    tau_end: float = 0.5
    crossover_points: int = 2
    parallelism: int = 12

    def __post_init__(self):
        self.steps = 1
        self.tau_k = (self.tau_end - self.tau_ini) / self.max_steps

    def iterate(self):

        for _ in range(self.max_steps):
            # with Pool(self.parallelism) as p:
            #     losses = p.map(evaluate, self.population)
            #     p.close()
            #     p.join()
            #     for idx, _ in enumerate(self.population):
            #         self.population[idx].loss = losses[idx]

            for p in self.population:
                p.loss = p.eval()
            yield self.best_individual
            print(self.population[0].loss)
            self.population = self.create_new_generation(self.population)
            self.steps += 1

    @property
    def best_individual(self) -> "Individual":
        self.population.sort(key=lambda i: i.loss)
        return self.population[0]

    def create_new_generation(self, old_generation) -> List[Individual]:
        new_generation = []
        from_genome = old_generation[0].from_genome

        old_generation.sort(key=lambda i: i.loss)
        new_generation += old_generation[0 : self.elitism + 1]

        encoding = old_generation[0].get_encoding()
        mutation_weights = old_generation[0].get_mutation_weights(encoding)

        old_genomes = [i.get_genome() for i in old_generation]
        parent_pairs = self.select(old_generation)

        zero = np.zeros(old_genomes[0].size)
        mumentums = [
            i.mumentum if i.mumentum is not None else zero for i in old_generation
        ]

        for i in range(self.elitism + 1, len(old_generation)):
            p = parent_pairs[i]
            p_genomes = old_genomes[p[0]], old_genomes[p[1]]
            p_mumentums = mumentums[p[0]], mumentums[p[1]]

            cross_genome = self.crossover(p_genomes)
            mutated_genome = self.mutate(cross_genome, mutation_weights)
            cross_mumentum = self.crossover(p_mumentums)

            new_genome = mutated_genome + cross_mumentum
            new_individual = from_genome(new_genome, encoding)
            new_individual.mumentum = new_genome - cross_genome

            new_generation.append(new_individual)
        return new_generation

    def select(self, population) -> List[np.ndarray]:
        tau = self.steps * self.tau_k + self.tau_ini
        losses = -np.array(list(map(lambda i: i.loss, population)))
        probabilities = np.exp(losses / tau) / sum(np.exp(losses / tau))
        probabilities = np.nan_to_num(probabilities)

        parent_pairs = []
        while len(parent_pairs) < len(population):
            pair = np.random.choice(len(population), 2, p=probabilities)
            parent_pairs.append(pair)
        return parent_pairs

    def crossover(self, genomes) -> np.ndarray:
        g1, g2 = genomes
        ng = g1.copy()
        indexes = np.random.choice(
            np.arange(ng.size), size=self.crossover_points, replace=False
        )
        indexes = np.sort(indexes)

        offset = 0
        for k, _ in enumerate(indexes):
            offset += 1
            if offset % 2 == 1:
                continue
            a, b = indexes[k - 1], indexes[k]
            ng[a:b] = g2[a:b]
        return ng

    def mutate(self, genome: np.ndarray, mutation_weights: np.ndarray) -> np.ndarray:
        genome = genome.copy()
        tau = self.steps * self.tau_k + self.tau_ini
        is_mut = np.random.random(genome.size) <= self.mutation_rate
        len_mut = np.count_nonzero(is_mut)
        mutation = np.random.normal(0, self.mutation_strength, len_mut)
        genome[is_mut] += mutation_weights[is_mut] * mutation * tau
        return genome
