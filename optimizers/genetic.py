from typing import List

import numpy as np

from optimizers.individual import Individual


class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,
        initial_population: List[Individual],
        mutation_rate: float,
        mutation_strength: float,
        elitism: int,
        max_steps: int,
        tau_init: float = 5.0,
        tau_end: float = 0.2,
    ):
        self.mutation_strength = mutation_strength
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size
        self.elitism = elitism
        self.max_steps = max_steps
        self.current_step = 1
        self.tau_init = tau_init
        self.tau_end = tau_end
        self.population = initial_population
        self.individual_class = initial_population[0].__class__

    def iterate(self, x, y):
        for _ in range(self.max_steps):
            for individual in self.population:
                individual.loss = individual.eval(x, y)
            yield self.population
            self.population = self.create_new_generation(self.population)
            self.current_step += 1

    def run(self, x, y):
        for individual in self.population:
            individual.loss = individual.eval(x, y)
        self.population = self.create_new_generation(self.population)
        self.current_step += 1

    def create_new_generation(self, old_generation) -> List[Individual]:
        new_generation = []
        old_generation.sort(key=lambda i: i.loss)
        new_generation += old_generation[0 : self.elitism + 1]
        encoding = old_generation[0].get_encoding()
        mutation_weights = self.individual_class.get_mutation_weights(encoding)

        old_genomes = [i.get_genome() for i in old_generation]
        parent_pairs = [
            (old_genomes[p[0]], old_genomes[p[1]]) for p in self.select(old_generation)
        ]

        for i in range(self.elitism + 1, len(old_generation)):
            parents = parent_pairs[i]
            new_genome = self.crossover(parents)
            new_genome = self.mutate(new_genome, mutation_weights)
            new_generation.append(
                self.individual_class.from_genome(new_genome, encoding)
            )
        return new_generation

    def select(self, population) -> List[np.ndarray]:
        tau = (
            self.current_step * (self.tau_end - self.tau_init) / self.max_steps
            + self.tau_init
        )
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
        ind = np.random.randint(0, len(ng) + 1)
        ng[ind:] = g2[ind:]
        return ng

    def mutate(self, genome: np.ndarray, mutation_weights: np.ndarray) -> np.ndarray:
        genome = genome.copy()
        is_mut = np.random.random(genome.size) <= self.mutation_rate
        len_mut = np.count_nonzero(is_mut)
        mutation = np.random.normal(0, self.mutation_strength, len_mut)
        genome[is_mut] += mutation_weights[is_mut] * mutation
        return genome
