from dataclasses import dataclass
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

    def __post_init__(self):
        self.steps = 1
        self.tau_k = (self.tau_end - self.tau_ini) / self.max_steps

    def iterate(self) -> List[Individual]:
        for step in range(1, self.max_steps):
            # Evaluate population
            for p in self.population:
                p.loss = p.eval()
            yield self.population
            self._print_loss(step, self.population[0].loss)
            # Create new generation
            self.population = self.create_new_generation(self.population)
            self.steps += 1

    @property
    def best_individual(self) -> "Individual":
        self.population.sort(key=lambda i: i.loss)
        return self.population[0]

    def create_new_generation(self, old_generation) -> List[Individual]:
        # Initialize new generation
        new_generation = []
        from_genome = old_generation[0].from_genome
        encoding = old_generation[0].get_encoding()
        mutation_weights = old_generation[0].get_mutation_weights(encoding)

        # Save the best individuals (elitism)
        old_generation.sort(key=lambda i: i.loss)
        new_generation += old_generation[0 : self.elitism + 1]

        # Take the genomes from the current population and generate crossover pairs
        old_genomes = [i.get_genome() for i in old_generation]
        parent_pairs = self.select(old_generation)

        # Take the mumentums from the current population
        zero = np.zeros(old_genomes[0].size)
        mumentums = [
            i.mumentum if i.mumentum is not None else zero for i in old_generation
        ]

        # Create each new individual, using each crossover-pair
        for i in range(self.elitism + 1, len(old_generation)):
            p = parent_pairs[i]  # Crossover parent pair
            # Take parent genomes and mumentums
            p_genomes = old_genomes[p[0]], old_genomes[p[1]]
            p_mumentums = mumentums[p[0]], mumentums[p[1]]
            # Crossover the genomes and mumentums
            cross_genome = self.crossover(p_genomes)
            cross_mumentum = self.crossover(p_mumentums)
            # Mutate the crossover genome
            mutated_genome = self.mutate(cross_genome, mutation_weights)
            # Sum the mutated genome and the crossover-mumentum
            new_genome = mutated_genome + cross_mumentum

            # Finally, create the new individual storing its mumentum
            new_individual = from_genome(new_genome, encoding)
            new_individual.mumentum = new_genome - cross_genome
            new_generation.append(new_individual)
        return new_generation

    def select(self, population) -> List[np.ndarray]:
        # Tau = inverse selection pressure (lower -> select more fit, higher -> more uniform)
        tau = self.steps * self.tau_k + self.tau_ini
        # Softmax the complement loss to determine the selection probabilities (Boltzmann-like-selection)
        losses = 1 - np.array(list(map(lambda i: i.loss, population)))
        probabilities = np.exp(losses / tau) / sum(np.exp(losses / tau))
        probabilities = np.nan_to_num(probabilities)  # Avoid NaN

        # Create the parent pairs, using the probability distribution
        parent_pairs = []
        while len(parent_pairs) < len(population):
            pair = np.random.choice(len(population), 2, p=probabilities)
            parent_pairs.append(pair)
        return parent_pairs

    def crossover(self, genomes) -> np.ndarray:
        # Get the parent genomes and set the offspring genome = parent1 genome
        g1, g2 = genomes
        ng = g1.copy()

        # Select random crossover points
        indexes = np.random.choice(
            np.arange(ng.size), size=self.crossover_points, replace=False
        )
        indexes = np.sort(indexes)

        # For each odd crossover point, copy the parents 2 slice into offspring
        offset = 0
        for k, _ in enumerate(indexes):
            offset += 1
            if offset % 2 == 1:
                continue
            a, b = indexes[k - 1], indexes[k]
            ng[a:b] = g2[a:b]
        return ng

    def mutate(self, genome: np.ndarray, mutation_weights: np.ndarray) -> np.ndarray:
        # Copy the genome
        genome = genome.copy()
        # Use tau to scale the mutation strength
        tau = self.steps * self.tau_k + self.tau_ini
        tau = min(1.0, tau)
        # Determine which genes to mutate
        is_mut = np.random.random(genome.size) <= self.mutation_rate
        len_mut = np.count_nonzero(is_mut)
        # Add a random normal mutation to those genes
        mutation = np.random.normal(0, self.mutation_strength, len_mut)
        genome[is_mut] += mutation_weights[is_mut] * mutation * tau
        return genome
