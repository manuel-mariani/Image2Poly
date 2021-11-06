from typing import List

from numpy.random import default_rng

from optimizers.optimizer import Optimizer
from problems.individual import Individual


class ParticleSwarmOptimizer(Optimizer):
    def __init__(
        self,
        max_steps,
        population,
        inertia,
        phi_cognitive,
        phi_social,
        velocity_strength,
    ):
        super().__init__(max_steps)
        self.population = population
        self.inertia = inertia
        self.phi_cognitive = phi_cognitive
        self.phi_social = phi_social
        self.velocity_strength = velocity_strength

    @property
    def best_individual(self) -> "Individual":
        self.population.sort(key=lambda i: i.loss)
        return self.population[0]

    def iterate(self) -> List["Individual"]:
        # Initialize the individuals best known position to current and evaluate
        for particle in self.population:
            particle.pos = particle.get_genome()
            particle.loss = particle.eval()
            particle.best_pos = particle.pos.copy()
            particle.best_loss = particle.loss

        # Initialize swarm best position
        swarm_best_pos = self.best_individual.get_genome()
        swarm_best_loss = self.best_individual.loss

        # Initialize velocities
        rng = default_rng()
        encoding = self.population[0].get_encoding()
        for particle in self.population:
            particle.velocity = particle.get_mutation_weights(encoding) * rng.uniform(
                -self.velocity_strength, self.velocity_strength, encoding.length
            )

        # Optimize loop
        for step in range(self.max_steps):
            print(swarm_best_loss)
            yield self.population

            for i in range(len(self.population)):
                particle = self.population[i]
                # Random mixing of cognitive and social best
                rp = rng.uniform(0, 1, encoding.length)
                rg = rng.uniform(0, 1, encoding.length)
                # Get new velocity
                xi = particle.pos
                particle.velocity = (
                    self.inertia * particle.velocity
                    + self.phi_cognitive * rp * (particle.best_pos - xi)
                    + self.phi_social * rg * (swarm_best_pos - xi)
                )
                # Update position
                particle.pos += particle.velocity

                # Update the particle (must re-generate a particle)
                new_particle = particle.from_genome(particle.pos, encoding)
                new_particle.pos = particle.pos
                new_particle.loss = new_particle.eval()
                new_particle.velocity = particle.velocity
                new_particle.best_pos = particle.best_pos
                new_particle.best_loss = particle.best_loss

                # Store best found position
                if new_particle.loss < new_particle.best_loss:
                    new_particle.best_pos = new_particle.pos
                    new_particle.best_loss = new_particle.loss

                    if new_particle.loss < swarm_best_loss:
                        swarm_best_pos = new_particle.pos
                        swarm_best_loss = new_particle.loss

                # Replace particle
                self.population[i] = new_particle
