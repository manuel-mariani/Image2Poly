import numpy as np

from optimizers.annealing import SimulatedAnnealing
from optimizers.cma_es import CmaEs
from optimizers.genetic import GeneticAlgorithm
from optimizers.pso import ParticleSwarmOptimizer
from problems.delaunay import DelaunayIndividual
from problems.hillclimb import HillClimbIndividual
from utils import load_image, image_show_loop, Problem, Optimizer, hillclimb_show_loop

# ------------------------------------------- #
#  CONFIG                                     #
# ------------------------------------------- #

OPTIMIZER = Optimizer.CMA
OPTIMIZATION_PROBLEM = Problem.DELAUNAY

# - Shared
IMAGE_PATH = "assets/monnalisa.jpg"
IMAGE_DOWNSCALING = 4
POP_SIZE = 50
MAX_STEPS = 500

# - Delaunay
N_POINTS = 300

# - Simulated annealing
EXPLORATION_FACTOR = 0.01
TAU_INI = 100
TAU_END = 0.001

# - Genetic Algorithm
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.1
ELITISM = 5
CROSSOVER_POINTS = 1

# - PSO
INERTIA = 0.5
PHI_COGNITIVE = 0.25
PHI_SOCIAL = 0.15
VELOCITY_STRENGTH = 1

# - CMA ES
SIGMA = 5

# - Hillclimbing function
N_PARAMS = 2
BOUNDS = 10


def fun(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return (np.sin(r) + 2) * r


# ------------------------------------------- #
#  MAIN                                       #
# ------------------------------------------- #


def main():
    # Load the image (for delaunay)
    img, edg = load_image(IMAGE_PATH, IMAGE_DOWNSCALING)

    # Initialize the problem
    problem = None
    if OPTIMIZATION_PROBLEM == Problem.DELAUNAY:
        problem = DelaunayIndividual.initialize_population(
            POP_SIZE,
            N_POINTS,
            img.size[:2],
            np.asarray(img),
            np.asarray(edg),
        )
    elif OPTIMIZATION_PROBLEM == Problem.HILLCLIMB:
        problem = HillClimbIndividual.initialize_population(
            POP_SIZE,
            N_PARAMS,
            BOUNDS,
            fun,
        )
    else:
        raise "Wrong optimization problem"

    # Initialize the optimizer
    optimizer = None
    if OPTIMIZER == Optimizer.GA:
        optimizer = GeneticAlgorithm(
            POP_SIZE,
            MAX_STEPS,
            problem,
            MUTATION_RATE,
            MUTATION_STRENGTH,
            ELITISM,
            TAU_INI,
            TAU_END,
            CROSSOVER_POINTS,
        )
    elif OPTIMIZER == Optimizer.SA:
        optimizer = SimulatedAnnealing(
            problem[0],
            MAX_STEPS,
            EXPLORATION_FACTOR,
            TAU_INI,
            TAU_END,
        )
    elif OPTIMIZER == Optimizer.CMA:
        optimizer = CmaEs(MAX_STEPS, problem[0], SIGMA, POP_SIZE)
    elif OPTIMIZER == Optimizer.PSO:
        optimizer = ParticleSwarmOptimizer(
            MAX_STEPS,
            problem,
            INERTIA,
            PHI_COGNITIVE,
            PHI_SOCIAL,
            VELOCITY_STRENGTH,
        )
    else:
        raise "Wrong optimizer"

    # Optimize (and show)
    if OPTIMIZATION_PROBLEM == Problem.DELAUNAY:
        image_show_loop(optimizer, img, IMAGE_DOWNSCALING)
    elif OPTIMIZATION_PROBLEM == Problem.HILLCLIMB and N_PARAMS == 2:
        hillclimb_show_loop(optimizer, fun, BOUNDS)


if __name__ == "__main__":
    main()
