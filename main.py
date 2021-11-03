import numpy as np

from optimizers.annealing import SimulatedAnnealing
from optimizers.cma_es import CmaEs
from optimizers.genetic import GeneticAlgorithm
from problems.delaunay import DelaunayIndividual
from problems.hillclimb import HillClimbIndividual
from utils import load_image, image_show_loop, Problem, Optimizer, hillclimb_show_loop

# ====================== CONFIG ======================

OPTIMIZATION_PROBLEM = Problem.HILLCLIMB
OPTIMIZER = Optimizer.CMA

# - Shared
IMAGE_PATH = "assets/monnalisa.jpg"
IMAGE_DOWNSCALING = 2
IMAGE_EDGE_THRESHOLD = 2
POP_SIZE = 100
MAX_STEPS = 1000

# - Delaunay
N_POINTS = 100
N_VERTICES = 3
N_COLORS = 10

# - Simulated annealing
EXPLORATION_FACTOR = 0.01
TAU_INI = 4
TAU_END = 0.01

# - Genetic Algorithm
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.1
ELITISM = 2
CROSSOVER_POINTS = 1
PARALLEL_THREADS = 12

# - CMA ES
SIGMA = 10

# - Hillclimbing function
N_PARAMS = 2
BOUNDS = 10


def fun(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return (np.sin(r) + 2) * r


def main():
    img, edg = load_image(IMAGE_PATH, IMAGE_DOWNSCALING, IMAGE_EDGE_THRESHOLD)

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
            PARALLEL_THREADS,
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
        optimizer = CmaEs(
            MAX_STEPS,
            problem[0],
            SIGMA,
        )
    else:
        raise "Wrong optimizer"

    if OPTIMIZATION_PROBLEM == Problem.DELAUNAY:
        image_show_loop(optimizer, img)
    elif OPTIMIZATION_PROBLEM == Problem.HILLCLIMB and N_PARAMS == 2:
        hillclimb_show_loop(optimizer, fun, BOUNDS)


if __name__ == "__main__":
    main()
