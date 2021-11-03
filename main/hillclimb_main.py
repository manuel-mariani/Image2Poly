import numpy as np

from problems.hillclimb import HillClimbIndividual
from optimizers.algorithm import GeneticAlgorithm
from matplotlib import pyplot as plt


def f(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return (np.sin(r) + 2) * r


def main_hill_climbing_ga():

    # install(show_locals=True)  # Better tracebacks
    proto_point = np.array([20.0, 30.0])
    proto = HillClimbIndividual(proto_point)
    pop_size = 100
    ga = GeneticAlgorithm(
        pop_size=pop_size,
        initial_population=HillClimbIndividual.initialize_population(pop_size, 2, 20),
        mutation_rate=0.2,
        mutation_strength=10,
        elitism=1,
        max_steps=500,
    )

    b, g = 15, 0.125
    grid_x = np.arange(-b, b, g)
    grid_y = np.arange(-b, b, g)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = f(grid_x, grid_y)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    surf = ax.plot_surface(grid_x, grid_y, grid_z, linewidth=0, alpha=0.2)
    plt.ion()
    plt.pause(0.01)

    def get_xyz_from_pop(population):
        points = [i.point for i in population]
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        z = np.array([f(x[i], y[i]) for i in range(len(x))])
        return x, y, z

    x, y, z = get_xyz_from_pop(ga.population)
    pop_plt = ax.plot(x, y, z, label="toto", ms=5, color="r", marker="^", ls="")[0]

    while ga.steps < ga.max_steps:
        x, y, z = get_xyz_from_pop(ga.population)

        pop_plt.set_data_3d(x, y, z)
        fig.canvas.draw()
        plt.pause(0.01)

        ga.run(f, 0)


if __name__ == "__main__":
    main_hill_climbing_ga()
