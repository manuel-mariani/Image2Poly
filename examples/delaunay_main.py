import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageFilter import GaussianBlur
from rich import print

from examples.delaunay import DelaunayIndividual
from optimizers.annealing import SimulatedAnnealing
from optimizers.genetic import GeneticAlgorithm


def run_delaunay_ga(image_path):
    losses = []
    plt.ion()
    plt.pause(0.0001)
    fig = plt.figure()
    ax = fig.add_subplot()

    with Image.open(image_path) as img:
        scale = 4
        original = img
        img = img.resize(tuple(np.array(img.size) // scale))
        img = img.filter(GaussianBlur(radius=2))
        img.show()
        pop_size = 30
        pop = DelaunayIndividual.initialize_population(pop_size, 500, img.size[:2])
        ga = GeneticAlgorithm(
            pop_size=pop_size,
            population=pop,
            mutation_rate=0.1,
            mutation_strength=0.10,
            elitism=1,
            max_steps=100,
            tau_ini=5.0,
            tau_end=0.5,
            crossover_points=10,
        )
        size = img.size
        img_color = np.asarray(img)

        for pop in ga.iterate(size, img_color):
            best = min(pop, key=lambda i: i.loss)
            losses.append(best.loss)
            print(
                f"[magenta][{ga.steps}/{ga.max_steps}][/] [bold cyan]Loss:[/] {best.loss} "
            )
            ax.imshow(best.generate_image(size, img_color))
            fig.canvas.start_event_loop(0.001)
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(len(losses)), losses)

    plt.show()


# === ANNEALING ===


def run_delaunay_sa(image_path):
    plt.pause(0.0001)
    plt.axis("off")
    plt.tight_layout()

    with Image.open(image_path) as img:
        img = img.resize(tuple(np.array(img.size) // 2))
        individual = DelaunayIndividual.initialize_population(1, 500, img.size[:2])[0]
        max_steps = 1000
        sa = SimulatedAnnealing(
            max_steps=max_steps,
            max_restarts=1,
            individual=individual,
            exploration_factor=0.5,
            tau_ini=0.01,
            tau_end=0.00001,
        )
        size = img.size
        img_color = np.asarray(img)
        iterator = sa.iterate(size, img_color)

        im = plt.imshow(img_color)
        plt.gcf().set_size_inches(size[0] / 100, size[1] / 100)

        def update(i):
            img = next(iterator).generate_image(size, img_color)
            im.set_array(img)
            return [im]

        from matplotlib.animation import FuncAnimation

        ani = FuncAnimation(plt.gcf(), update, frames=max_steps, interval=1, blit=True)
        plt.show()
