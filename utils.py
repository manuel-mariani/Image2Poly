from enum import Enum

import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from optimizers.optimizer import Optimizer as Opt


class Problem(Enum):
    HILLCLIMB = 1
    DELAUNAY = 2


class Optimizer(Enum):
    GA = 1
    SA = 2
    CMA = 3


def load_image(path, downscaling, edge_threshold):
    with Image.open(path) as img:
        img = img.resize((img.size[0] // downscaling, img.size[1] // downscaling))
        edges = (
            img.convert("L")
                .filter(ImageFilter.GaussianBlur)
                .filter(ImageFilter.FIND_EDGES)
        )
        edges = (
            edges.filter(ImageFilter.GaussianBlur)
                .point(lambda p: 0 if p > edge_threshold else 255)
                .filter(ImageFilter.GaussianBlur)
        )
        # edges = invert(edges).point(lambda p: 0 if p > IMAGE_EDGE_THRESHOLD else 255).filter(ImageFilter.GaussianBlur)
        size_diff = img.size[0] - edges.size[0], img.size[1] - edges.size[1]
        new_edges = Image.new("L", img.size[:2], color=0)
        new_edges.paste(edges, (size_diff[0] // 2, size_diff[1] // 2))
        # new_edges.show()
        return img, new_edges


def image_show_loop(optimizer: Opt, target_image: Image.Image):
    size = target_image.size[:2]
    im = plt.imshow(target_image)
    plt.axis("off")
    plt.gcf().tight_layout()
    plt.gcf().set_size_inches(size[0] / 100, size[1] / 100)
    toolbar = plt.get_current_fig_manager().toolbar
    for a in toolbar.actions():
        toolbar.removeAction(a)

    iterator = optimizer.iterate()

    def update(individual):
        if individual is not None:
            gi = individual.generate_image()
            im.set_array(gi)
            return [im]

    ani = FuncAnimation(
        plt.gcf(),
        update,
        frames=iterator,
        interval=1,
        blit=True,
        cache_frame_data=False,
    )
    plt.show()


def hillclimb_show_loop(optimizer: Opt, f, bound=15):
    b, g = bound, 0.125
    grid_x = np.arange(-b, b, g)
    grid_x = np.arange(-b, b, g)
    grid_y = np.arange(-b, b, g)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = f(grid_x, grid_y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    surf = ax.plot_surface(grid_x, grid_y, grid_z, linewidth=0, alpha=0.2)
    plt.ion()
    plt.pause(0.01)

    def xyz_from_pop(population):
        points = np.stack([ind.point for ind in population])
        x = points.T[0, :]
        y = points.T[1, :]
        z = np.array([f(x[j], y[j]) for j in range(len(x))])
        return x, y, z

    iterator = optimizer.iterate()
    i = next(iterator, None)
    x, y, z = xyz_from_pop(optimizer.population)
    pop_plt = ax.plot(x, y, z, label="toto", ms=5, color="r", marker="^", ls="")[0]

    while i is not None:
        x, y, z = xyz_from_pop(optimizer.population)
        pop_plt.set_data_3d(x, y, z)
        fig.canvas.draw()
        plt.pause(0.01)
        i = next(iterator, None)
