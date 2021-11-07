from enum import Enum

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from optimizers.optimizer import Optimizer as Opt


# ------------------------------------------- #
#  ENUMS                                      #
# ------------------------------------------- #


class Problem(Enum):
    HILLCLIMB = 1
    DELAUNAY = 2


class Optimizer(Enum):
    GA = 1
    SA = 2
    CMA = 3
    PSO = 4


# ------------------------------------------- #
#  FUNCTIONS                                  #
# ------------------------------------------- #


def load_image(path, downscaling):
    """
    Loads an image from file, performing a downscale, blur and edge detection
    :param path: The image's path
    :param downscaling: Downscaling factor
    :return: Tuple containing the color image and the edges
    """
    with Image.open(path) as img:
        img = img.resize((img.size[0] // downscaling, img.size[1] // downscaling))
        size = img.size

        # Find the edges
        edges = (
            img.convert("L")
            .filter(ImageFilter.MedianFilter)
            .filter(ImageFilter.CONTOUR)
            .filter(ImageFilter.GaussianBlur)
        )

        # Blur the image if size is relatively large
        if sum(img.size) > 500:
            img = img.filter(ImageFilter.GaussianBlur(4))

        # Normalize the edge map (equalize)
        edges = ImageOps.equalize(edges).filter(ImageFilter.GaussianBlur)

        # Resize the edge-map to the original size
        padding = lambda _img: (
            (img.size[0] - _img.size[0]) // 2,
            (img.size[1] - _img.size[1]) // 2,
        )
        new_edges = Image.new("L", size[:2], color=0)
        new_edges.paste(edges, padding(edges))

        # new_edges.show()
        # img.show()
        return img, new_edges


def image_show_loop(optimizer: Opt, target_image: Image.Image, scaling: int):
    """
    Runs and displays the evolution of an optimizer for delaunay
    :param optimizer: The initialized optimizer
    :param target_image: The input image of the optimizer
    :param scaling: Multiplicative scaling for the final image
    """

    # Initialize plot
    size = target_image.size[:2]
    im = plt.imshow(target_image)
    plt.axis("off")
    plt.gcf().tight_layout()
    plt.gcf().set_size_inches(size[0] / 100, size[1] / 100)
    toolbar = plt.get_current_fig_manager().toolbar
    for a in toolbar.actions():
        toolbar.removeAction(a)

    # Call the optimizer iterator
    iterator = optimizer.iterate()

    # Define a function to update the image plot
    def update(population):
        if population is not None:
            individual = population[0]
            gi = individual.generate_image()
            im.set_array(gi)
        return [im]

    # Run the animation using the data from the optimizer and the update function
    ani = FuncAnimation(
        plt.gcf(),
        update,
        frames=iterator,
        interval=1000 // 24,
        blit=True,
        init_func=lambda: [im],
        cache_frame_data=False,
    )
    plt.show()

    # Get the best image, producing a 8 * scaling scaled image [SSAAx8 super-sampling-anti-aliasing]
    result = optimizer.best_individual.generate_image(8 * scaling)
    result = result.resize((result.size[0] // 8, result.size[1] // 8), Image.ANTIALIAS)
    result.save("result.png")


def hillclimb_show_loop(optimizer: Opt, f, bound=15):
    """
    Runs and displays the evolution of an optimizer for hill climbing
    :param optimizer: The initializer optimizer
    :param f: Function to display
    :param bound: the boundary of the function
    """

    # Initialize the plot surface
    b, g = bound, 0.125
    grid_x = np.arange(-b, b, g)
    grid_y = np.arange(-b, b, g)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = f(grid_x, grid_y)

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    surf = ax.plot_surface(grid_x, grid_y, grid_z, linewidth=0, alpha=0.2)
    plt.ion()
    plt.pause(0.01)

    # Define a function to generate plot points from a population
    def xyz_from_pop(population):
        points = np.stack([ind.point for ind in population])
        x = points.T[0, :]
        y = points.T[1, :]
        z = np.array([f(x[j], y[j]) for j in range(len(x))])
        return x, y, z

    # Call the optimizer and define the initial points
    iterator = optimizer.iterate()
    population = next(iterator, None)
    x, y, z = xyz_from_pop(population)
    pop_plt = ax.plot(x, y, z, label="toto", ms=5, color="r", marker="^", ls="")[0]

    # Run the display loop and optimize
    while population is not None:
        x, y, z = xyz_from_pop(population)
        pop_plt.set_data_3d(x, y, z)
        fig.canvas.draw()
        plt.pause(0.01)
        population = next(iterator, None)
