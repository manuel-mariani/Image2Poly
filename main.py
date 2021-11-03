import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from matplotlib.animation import FuncAnimation

from optimizers.annealing import SimulatedAnnealing
from optimizers.cma_es import CmaEs
from optimizers.optimizer import Optimizer
from problems.delaunay import DelaunayIndividual

# ====================== CONFIG ======================
# - Shared

IMAGE_PATH = "assets/monnalisa.jpg"
IMAGE_DOWNSCALING = 2
POP_SIZE = 100
MAX_STEPS = 1000

N_POINTS = 500
N_VERTICES = 3
N_COLORS = 10

# - Simulated annealing
EXPLORATION_FACTOR = 0.5
TAU_INI = 2
TAU_END = 0.01


def main():
    img, edg = load_image(IMAGE_PATH)
    delaunay = DelaunayIndividual.initialize_population(
        POP_SIZE, N_POINTS, img.size[:2]
    )
    annealing = SimulatedAnnealing(
        delaunay[0], MAX_STEPS, EXPLORATION_FACTOR, TAU_INI, TAU_END
    )
    # image_show_loop(annealing, img)
    cma_es = CmaEs(MAX_STEPS, delaunay[0], 0.9)
    image_show_loop(cma_es, img)


# ====================== UTILS ======================
def load_image(path, downscaling=IMAGE_DOWNSCALING):
    with Image.open(path) as img:
        img = img.resize((img.size[0] // downscaling, img.size[1] // downscaling))
        edges = img.filter(ImageFilter.FIND_EDGES)
        return img, edges


def image_show_loop(
    optimizer: Optimizer, target_image: Image.Image
):

    size = target_image.size[:2]
    target_image = np.asarray(target_image)
    im = plt.imshow(optimizer.best_individual.generate_image(size, target_image))
    plt.axis("off")
    plt.gcf().tight_layout()
    plt.gcf().set_size_inches(size[0] / 100, size[1] / 100)
    toolbar = plt.get_current_fig_manager().toolbar
    for a in toolbar.actions():
        toolbar.removeAction(a)

    iterator = optimizer.iterate(size, target_image)

    def update(individual):
        if individual is not None:
            gi = individual.generate_image(size, target_image)
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


if __name__ == "__main__":
    main()
