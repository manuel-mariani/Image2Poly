import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from rich import print

from examples.delaunay import DelaunayIndividual
from optimizers.genetic import GeneticAlgorithm


def test(pop):
    genome = pop[0].get_genome()
    encoding = pop[0].get_encoding()
    ind = DelaunayIndividual.from_genome(genome, encoding)
    assert np.array_equiv(ind.coordinates, pop[0].coordinates)
    assert np.array_equiv(ind.palette, pop[0].palette)
    assert np.array_equiv(ind.colors, pop[0].colors)

    img = ind.generate_image((500, 1000))
    plt.imshow(img)
    plt.show()


def edges(img):
    img = (
        img.convert("L")
        .filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0))
        .point(lambda x: 255 if x < 35 else 0, "1")
    )
    # img.show()
    return img


def delaunay_main():
    with Image.open("../monnalisa.jpg") as img:
        img = img.resize(tuple(np.array(img.size) // 4))
        pop_size = 100
        pop = DelaunayIndividual.initialize_population(pop_size, 100, img.size[:2], 16)
        ga = GeneticAlgorithm(
            pop_size=pop_size,
            initial_population=pop,
            mutation_rate=0.1,
            mutation_strength=0.1,
            elitism=1,
            max_steps=1000,
        )

        plt.ion()
        plt.pause(0.0001)
        fig = plt.figure()
        ax = fig.add_subplot()
        size = img.size

        img_color = img
        img_edges = np.asarray(edges(img))

        for pop in ga.iterate(size, (img_color, img_edges)):
            best = min(pop, key=lambda i: i.loss)
            print(
                f"[bold cyan] - Loss:[/] {best.loss} [magenta]f[{ga.current_step}/{ga.max_steps}]"
            )
            ax.imshow(best.generate_image(size))
            plt.pause(0.01)


if __name__ == "__main__":
    # edges()
    delaunay_main()
