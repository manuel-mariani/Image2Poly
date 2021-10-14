from dataclasses import dataclass
from numbers import Number
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from optimizers.algorithm import GeneticAlgorithm
from optimizers.individual import Individual, Encoding


class PolygonImageIndividual(Individual):
    @staticmethod
    def initialize_population(pop_size, img_bounds, n_polygons, n_vertices, n_colors):
        pop = []
        encoding = PolygonEncoding(
            n_polygons * (n_vertices + n_colors), n_polygons, n_vertices, n_colors
        )
        while len(pop) < pop_size:
            r = np.random.randint(
                0, max(img_bounds), (n_polygons, n_vertices * 2 + n_colors)
            )
            pop.append(
                PolygonImageIndividual([Polygon.from_np(k, encoding) for k in r])
            )
        return pop

    def __init__(self, polygons: List["Polygon"], genome=None):
        super().__init__()
        self.polygons = polygons
        self._genome = genome

    def generate_image(self, size):
        img = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)  # Can be RGBA
        for poly in self.polygons:
            draw.polygon(tuple(map(tuple, poly.vertices)), fill=tuple(poly.colors))
        return img

    def eval(self, x, target) -> Number:
        img = self.generate_image(x.size)
        diff = np.asarray(img) - target
        loss = np.log(abs(np.sum(diff)))
        return loss

    def get_genome(self) -> np.ndarray:
        if self._genome is not None:
            return self._genome
        return np.concatenate([p.to_np() for p in self.polygons]).astype(np.float)

    def get_encoding(self) -> "PolygonEncoding":
        p = self.polygons[0]
        n_poly, n_vert, n_cols = len(self.polygons), len(p.vertices), len(p.colors)
        tot = n_poly * (n_vert + n_cols)
        return PolygonEncoding(tot, n_poly, n_vert, n_cols)

    @staticmethod
    def from_genome(
        genome: np.ndarray, encoding: "PolygonEncoding"
    ) -> "PolygonImageIndividual":
        polygons_genomes = genome.reshape(
            (-1, encoding.n_colors + encoding.n_vertices * 2)
        )
        polygons = [Polygon.from_np(pg, encoding) for pg in polygons_genomes]
        return PolygonImageIndividual(polygons, genome)


@dataclass
class PolygonEncoding(Encoding):
    n_polygons: int
    n_vertices: int
    n_colors: int


@dataclass
class Polygon:
    vertices: np.ndarray
    colors: np.ndarray

    def to_np(self):
        return np.concatenate((self.vertices.flatten(), self.colors.flatten()))

    @staticmethod
    def from_np(array: np.ndarray, encoding: PolygonEncoding):
        array = np.floor(array).astype(int)
        split_at = encoding.n_vertices * 2
        vertices, colors = array[:split_at].reshape(-1, 2), array[split_at:]
        colors[colors < 0] += 255
        colors[colors > 255] -= 255
        return Polygon(vertices, colors)


# -------------------
def polygons_main():
    ga = GeneticAlgorithm(
        pop_size=100,
        initial_population=PolygonImageIndividual.initialize_population(
            100, (10, 10), 100, 3, 3
        ),
        mutation_rate=0.5,
        mutation_strength=10.0,
        elitism=2,
        max_steps=300,
    )
    plt.ion()
    plt.pause(0.01)
    fig = plt.figure()
    ax = fig.add_subplot()
    with Image.open("../monnalisa.jpg") as im:
        im = im.resize(tuple(np.array(im.size) // 4))
        # show = ax.imshow(im)
        # plt.pause(0.1)
        while ga.current_step < ga.max_steps:
            ga.run(im, im)
            ga.population.sort(key=lambda i: i.loss)
            best = ga.population[0]
            best_image = best.generate_image(im.size)
            ax.imshow(best_image)
            plt.pause(0.01)

    input("=============DONE=============")


if __name__ == "__main__":
    polygons_main()
