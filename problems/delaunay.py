from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import matplotlib.tri as tri
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

from problems.individual import Individual, Encoding


class DelaunayIndividual(Individual):
    image_shape = None
    image_color = None
    image_edges = None

    def __init__(self, coordinates: np.ndarray, genome: np.ndarray):
        super().__init__()

        # Wrap the coordinates within the image bounds
        oob_0 = coordinates[:, :] < 0
        coordinates[oob_0] = -coordinates[oob_0]

        oob_x = coordinates[0, :] >= self.image_shape[0]
        oob_y = coordinates[1, :] >= self.image_shape[1]
        coordinates[0, oob_x] = coordinates[0, oob_x] % self.image_shape[0]
        coordinates[1, oob_y] = coordinates[1, oob_y] % self.image_shape[1]

        self.coordinates = coordinates
        self.genome = self.coordinates.flatten()

    def generate_image(self):
        coords = self.coordinates.T
        img = Image.new("RGB", self.image_shape, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        triangulation = Delaunay(coords)
        polygons = coords[triangulation.vertices]
        for poly in polygons:
            barycenter = np.mean(poly, axis=0, dtype=int)
            color = self.image_color[barycenter[1], barycenter[0]]
            draw.polygon(tuple(poly.ravel()), fill=tuple(color))
        return img

    def eval(self) -> Number:
        img = self.generate_image()
        img = np.asarray(img)

        color_loss = (
            np.sum(np.abs(img - self.image_color))
            / (255 * 3)
            / np.prod(self.image_shape)
        )
        edges_loss = self.image_edges[self.coordinates.astype(int)] / 255
        edges_loss = np.sum(edges_loss) / edges_loss.size
        edges_loss = edges_loss ** 2
        k = 0
        loss = k * color_loss + (1 - k) * edges_loss
        return loss

    def get_genome(self) -> np.ndarray:
        return self.genome

    def get_encoding(self) -> "DelaunayEncoding":
        return DelaunayEncoding(self.coordinates.size, self.coordinates.shape)

    def sanitize(self):
        self.coordinates = self.coordinates.astype(int)
        self.coordinates[0, :] = np.clip(
            self.coordinates[0, :], 0, self.image_shape[0] - 1
        )
        self.coordinates[1, :] = np.clip(
            self.coordinates[1, :], 0, self.image_shape[1] - 1
        )

    @staticmethod
    def from_genome(
        genome: np.ndarray, encoding: "DelaunayEncoding"
    ) -> "DelaunayIndividual":
        coordinates = genome.reshape(encoding.coordinates_shape)
        return DelaunayIndividual(coordinates, genome)

    @staticmethod
    def initialize_population(
        pop_size, n_points, image_shape, image_color, image_edges
    ):
        pop = []
        max_x, max_y = image_shape
        DelaunayIndividual.image_shape = image_shape
        DelaunayIndividual.image_color = image_color
        DelaunayIndividual.image_edges = image_edges

        while len(pop) < pop_size:
            c1 = np.random.uniform(0, max_x, n_points)
            c2 = np.random.uniform(0, max_y, n_points)
            coordinates = np.vstack((c1, c2))
            individual = DelaunayIndividual(coordinates, coordinates.flatten())
            pop.append(individual)
        return pop

    @staticmethod
    def get_mutation_weights(encoding: "DelaunayEncoding") -> np.ndarray:
        w = np.ones(encoding.length)
        w *= min(DelaunayIndividual.image_shape)
        return w


@dataclass
class DelaunayEncoding(Encoding):
    coordinates_shape: Tuple[int, ...]
