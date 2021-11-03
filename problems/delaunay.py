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

    def __init__(self, coordinates: np.ndarray, genome: np.ndarray):
        super().__init__()
        self.coordinates = coordinates
        self.genome = genome
        self.sanitize()

    def generate_image(self, size, target_image):
        coords = self.coordinates.T
        img = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        triangulation = Delaunay(coords)
        polygons = coords[triangulation.vertices]
        for poly in polygons:
            barycenter = np.mean(poly, axis=0, dtype=int)
            color = target_image[barycenter[1], barycenter[0]]
            draw.polygon(tuple(poly.ravel()), fill=tuple(color))
        return img

    def generate_image_(self, size, target_image):
        # TODO: Delete me
        triangulation = tri.Triangulation(*self.coordinates)
        img = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        coords = self.coordinates.T
        for idx, triangle_indices in enumerate(triangulation.triangles):
            triangle = coords[triangle_indices, :]
            barycenter = np.mean(triangle, axis=0, dtype=int)
            color = target_image[barycenter[1], barycenter[0]]
            # FAST
            # color = target_image[triangle[0, 1], triangle[0, 0]]
            draw.polygon(tuple(triangle.ravel()), fill=tuple(color))
        return img

    def eval(self, size, target_image) -> Number:
        img = self.generate_image(size, target_image)
        img = np.asarray(img)
        color_diff = np.abs(img - target_image) / (255 * 3)
        loss = np.sum(color_diff) / np.prod(size)
        return loss

    def get_genome(self) -> np.ndarray:
        return self.genome
        # return self.coordinates.flatten().astype(np.float32)

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
    def initialize_population(pop_size, n_points, image_shape):
        pop = []
        max_x, max_y = image_shape
        DelaunayIndividual.image_shape = image_shape
        while len(pop) < pop_size:
            c1 = np.random.uniform(0, max_x, n_points)
            c2 = np.random.uniform(0, max_y, n_points)
            coordinates = np.vstack((c1, c2))
            individual = DelaunayIndividual(
                coordinates.astype(int), coordinates.flatten()
            )
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
