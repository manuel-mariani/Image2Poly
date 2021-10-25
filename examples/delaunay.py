from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import matplotlib.tri as tri
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

from optimizers.individual import Individual, Encoding


class DelaunayIndividual(Individual):
    image_shape = None

    def __init__(self, coordinates: np.ndarray):
        super().__init__()
        self.coordinates = coordinates
        self.sanitize()

    def generate_image(self, size, target_image, outline=None):
        coords = self.coordinates.T
        img = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        triangulation = Delaunay(coords)
        vertexes = coords[triangulation.vertices]
        for vert in vertexes:
            color = target_image[vert[0, 1], vert[0, 0]]
            draw.polygon(tuple(vert.ravel()), fill=tuple(color), outline=outline)
        return img

    def generate_image_(self, size, target_image, outline=None):
        triangulation = tri.Triangulation(*self.coordinates)
        img = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        coords = self.coordinates.T
        for idx, triangle_indices in enumerate(triangulation.triangles):
            triangle = coords[triangle_indices, :]
            # ALTERNATIVE
            # min_x, max_x = np.min(triangle[:, 0]), np.max(triangle[:, 0])
            # min_y, max_y = np.min(triangle[:, 1]), np.max(triangle[:, 1])
            # color = np.mean(
            #     target_image[min_y:max_y, min_x:max_x], axis=(0, 1), dtype=int
            # )
            # SLOWER
            # barycenter = np.mean(triangle, axis=0, dtype=int)
            # color = target_image[barycenter[1], barycenter[0]]
            # FAST
            color = target_image[triangle[0, 1], triangle[0, 0]]
            draw.polygon(tuple(triangle.ravel()), fill=tuple(color), outline=outline)
        return img

    def eval(self, size, target_image) -> Number:
        img = self.generate_image(size, target_image)
        img = np.asarray(img)
        color_diff = np.abs(img - target_image) / (255 * 3)
        loss = np.sum(color_diff) / np.prod(size)
        return loss

    def get_genome(self) -> np.ndarray:
        return self.coordinates.flatten().astype(np.float32)

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
        return DelaunayIndividual(coordinates)

    @staticmethod
    def initialize_population(pop_size, n_points, image_shape):
        pop = []
        max_x, max_y = image_shape
        DelaunayIndividual.image_shape = image_shape
        while len(pop) < pop_size:
            c1 = np.random.normal(max_x / 2, max_x / 4, n_points)
            c2 = np.random.normal(max_y / 2, max_y / 4, n_points)
            coordinates = np.vstack((c1, c2)).astype(int)
            individual = DelaunayIndividual(coordinates)
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
