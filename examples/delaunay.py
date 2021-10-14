from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import matplotlib.tri as tri
import numpy as np
from PIL import Image, ImageDraw

from optimizers.individual import Individual, Encoding


class DelaunayIndividual(Individual):
    image_shape = None

    def __init__(
        self, coordinates: np.ndarray, palette: np.ndarray, colors: np.ndarray
    ):
        super().__init__()
        self.coordinates = coordinates
        self.palette = palette
        self.colors = colors
        self.sanitize()

    def generate_image(self, size):
        triangulation = tri.Triangulation(*self.coordinates)
        img = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        coords = self.coordinates.T
        for idx, triangle_indices in enumerate(triangulation.triangles):
            triangle = coords[triangle_indices, :].flatten()
            palette_idx = self.colors[idx]
            color = self.palette[palette_idx]
            draw.polygon(tuple(triangle), fill=tuple(color))
        return img

    def eval(self, size, target_image) -> Number:
        img_color, img_edges = target_image
        img = self.generate_image(size)
        # color_diff = np.asarray(img) - img_color
        color_diff = 0
        edges_diff = 0
        for x, y in self.coordinates.T:
            edges_diff += 5 if img_edges[y, x] > 0 else 0
        loss = np.sum(color_diff) / np.prod(size) * 3
        loss = loss + edges_diff * 5
        loss = abs(loss)
        return loss

    def get_genome(self) -> np.ndarray:
        return np.concatenate(
            (self.coordinates.flatten(), self.palette.flatten(), self.colors.flatten())
        ).astype(np.float32)

    def get_encoding(self) -> "DelaunayEncoding":
        return DelaunayEncoding(
            self.coordinates.size + self.palette.size + self.colors.size,
            self.coordinates.shape,
            self.palette.shape,
            self.colors.shape,
        )

    def sanitize(self):
        self.coordinates = self.coordinates.astype(int)
        self.palette = self.palette.astype(int)
        self.colors = self.colors.astype(int)

        self.coordinates[0, :] = np.clip(
            self.coordinates[0, :], 0, self.image_shape[0] - 1
        )
        self.coordinates[1, :] = np.clip(
            self.coordinates[1, :], 0, self.image_shape[1] - 1
        )

        self.palette = np.clip(-255, 510, self.palette)
        self.palette[self.palette < 0] += 255
        self.palette[self.palette > 255] -= 255

        c_len = self.palette.shape[0] - 1
        self.colors = np.clip(self.colors, -c_len, 2 * c_len)
        # self.colors = np.clip(-c_len, 2 * c_len, self.colors)
        self.colors[self.colors < 0] += c_len
        self.colors[self.colors > c_len] -= c_len

    @staticmethod
    def from_genome(
        genome: np.ndarray, encoding: "DelaunayEncoding"
    ) -> "DelaunayIndividual":
        assert genome.size == encoding.length
        a, b, c = encoding.slices
        coordinates = genome[0:a].reshape(encoding.coordinates_shape)
        palette = genome[a:b].reshape(encoding.palette_shape)
        colors = genome[b:c].reshape(encoding.colors_size)
        return DelaunayIndividual(coordinates, palette, colors)

    @staticmethod
    def initialize_population(pop_size, n_points, image_shape, n_palette_colors):
        pop = []
        max_x, max_y = image_shape
        DelaunayIndividual.image_shape = image_shape
        while len(pop) < pop_size:
            coordinates = np.vstack(
                [
                    np.random.randint(0, max_x, n_points),
                    np.random.randint(0, max_y, n_points),
                ]
            )
            palette = np.random.randint(0, 255, (n_palette_colors, 3))
            colors = np.random.randint(0, n_palette_colors, (n_points * 2,))
            individual = DelaunayIndividual(coordinates, palette, colors)
            pop.append(individual)
        return pop

    @staticmethod
    def get_mutation_weights(encoding: "DelaunayEncoding") -> np.ndarray:
        w = np.zeros(encoding.length)
        a, b, c = encoding.slices
        w[0:a] = min(DelaunayIndividual.image_shape)
        w[a:b] = 255
        w[b:c] = encoding.palette_shape[0]
        return w


@dataclass
class DelaunayEncoding(Encoding):
    coordinates_shape: Tuple[int, ...]
    palette_shape: Tuple[int, ...]
    colors_shape: Tuple[int, ...]

    @property
    def coordinates_size(self):
        return np.prod(self.coordinates_shape)

    @property
    def palette_size(self):
        return np.prod(self.palette_shape)

    @property
    def colors_size(self):
        return np.prod(self.colors_shape)

    @property
    def sizes(self):
        return self.coordinates_size, self.palette_size, self.colors_size

    @property
    def slices(self):
        a, b, c = self.sizes
        return a, a + b, a + b + c
