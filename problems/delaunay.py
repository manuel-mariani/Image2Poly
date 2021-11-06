from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

from problems.individual import Individual, Encoding


class DelaunayIndividual(Individual):
    # Static attributes (shape, color, edges)
    image_shape = None
    image_color = None
    image_edges = None

    def __init__(self, coordinates: np.ndarray):
        super().__init__()

        # Wrap the coordinates within the image bounds
        oob_0 = coordinates[:, :] < 0
        coordinates[oob_0] = -coordinates[oob_0]

        oob_x = coordinates[0, :] >= self.image_shape[0]
        oob_y = coordinates[1, :] >= self.image_shape[1]
        coordinates[0, oob_x] = coordinates[0, oob_x] % self.image_shape[0]
        coordinates[1, oob_y] = coordinates[1, oob_y] % self.image_shape[1]

        # Store the coordinates and the genome
        self.coordinates = coordinates
        self.genome = self.coordinates.flatten()

    def generate_image(self, upscaling=1):
        """Generate an image, using the points in the instance and the color image in the class"""

        # Handle upscaling (used in producing the final output)
        size = self.image_shape[0] * upscaling, self.image_shape[1] * upscaling

        # Triangulate the coordinates using Delaunay triangulation
        coords = self.coordinates.T
        img = Image.new("RGB", size, (0, 0, 0))
        triangulation = Delaunay(coords)

        # Initialize the image drawer and for each polygon in the triangulation draw it
        draw = ImageDraw.Draw(img)
        polygons = coords[triangulation.vertices]
        for poly in polygons:
            barycenter = np.mean(
                poly, axis=0, dtype=int
            )  # Compute the barycenter of the polygon
            color = self.image_color[
                barycenter[1], barycenter[0]
            ]  # Pick the color from in the barycenter
            poly *= upscaling
            draw.polygon(tuple(poly.ravel()), fill=tuple(color))  # Draw
        return img

    def eval(self) -> Number:
        # Generate the image and convert it into matrix
        img = self.generate_image()
        img = np.asarray(img)

        # Compute the pixel wise loss (MAE)
        color_loss = (
            np.sum(np.abs(img - self.image_color))
            / (255 * 3)
            / np.prod(self.image_shape)
        )

        # Compute the edge-loss:
        # ideally 0 if each point is over an edge
        # and 1 otherwise
        coords = self.coordinates.astype(int).T
        edges_loss = self.image_edges[coords[:, 1], coords[:, 0]]
        edges_loss = np.sum(edges_loss) / (255 * self.coordinates.size / 2)

        # Mix the color and the edges losses by a factor
        k = 0.3
        loss = k * color_loss + (1 - k) * edges_loss
        return loss

    def get_genome(self) -> np.ndarray:
        return self.genome

    def get_encoding(self) -> "DelaunayEncoding":
        return DelaunayEncoding(self.coordinates.size, self.coordinates.shape)

    @staticmethod
    def from_genome(
        genome: np.ndarray, encoding: "DelaunayEncoding"
    ) -> "DelaunayIndividual":
        # Create an individual from a genome. In this case it is simply a reshaping
        coordinates = genome.reshape(encoding.coordinates_shape)
        return DelaunayIndividual(coordinates)

    @staticmethod
    def initialize_population(
        pop_size, n_points, image_shape, image_color, image_edges
    ):
        # Initialize the class parameters
        DelaunayIndividual.image_shape = image_shape
        DelaunayIndividual.image_color = image_color
        DelaunayIndividual.image_edges = image_edges

        pop = []
        max_x, max_y = image_shape
        while len(pop) < pop_size:
            # Create random points, bounded inside the image
            c1 = np.random.uniform(0, max_x, n_points)
            c2 = np.random.uniform(0, max_y, n_points)
            coordinates = np.vstack((c1, c2))
            # Store the points in an individual and append it to the population
            individual = DelaunayIndividual(coordinates)
            pop.append(individual)
        return pop

    @staticmethod
    def get_mutation_weights(encoding: "DelaunayEncoding") -> np.ndarray:
        w = np.ones(encoding.coordinates_shape)
        w[0, :] *= DelaunayIndividual.image_shape[0]
        w[1, :] *= DelaunayIndividual.image_shape[1]
        return w.flatten()


@dataclass
class DelaunayEncoding(Encoding):
    # This encoding stores also the shapes of the coordinates (2 x N_POINTS) to reshape the flattened genome
    coordinates_shape: Tuple[int, ...]
