<div align="center">
    <img src="assets/monnalisa_poly.png" style="height:300px">
</div>

# Image2Poly
Convert images into low-polygon-count versions.

It works by optimizing a set of points, which are then triangulated using Delaunay to produce a set of polygons.
The color of each polygon is determined by the original image's color at the polygon barycenter.

The optimizer downscales the image by a customizable factor, but the final result is of the same size (by default) as the original image, with super sampling anti aliasing to reduce artifacts. 

## Installation
Clone the repository, navigate to the folder and install the requirements using [`poetry`](https://github.com/python-poetry/poetry) or `pip`

Using `pip`
```
pip install -r requirements.txt
```

Using `poetry`
```
poetry install
```

## Usage
Simply run `python main.py` or `poetry run python main.py`

## Configuration
The configuration is in `main.py`:

| Constant | Values | Description |
| --- | --- | --- |
| OPTIMIZER | Optimizer.[GA \| SA \| PSO \| CMA] | The optimizer to use: **Genetic Algorithm**, **Simulated Annealing**, **Particle Swarm Optimization**, **Covariance Matrix Adaptation**.
| OPTIMIZATION_PROBLEM | Problem.[DELAUNAY \| HILLCLIMB] | The optimization problem to optimize. Hillclimb is only for debug / learning purposes. |
| IMAGE_PATH | \<str> | Path to the image to convert. |
| IMAGE_DOWNSCALING | \<int> | Downscaling factor for the optimization process (does not impact the final result). |
| POP_SIZE | \<int> | Number of individuals. |
| MAX_STEPS | \<int> | Number of optimization steps until termination. |
| N_POINTS | \<int> | Number of points to triangulate. |
| EXPLORATION_FACTOR | \<float> | **SA** Strength of mutation. |
| TAU_INI | \<float> | **SA & GA** Temperature / selectivity pressure (initial) |
| TAU_END | \<float> | **SA & GA** Temperature / selectivity pressure (final) |
| MUTATION_RATE | \<float> | **GA** Probability of mutating each gene in the genome. |
| MUTATION_STRENGTH | \<float> | **GA** Multiplicative coefficient for the mutation. |
| ELITISM | \<int> | **GA** Number of best individuals to keep at each generation. |
| CROSSOVER_POINTS | \<int> | **GA** Number of genome-sub-sequence to choose from each parent during crossover. |
| INERTIA | \<float> | **PSO** Inertia of velocity. |
| PHI_COGNITIVE | \<float> | **PSO** Velocity weight for the particle's best found position. |
| PHI_SOCIAL | \<float> | **PSO** Velocity weight for the swarm's best found position. |
| VELOCITY_STRENGTH | \<float> | **PSO** Initial velocity factor. |
| SIGMA | \<float> | **CMA** Initial step size |

## Extending the project
The provides interfaces to customize both the optimizer or the optimization problem.

For example, it is possible to define a new optimization problem simply by implementing the `Individual` abstract class, and then use an optimizer to minimize its loss.

## Future developments
This project has been done has work for the CMDO course @UNIBO-AI 2020/21, so performance was not its main objective. 
For this version no CLI or library is planned, but in the future a faster version made in Rust will be released.
