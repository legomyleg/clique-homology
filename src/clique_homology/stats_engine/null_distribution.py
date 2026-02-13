<<<<<<< HEAD
from clique_homology import random_coloring, betti_numbers
from networkit.graph import Graph # type: ignore
import numpy as np

def null_distribution(
    graph: Graph,
    coloring: list[str],
    iterations: int = 100,
    allowed_colors: list[str] | None = None,
) -> list[np.ndarray]:
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")

    distribution: list[np.ndarray] = []
    
    for _ in range(iterations):
        new_coloring = random_coloring(coloring, allowed_colors=allowed_colors)
        distribution.append(betti_numbers(graph, new_coloring, allowed_colors=allowed_colors))
    
    return distribution
=======
from clique_homology import random_coloring, betti_numbers
from networkit.graph import Graph # type: ignore
import numpy as np

def null_distribution(
    graph: Graph,
    coloring: list[str],
    iterations: int = 100,
    allowed_colors: list[str] | None = None,
) -> list[np.ndarray]:
    
    distribution: list[np.ndarray] = []
    
    for _ in range(iterations):
        new_coloring = random_coloring(coloring, allowed_colors=allowed_colors)
        distribution.append(betti_numbers(graph, new_coloring, allowed_colors=allowed_colors))
    
    return distribution
>>>>>>> 31884b0 (added visualization package (Coby))
