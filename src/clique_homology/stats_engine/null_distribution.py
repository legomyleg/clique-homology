from clique_homology import random_coloring, Color, betti_numbers
from networkit.graph import Graph # type: ignore
import numpy as np

def null_distribution(graph: Graph, coloring: list[Color], iterations:int=100) -> list[np.ndarray]:
    
    distribution: list[np.ndarray] = []
    
    for _ in range(iterations):
        new_coloring: list[Color] = random_coloring(coloring)
        distribution.append(betti_numbers(graph, new_coloring))
    
    return [np.array(None)]