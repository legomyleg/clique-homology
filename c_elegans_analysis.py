from clique_homology.stats_engine.stats_engine import stats_engine
from docs.data.c_elegans_colors import c_elegans_colors
from docs.data.c_elegans_edges import c_elegans_edges

import networkit as nk

if __name__ == "__main__":
    # 2. Initialize the graph
    edges = c_elegans_edges()
    colors = c_elegans_colors()
    num_nodes = len(colors)
    g = nk.Graph(num_nodes, weighted=False, directed=False)

    # 3. Add edges efficiently
    for u, v in edges:
        g.addEdge(u, v)
    g.removeSelfLoops()
    print("P-value:", stats_engine(g, colors, iters=100))
    print("Number of nodes:", num_nodes)
    print("Number of edges:", len(edges))