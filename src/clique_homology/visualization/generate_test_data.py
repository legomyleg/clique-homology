import networkit as nk
import random

def generate_colored_graph(n):
    G = nk.Graph()
    colors = list(range(4)) # start with 4 colors, we can add more later
    node_colors = []
    for i in range(n):
        color = random.choice(colors)
        node_colors.append(color)
        G.addNode()
    for i in range(n):
        for j in range(i+1, n):
            if node_colors[i] == node_colors[j]:
                if random.random() < 0.5:
                    G.addEdge(i, j)

    return G, node_colors


import networkit as nk
import random

def generate_connected_colored_graph(n, m=4):
    """
    Generates a graph that is guaranteed to be connected, 
    while still favoring edges between nodes of the same color.
    """
    if n < 1:
        return nk.Graph(), []

    G = nk.Graph(n)
    colors = list(range(m))
    node_colors = [random.choice(colors) for _ in range(n)]

    # 1. Guarantee Connectivity (The Backbone)
    # We create a simple spanning tree by connecting each node i > 0 
    # to a random node that came before it.
    for i in range(1, n):
        target = random.randint(0, i - 1)
        G.addEdge(i, target)

    # 2. Add Homophily Edges (Your original logic)
    # We add additional edges between nodes of the same color
    for i in range(n):
        for j in range(i + 1, n):
            # If they are same color AND don't already have an edge from the backbone
            if node_colors[i] == node_colors[j] and not G.hasEdge(i, j):
                if random.random() < 0.5:
                    G.addEdge(i, j)

    return G, node_colors

# Example of how to use both together:
# G_disc, colors_disc = generate_colored_graph(30) # Your original
# G_conn, colors_conn = generate_connected_colored_graph(30) # The new one