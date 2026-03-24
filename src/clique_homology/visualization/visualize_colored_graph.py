import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
from . import generate_test_data
from clique_homology.stats_engine import betti_numbers
import math
import random


def get_packed_layout(G):
    """
    Layouts disconnected components separately and packs them into a grid.
    Returns a dictionary of positions {node: [x, y]}.
    """
    pos = {}
    # Get all connected components (the subgraphs)
    # sorted by size so we pack them neatly
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    
    # Calculate grid size (e.g., for 4 components, we want a 2x2 grid)
    side_len = math.ceil(math.sqrt(len(components)))
    
    # Scale factor: how much space each subgraph gets
    scale = 2.5 
    
    for i, nodes in enumerate(components):
        # Create a subgraph for just this component
        subgraph = G.subgraph(nodes)
        
        # Calculate grid coordinates (row, col)
        row = i // side_len
        col = i % side_len
        
        # Calculate center for this component: (col * scale, row * scale)
        center_x = col * scale
        center_y = row * scale # Invert row if you want top-down
        
        # Run spring layout ONLY for this component, centered at our calculated spot
        # k=None uses default spacing within the component
        sub_pos = nx.spring_layout(subgraph, center=(center_x, center_y), seed=42)
        
        # Add these positions to our master list
        pos.update(sub_pos)
        
    return pos


def visualize_colored_graph(G: nk.Graph, node_colors: list):
    """
    Create a visualization of a colored graph, as well as each colored subgraph.

    :param G: A NetworKit graph.
    :param node_colors: A list of integers matching the node IDs.
    """
    # 1. Convert NetworKit -> NetworkX
    nx_graph = nk.nxadapter.nk2nx(G)
    
    # Define our color palette (extend this list if you have >4 colors)
    palette = ['red', 'blue', 'green', 'orange']
    
    # Map the integer IDs in 'node_colors' to actual color strings
    # e.g., [0, 1, 0] -> ['red', 'blue', 'red']
    visual_colors = [palette[c % len(palette)] for c in node_colors]

    # ==========================================
    # PLOT 1: The Whole Graph
    # ==========================================
    plt.figure(figsize=(8, 8)) # New independent figure
    
    # Calculate layout once (can reuse 'pos' for subgraphs if you want them relative)
    pos = get_packed_layout(nx_graph)
    
    nx.draw(nx_graph, pos=pos, node_color=visual_colors, 
            with_labels=True, node_size=500, edge_color="gray", font_color="white")
    
    plt.title("Full Colored Graph")
    plt.savefig('clique_homology/visualization/plots/full_graph.png')
    
    
# ==========================================
# PLOT 2: The Subgraphs Grid
# ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    subgraphs = betti_numbers.get_colored_subgraphs(G, node_colors)

    plot_colors = ['red', 'blue', 'green', 'orange']

    index = 0

    for nk_subgraph in subgraphs:
        if index > 3: 
            break

        nx_subgraph = nk.nxadapter.nk2nx(nk_subgraph)
        row = index // 2
        col = index % 2
        sub_pos = nx.spring_layout(nx_subgraph, seed=42)
        color = palette[index % len(palette)]
        
        # Added with_labels=True to the line below
        nx.draw(nx_subgraph, ax=axes[row, col], node_color=color, with_labels=True, edge_color="gray", font_color="white")

        index += 1

    plt.tight_layout() # Prevents overlap
    plt.savefig('clique_homology/visualization/plots/colored_graph.png')





def visualize_random_colored_graph():
    """
    Generate a random colored graph and visualize it.
    """
    # UNPACK the tuple here (G, colors)
    G, node_colors = generate_test_data.generate_colored_graph(30)
    
    # Pass BOTH to the visualizer
    visualize_colored_graph(G, node_colors)

def visualize_random_connected_colored_graph():
    """
    Generate a random colored graph and visualize it.
    """
    # UNPACK the tuple here (G, colors)
    G, node_colors = generate_test_data.generate_connected_colored_graph(30)
    
    # Pass BOTH to the visualizer
    visualize_colored_graph(G, node_colors)    


def visualize_four_random_colorings(n_nodes: int):
    """
    Generates a random graph of n nodes, applies 4 random colorings 
    (using 3 colors), and plots them as subplots on a 2x2 grid.
    """
    # 1. Generate the base graph using your existing generator (we ignore the default colors)
    G, _ = generate_test_data.generate_connected_colored_graph(n_nodes)
    
    # 2. Convert NetworKit -> NetworkX
    nx_graph = nk.nxadapter.nk2nx(G)
    
    # 3. Calculate layout ONCE so the graph structure is identical across all 4 subplots
    pos = get_packed_layout(nx_graph)
    
    # 4. Define our 3-color palette
    palette = ['red', 'blue', 'green']
    
    # 5. Set up the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    for index in range(4):
        # Generate random colors (0, 1, or 2) for each node
        node_colors = [random.randint(0, 2) for _ in range(n_nodes)]
        
        # Map integer IDs to actual color strings
        visual_colors = [palette[c % len(palette)] for c in node_colors]
        
        row = index // 2
        col = index % 2
        
        # Draw using the consistent settings from your other plots
        nx.draw(nx_graph, pos=pos, ax=axes[row, col], node_color=visual_colors, 
                with_labels=True, node_size=500, edge_color="gray", font_color="white")

    plt.tight_layout() # Prevents overlap
    plt.savefig('clique_homology/visualization/plots/four_random_colorings.png')








# Run it
if __name__ == "__main__":
    # visualize_random_colored_graph()
    visualize_four_random_colorings(12)