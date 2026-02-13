import unittest
import networkit as nk
import random

# --- The Updated Function (for context) ---
def generate_colored_graph(n, m):
    """
    Generates a graph with n nodes and m possible colors.
    Edges are only created between nodes of the same color (p=0.5).
    Returns: (G, node_colors)
    """
    G = nk.Graph(n) # efficient: init with n nodes immediately
    colors = list(range(m))
    node_colors = []
    
    # Assign colors
    for i in range(n):
        color = random.choice(colors)
        node_colors.append(color)
    
    # Create edges based on color match
    for i in range(n):
        for j in range(i+1, n):
            if node_colors[i] == node_colors[j]:
                if random.random() < 0.5:
                    G.addEdge(i, j)
                    
    return G, node_colors

# --- The Unit Tests ---
class TestColoredGraph(unittest.TestCase):

    def setUp(self):
        # Set seed for reproducibility in tests
        random.seed(42)

    def test_return_structure(self):
        """Test that it returns a Graph and a list of correct length."""
        n, m = 10, 3
        G, colors = generate_colored_graph(n, m)
        
        self.assertIsInstance(G, nk.Graph)
        self.assertIsInstance(colors, list)
        self.assertEqual(G.numberOfNodes(), n)
        self.assertEqual(len(colors), n)

    def test_homophily_constraint(self):
        """
        CRITICAL TEST: Verify that edges ONLY exist between 
        nodes of the same color.
        """
        n, m = 100, 5
        G, colors = generate_colored_graph(n, m)
        
        # Iterate over every single edge in the graph
        for u, v in G.iterEdges():
            self.assertEqual(
                colors[u], 
                colors[v], 
                f"Found an invalid edge! Node {u} (color {colors[u]}) is connected to Node {v} (color {colors[v]})"
            )

    def test_single_color_connectivity(self):
        """
        If m=1, all nodes are the same color.
        We expect a dense graph (roughly 50% density).
        """
        n, m = 50, 1
        G, colors = generate_colored_graph(n, m)
        
        # Check that everyone is color 0
        self.assertTrue(all(c == 0 for c in colors))
        
        # Check that we have edges (it's nearly impossible to have 0 edges with n=50, p=0.5)
        self.assertTrue(G.numberOfEdges() > 0)
        
        # Optional: Check density is roughly 0.5 (allow variance)
        max_edges = n * (n - 1) / 2
        density = G.numberOfEdges() / max_edges
        self.assertAlmostEqual(density, 0.5, delta=0.15)

    def test_zero_nodes(self):
        """Test edge case with n=0."""
        G, colors = generate_colored_graph(0, 5)
        self.assertEqual(G.numberOfNodes(), 0)
        self.assertEqual(len(colors), 0)

    def test_unique_colors_disconnect(self):
        """
        If every node has a unique color (m >= n, and we force uniqueness),
        there should be ZERO edges.
        """
        # We can't easily force uniqueness with random.choice, 
        # so we test the property: different colors = no edge.
        n, m = 20, 20
        G, colors = generate_colored_graph(n, m)
        
        # Manually check pairs that are NOT connected
        # If color[i] != color[j], assert hasEdge is False
        for i in range(n):
            for j in range(i+1, n):
                if colors[i] != colors[j]:
                    self.assertFalse(G.hasEdge(i, j), 
                        f"Nodes {i} and {j} have different colors but are connected!")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)