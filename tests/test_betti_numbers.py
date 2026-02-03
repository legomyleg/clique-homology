import pytest
import networkx as nx
import networkit as nk
from random import choice
from clique_homology import betti_numbers

v_1_0_0 = {0, 1, 2}
e_1_0_0 = [(0, 1), (0, 2), (1, 2)]
c1 = {0: "blue", 1: "blue", 2:"blue"}

v_1_1_0 = {0, 1, 2, 3}
e_1_1_0 = [(0, 1), (0, 3), (1, 2), (2, 3)]
c2 = {0: "blue", 1: "red"}

v_1_0_1 = {0, 1, 2, 3, 4}
e_1_0_1 = [(0, 1), (0, 2), (0, 3), (0, 4), 
           (1, 2), (1, 3), 
           (2, 3), (2, 4), 
           (3, 4)]

def test_simple_betti_0_0():
    pass

def test_simple_betti_1_0():
    pass

def test_simple_betti_0_1():
    pass

def test_simple_betto_1_1():
    pass

def test_simple_betti_2_0():
    pass

def test_simple_betti_0_2():
    pass

def test_simple_betto_2_2():
    pass

# --- test Betti numbers ---

def test_bett_numbers_1():
    G = nx.petersen_graph()
    attr = dict(zip(list(G.nodes), [{"color": choice(["red", "blue"])} for _ in range(G.number_of_nodes())]))
    print(attr)
    nx.set_node_attributes(G, attr)
    print(betti_numbers.betti_numbers(G))

# --- Test helper functions individually ---

def test_parse_graph_input_1():
    pass

def test_parse_graph_input_2():
    pass

def test_parse_graph_input_3():
    pass

def test_get_cliques_1():
    G = nx.complete_graph(5)
    G = nk.nxadapter.nx2nk(G)
    return [_ for _ in betti_numbers.get_cliques(G)]


def test_get_cliques_2():
    G = nx.petersen_graph()
    G = nk.nxadapter.nx2nk(G)
    print([_ for _ in betti_numbers.get_cliques(G)])

def test_get_cliques_3():
    G = nx.complete_graph(1)
    G = nk.nxadapter.nx2nk(G)
    print([_ for _ in betti_numbers.get_cliques(G)])

def test_get_colored_subgraphs_1():
    G = nx.complete_graph(5)
    G = nk.nxadapter.nx2nk(G)
    attr = ["red", "red", "red", "blue", "blue"]
    print([list(H.iterNodes()) for H in betti_numbers.get_colored_subgraphs(G, attr)])

def test_get_colored_subgraphs_2():
    pass

def test_get_colored_subgraphs_3():
    G = nx.complete_graph(5)
    G = nk.nxadapter.nx2nk(G)
    attr = ["red"]*5
    print([list(H.iterNodes()) for H in betti_numbers.get_colored_subgraphs(G, attr)])

def boundary_maps_1(cliques):
    return betti_numbers.boundary_maps(cliques)

def boundary_maps_2():
    pass

def boundary_maps_3():
    pass

def test_ranks_and_nullities_1():
    pass

def test_ranks_and_nullities_2():
    pass

def test_ranks_and_nullities_3():
    pass

if __name__ == "__main__":
    cliques = test_get_cliques_1()
    print(boundary_maps_1(cliques))
    