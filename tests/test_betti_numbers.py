import pytest
import networkx as nx
import networkit as nk
from random import choice, seed
from clique_homology import betti_numbers
import matplotlib.pyplot as plt

seed(122)

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
    G = nk.nxadapter.nx2nk(G)
    attr = [choice(["red", "blue"]) for _ in range(10)]
    print(betti_numbers.betti_numbers(G, attr))

def test_betti_numbers_2():
    G = nx.complete_graph(10)
    G = nk.nxadapter.nx2nk(G)

    attr = ["red"] * 5 + ["blue"] * 5
    print(betti_numbers.betti_numbers(G, attr))

def test_betti_numbers_3():
    G = nx.petersen_graph()
    G = nk.nxadapter.nx2nk(G)
    attr = ["red"] * 5 + ["blue"] * 5
    print(betti_numbers.betti_numbers(G, attr))

def test_betti_numbers_4():
    G = nx.Graph([(0, 1)])
    G = nk.nxadapter.nx2nk(G)
    attr = ["red", "red"]
    print(betti_numbers.betti_numbers(G, attr))

def test_betti_numbers_5():
    G = nx.petersen_graph()
    attr = ['red'] * 5 + ['blue'] * 5
    G2 = nk.nxadapter.nx2nk(G.copy())

    print(betti_numbers.betti_numbers(G2, attr, method = 'clique'))
    print(betti_numbers.betti_numbers(G2, attr, method = 'subgraph1'))
    print(betti_numbers.betti_numbers(G2, attr, method = 'subgraph2'))

    nx.draw(G, node_color = attr)
    plt.show()
    plt.clf()

def test_betti_numbers_6():
    G = nx.Graph([(0, 1)])
    G = nk.nxadapter.nx2nk(G)
    attr = ["red", "blue"]
    print(betti_numbers.betti_numbers(G, attr, method = 'clique'))


# --- Test helper functions individually ---

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
    G = nx.complete_graph(2)
    G = nk.nxadapter.nx2nk(G)
    attr = ["red", "red"]
    print([list(H.iterNodes()) for H in betti_numbers.get_colored_subgraphs(G, attr)])

def test_get_colored_subgraphs_3():
    G = nx.complete_graph(5)
    G = nk.nxadapter.nx2nk(G)
    attr = ["red"]*5
    print([list(H.iterNodes()) for H in betti_numbers.get_colored_subgraphs(G, attr)])

def boundary_maps_1():
    cliques = test_get_cliques_1()
    print(cliques)
    return betti_numbers.boundary_maps(cliques)

def boundary_maps_2():
    cliques = sorted([(0, 1, 2), (0, 1), (0, 2), (1, 2), (0,), (1,), (2,)], key=len)
    print(cliques)
    return betti_numbers.boundary_maps(cliques)

def boundary_maps_3():
    pass

def test_ranks_and_nullities_1():
    pass

def test_ranks_and_nullities_2():
    pass

def test_ranks_and_nullities_3():
    pass

def test_get_maximal_clique_size():
    G = nx.complete_graph(5)
    G = nk.nxadapter.nx2nk(G)
    print(betti_numbers.get_max_clique_size(G))

if __name__ == "__main__":
    #test_bett_numbers_1()
    #test_betti_numbers_2()
    #test_betti_numbers_3()
    #test_betti_numbers_4()
    #test_get_colored_subgraphs_2()
    #test_betti_numbers_5()
    test_betti_numbers_6()
    #test_get_maximal_clique_size()