# -------------------------------------------------------------------------------------------

import networkit as nk
import numpy as np
import networkx as nx

# -------------------------------------------------------------------------------------------

def parse_graph_input(G, node_attr=None):
    """
    Handle multiple different formats for G. Mostly for testing purposes.
    
    :param G: An undirected networkx.Graph object or networkit.Graph object.
    :type G: Union[nx.Graph, nk.Graph]
    :param node_attr: A NodeAttribute networkit class or dictionary of node attributes.
    :type node_attr: Union[nk.NodeAttribute, dict]
    """

    if type(G) == nk.Graph:
        pass
    elif type(G) == nx.Graph:
        G = nk.nxadapter(G)
    else:
        raise ValueError(f"Graph must be either networkx.Graph or networkit.Graph. Got {type(G)}")

def get_cliques(G:nk.Graph):
    """
    Return a generator for cliques of a graph G.
    
    :param G: A colored graph.
    :type G: nk.Graph
    """
    pass

# -------------------------------------------------------------------------------------------

def get_colored_subgraphs(G:nk.Graph, node_attr=nk.NodeAttribute):
    """
    Return a generator for colored subgraphs of a graph G.
    
    :param G: A colored graph.
    :type G: nk.Graph
    :param attr: The attribute to group nodes by.
    :type attr: nk.NodeAttribute
    """
    pass

# -------------------------------------------------------------------------------------------

def boundary_maps(cliques:list) -> list:
    """
    Construct the boundary maps D_k given a complete list of cliques (simplicies).
    
    :param cliques: A complete list of cliques for a graph. Cliques should be lists of vertices.
    :type cliques: list
    :return: a tuple of numpy arrays. These are the boundary maps D_k for each k.
    :rtype: tuple
    """
    def clique_order(cliques:list) -> list:
        """
        Define an ordering for each clique with respect to the other cliques of their given size.
        This will be used to construct the boundary maps.
        
        :param cliques: Description
        :type cliques: list
        :return: A tuple of dictionaries, one for each size of clique: tuple(dict(tuple:int), ...)
        :rtype: tuple
        """
        pass

    def build_map(position_dict1, position_dict2) -> np.array:
        """
        Construct a boundary map from position_dict1 to position_dict2.
        
        :param position_dict1: Dictionary of positions for (k-1)-cliques.
        :type position_dict1: dict
        :param position_dict2: Dictionary of positions for (k)-cliques.
        :type position_dict2: dict
        :return: Description
        :rtype: Any
        """

        # We are mapping from the (k-1)-cliques (nrow) to the k-cliques (ncol) 
        M = np.zeros((len(position_dict1), len(position_dict2)), dtype=int)
        for k2, v2 in position_dict2.items():
            # Iterate over all faces of the simplex k2
            for i in range(len(k2)):
                # Create face by removing the i-th vertex w/ tuple slicing
                face = k2[:i] + k2[i+1:]
                if face in position_dict1:
                    M[position_dict1[face], v2] = 1
                else:
                    raise ValueError(f"Face {face} not found in position_dict1.")
        
        return M

    positions = clique_order(cliques)
    return [build_map(positions[k-1], positions[k]) for k in range(1, len(positions))]

# -------------------------------------------------------------------------------------------

def ranks_and_nullities(M:np.array) -> tuple:
    """
    Return the rank and nullity of a matrix over Z_2.

    :param: M: a matrix M({0, 1}).
    :type M: np.array
    :return: a tuple of rank and nullity.
    :rtype: tuple
    """
    def row_reduce_Z2(M:np.array) -> np.array:
        """
        Row reduce a matrix with components in {0,1}.

        :param M: a matrix M({0, 1}).
        :type M: np.array
        :return: the row reduced matrix.
        :rtype: np.array
        """
        pass

    def rank_Z2(M2:np.array) -> int:
        """
        Return the rank of a matrix M2({0, 1}).

        :param M2: a matrix M2({0, 1}).
        :type M2: np.array
        :return: the rank of M2.
        :rtype: int
        """
        pass

    M2 = row_reduce_Z2(M)
    # utilize the rank-nullity theorem here.
    # returns (rank, nullity)
    return rank_Z2(M2), M2.shape[1] - rank_Z2(M2)

# -------------------------------------------------------------------------------------------

def betti_numbers(G:nk.Graph, attr:str="color", method:str="clique") -> np.array:
    """
    Compute the Betti numbers of a colored graph. 

    If 'clique' method is specified (default), 
    build simplicial complex out of all monochromatic cliques for the entire graph.
    Returns a vector of Betti numbers.
    
    If 'subgraph1' method is specified, 
    builds a distinct simplicial complex for each colored subgraph.
    Returns a matrix where each row vector gives the Betti numbers for a colored subgraph.

    If 'subgraph2' method is specified,
    partitions the graph by disregarding edges between nodes of different attributes, then computes
    the homology for this new graph.
    
    :param G: A colored graph.
    :type G: nk.Graph

    :param attr: The attribute to group nodes by. Defaults to color.
    :type attr: str
    """
    if method not in ["subgraph1", "subgraph2", "clique"]:
        raise ValueError(f"Invalid method '{method}'. Expected 'subgraph1', 'subgraph2', or 'clique'.")
    
    if type(G) != nk.Graph:
        raise ValueError(f"{G} is not a networkit.Graph graph.")

# note to self: a lot of this code can be refactored, and chunks can be combined between methods

    if method.startswith("subgraph"):
        # in this case, compute all the betti numbers separately for each colored subgraph.
        # for this one, we may want to rework it so that it simply considers colors to partition into components instead
        betti_lists = []
        for subgraph in get_colored_subgraphs(G):

            # get the maps for each subgraph
            cliques = [clique for clique in get_cliques(subgraph)]
            maps = boundary_maps(cliques)

            ranks, nullities = [], []
            for boundary_map in maps:
                # get the ranks and nullities for each map
                rank, nullity = ranks_and_nullities(boundary_map)
                ranks.append(rank)
                nullities.append(nullity)

            # compute the betti numbers    
            betti = [nullities[k] - ranks[k+1] for k in range(len(ranks)-1)]
            betti_lists.append(betti)

        # a matrix of betti numbers    
        B = np.array(betti_lists)

        if method == "subgraph1":
            return B
        elif method == "subgraph2":
            # aggregate Betti numbers
            return np.sum(B, axis=0)

    
    elif method == "clique":
        # the difference here is we compute the cliques, aggregate them, then compute the homology
        cliques = sorted([clique for H in get_colored_subgraphs(G) for clique in get_cliques(H)], key=len, reverse=True)
        maps = boundary_maps(cliques)
        ranks, nullities = [], []
        for boundary_map in maps:
            # get the ranks and nullities for each map
            rank, nullity = ranks_and_nullities(boundary_map)
            ranks.append(rank)
            nullities.append(nullity)

        # compute the betti numbers    
        betti = [nullities[k] - ranks[k+1] for k in range(len(ranks)-1)]

        # returns a matrix of betti numbers
        return np.array(betti)
    
    else:
        pass

if __name__ == "__main__":
    pass
