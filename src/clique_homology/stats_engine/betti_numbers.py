# ----------------------------------------------------------------------------------------------------------------

import networkit as nk
import numpy as np
import itertools
from typing import Mapping, Sequence

Vertex = int
Clique = tuple[Vertex, ...]
CliqueLike = Sequence[Vertex]
CliqueList = Sequence[CliqueLike]

# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
def get_max_clique_size(G:nk.Graph):
    """
    Get largest possible clique in a graph.
    """

    finder = nk.clique.MaximalCliques(G, maximumOnly=True)
    finder.run()
    cliques = finder.getCliques()
    
    if not cliques:
        return 0
    
    return len(cliques[0])


def get_cliques(G:nk.Graph):
    """
    Return a generator for cliques of a graph G.
    
    :param G: A colored graph.
    :type G: nk.Graph
    """

    # this part seems like it isn't the most efficient 
    # since we check so many duplicate cliques. 
    # Maybe have to look at different methods for this: write a custom bottom-up approach instead
    all_cliques = set()
    def collect_subcliques(C):
        for r in range(1, len(C)+1):
            for subset in itertools.combinations(C, r):
                all_cliques.add(tuple(sorted(subset)))

    # Find maximal cliques
    # the callback function runs when every maximal clique is found.
    clique_finder = nk.clique.MaximalCliques(G, maximumOnly=False, callback=collect_subcliques)
    clique_finder.run()

    all_cliques = sorted(list(all_cliques), key=len)

    # generate all the cliques
    for clique in all_cliques:
        yield clique

# ----------------------------------------------------------------------------------------------------------------

def get_colored_subgraphs(G:nk.Graph, node_colors:list[str]):
    """
    Return a generator for colored subgraphs of a graph G.
    
    :param G: A colored graph.
    :type G: nk.Graph
    :param node_attr: A dictionary mapping node IDs to their attribute (color).
    :type node_attr: list
    """
    # Group nodes by their attribute value - color:[nodes] key-value pairs.
    # Use graph node IDs from iterNodes() to support graphs with non-contiguous IDs.
    node_subsets: dict[str, list[int]] = {}
    for node, color in zip(G.iterNodes(), node_colors):
        if color not in node_subsets:
            node_subsets[color] = [node]
        else:
            node_subsets[color].append(node)
        
    for color in node_subsets.keys():
        yield nk.graphtools.subgraphFromNodes(G, node_subsets[color])

# ----------------------------------------------------------------------------------------------------------------

def boundary_maps(cliques: CliqueList) -> list[np.ndarray]:
    """
    Construct the boundary maps D_k given a complete list of cliques (simplicies).
    
    :param cliques: A complete list of cliques for a graph.
    :type cliques: Sequence[Sequence[int]]
    :return: A list of numpy arrays. These are the boundary maps D_k for each k.
    :rtype: list[np.ndarray]
    """
    canonical_cliques: list[Clique] = sorted(
        {tuple(sorted(clique)) for clique in cliques},
        key=lambda clique: (len(clique), clique),
    )

    def clique_order(cliques: Sequence[Clique]) -> list[dict[Clique, int]]:
        """
        Define an ordering for each clique with respect to the other cliques of their given size.
        This will be used to construct the boundary maps.
        
        :param cliques: Canonically sorted cliques.
        :type cliques: Sequence[tuple[int, ...]]
        :return: A list of dictionaries, one for each clique size.
        :rtype: list[dict[tuple[int, ...], int]]
        """

        if not cliques:
            return []

        max_clique_size = len(cliques[-1])
        result = [dict() for _ in range(max_clique_size)]

        for clique in cliques:
            size_index = len(clique) - 1
            result[size_index][clique] = len(result[size_index])

        return result

    def build_map(
        position_dict1: Mapping[Clique, int],
        position_dict2: Mapping[Clique, int],
    ) -> np.ndarray:
        """
        Construct a boundary map from position_dict1 to position_dict2.
        
        :param position_dict1: Dictionary of positions for (k-1)-cliques.
        :type position_dict1: Mapping[tuple[int, ...], int]
        :param position_dict2: Dictionary of positions for (k)-cliques.
        :type position_dict2: Mapping[tuple[int, ...], int]
        :return: Boundary map matrix.
        :rtype: np.ndarray
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
                    # hopefully this should never happen, with the way that we've written things
                    raise ValueError(f"Face {face} not found in position_dict1.")

        return M

    positions = clique_order(canonical_cliques)
    return [build_map(positions[k-1], positions[k]) for k in range(1, len(positions))]

# ----------------------------------------------------------------------------------------------------------------

def ranks_and_nullities(M:np.array) -> tuple:
    """
    Return the rank and nullity of a matrix over Z_2.

    :param: M: a matrix M({0, 1}).
    :type M: np.array
    :return: a tuple of rank and nullity.
    :rtype: tuple
    """

    def rank_Z2(M:np.array) -> int:
        """
        Return the rank of a matrix M({0, 1}).

        :param M: a matrix M({0, 1}).
        :type M: np.array
        :return: the rank of M.
        :rtype: int
        """
        M2 = M.copy()
        nrows, ncols = M2.shape
        rank = 0

        for j in range(ncols):
            if rank >= nrows:
                break

            # Find a pivot row for column j, looking only at rows >= rank
            # np.where returns a tuple, so we take [0] to get the array of indices
            pivot_candidates = np.where(M2[rank:, j] == 1)[0]

            if len(pivot_candidates) > 0:
                # Get the first available pivot (index is relative to the slice, so add rank)
                pivot_row = pivot_candidates[0] + rank

                # Swap the current row (rank) with the pivot row
                if pivot_row != rank:
                    M2[[rank, pivot_row]] = M2[[pivot_row, rank]]

                # Eliminate 1s in this column for all rows BELOW the pivot
                # We use (^) for addition modulo 2
                rows_to_eliminate = np.where(M2[rank+1:, j] == 1)[0] + (rank + 1)
                if len(rows_to_eliminate) > 0:
                    M2[rows_to_eliminate] ^= M2[rank]

                rank += 1
        
        return rank

    # utilize the rank-nullity theorem here.
    # returns (rank, nullity)
    rank = rank_Z2(M)
    nullity = M.shape[1] - rank
    return rank, nullity

# ----------------------------------------------------------------------------------------------------------------

def _validate_colors(
    G: nk.Graph,
    colors: list[str],
    allowed_colors: list[str] | None = None,
) -> None:
    if len(colors) != G.numberOfNodes():
        raise ValueError(
            f"Coloring length ({len(colors)}) must match number of graph nodes ({G.numberOfNodes()})."
        )

    if not all(isinstance(color, str) for color in colors):
        raise TypeError("All node colors must be strings.")

    if allowed_colors is None:
        return

    if not all(isinstance(color, str) for color in allowed_colors):
        raise TypeError("All allowed color values must be strings.")

    allowed_set = set(allowed_colors)
    if not allowed_set and colors:
        raise ValueError("Allowed color palette is empty but graph coloring is not.")

    invalid = sorted({color for color in colors if color not in allowed_set})
    if invalid:
        raise ValueError(
            "Found colors outside the allowed palette: "
            + ", ".join(invalid)
        )


def betti_numbers(
    G: nk.Graph,
    colors: list[str],
    method: str = "clique",
    allowed_colors: list[str] | None = None,
) -> np.ndarray: # type: ignore
    """
    Compute the Betti numbers of a colored graph. 

    If 'clique' method is specified (default), 
    build simplicial complex out of all monochromatic cliques for the entire graph.
    Returns a vector of Betti numbers.
    
    If 'subgraph' method is specified, 
    builds a distinct simplicial complex for each colored subgraph.
    Returns a matrix where each row vector gives the Betti numbers for a colored subgraph.

    If the resulting matrix of the 'subgraph' method is summed by rows, it will yield the result
    of the 'clique' method.
    
    :param G: A colored graph.
    :type G: Union[nk.Graph, nx.Graph]

    :param attr: A dictionary of node attributes, with attribute as value and node as key.
    :type attr: dict
    """
    if method not in ["subgraph", "clique"]:
        raise ValueError(f"Invalid method '{method}'. Expected 'subgraph', or 'clique'.")
    _validate_colors(G, colors, allowed_colors)

# note to self: a lot of this code can be refactored, and chunks can be combined between methods

    max_len = get_max_clique_size(G)

    if method == "subgraph":
        # in this case, compute all the betti numbers separately for each colored subgraph.
        # for this one, we may want to rework it so that it simply considers colors to partition into components instead
        betti_lists = []
        for subgraph in get_colored_subgraphs(G, colors):

            # get the maps for each subgraph
            cliques = [clique for clique in get_cliques(subgraph)]
            maps: list[np.ndarray] = boundary_maps(cliques)

            ranks, nullities = [], []
            for boundary_map in maps:
                # get the ranks and nullities for each map
                rank, nullity = ranks_and_nullities(boundary_map)
                ranks.append(rank)
                nullities.append(nullity)

            if maps:
                # prepend the number of nodes to nullities, append zero to ranks
                nullities = [maps[0].shape[0]] + nullities
                ranks.append(0)
                # compute the betti numbers    
                betti = [nullities[k] - ranks[k] for k in range(len(ranks))]
            else:
                # handle the edge cases
                if not cliques:
                    betti = []
                else:
                    betti = [len(cliques)]
            betti_lists.append(betti)
        
        # pad with zeros
        # a matrix of betti numbers
        padded_betti = [b + [0] * (max_len - len(b)) for b in betti_lists]
        if padded_betti:
            B = np.array(padded_betti)
        else:
            B = np.zeros((0, max_len), dtype=int)

        return B

    
    elif method == "clique":
        # the difference here is we compute the cliques, aggregate them, then compute the homology
        cliques = sorted([clique for H in get_colored_subgraphs(G, colors) for clique in get_cliques(H)], key=len)
        maps: list[np.ndarray] = boundary_maps(cliques)
        ranks, nullities = [], []

        for boundary_map in maps:
            # get the ranks and nullities for each map
            rank, nullity = ranks_and_nullities(boundary_map)
            ranks.append(rank)
            nullities.append(nullity)

        if maps:
            # prepend the number of nodes to nullities, append zero to ranks
            nullities = [maps[0].shape[0]] + nullities
            ranks.append(0)
            # compute the betti numbers    
            betti = [nullities[k] - ranks[k] for k in range(len(ranks))]
        else:
            if not cliques:
                betti = []
            else:
                betti = [len(cliques)]
        # pad with zeros to ensure consistency in size across permutations
        padded_betti = betti + [0] * (max_len - len(betti))
        # vector of betti numbers
        return np.array(padded_betti)
    
    else:
        pass

# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
