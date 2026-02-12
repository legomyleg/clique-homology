import random

import networkit as nk
import networkx as nx
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    random.seed(1337)
    np.random.seed(1337)


def nx_to_nk(graph: nx.Graph) -> nk.Graph:
    return nk.nxadapter.nx2nk(graph)
