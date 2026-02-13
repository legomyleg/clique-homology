# %% [markdown]
# # Generate Graph Models for Statistical Testing

# %%
import networkx as nx
import networkit as nk
import numpy as np
from sklearn.model_selection import ParameterSampler

np.random.seed(122)


# %% [markdown]
# ## SBM Growth Model (by Color) (Slow)

# %%
def generate_SBM_growth(n:int, P:np.ndarray) -> nx.Graph:
    """
    Generate iterative SBM of size n given a set of vertex edge pair probabilities
    
    :param n: Number of nodes
    :param P: n*n edge probability matrix
    """

    # add the first node
    G = nx.Graph()
    G.add_node(0)

    for i in range(1, n):

        # add the node to the graph, then check it against the existing nodes to see if we will add an edge or not
        G.add_node(i)

        for j in range(0, i):
            # choose rand num U(0, 1)
            runif = np.random.random()

            if runif <= P[j, i]:
                # there is an edge
                G.add_edge(i, j)

    return G

# %%
def generate_prob_matrix(n:int, c:int, x_sim:float, x_diff:float, sigma_eps:float):

    # possible color choices
    color_choices = np.array(range(c))
    # assign nodes a color uniformly at random
    color_arr = np.random.choice(color_choices, size=n, replace=True)

    # compute the error terms
    Epsilon = np.random.normal(0, sigma_eps, (n, n))
    Epsilon = np.triu(Epsilon)

    # compute the fixed similarities
    X = np.array([[x_sim if color_arr[i]==color_arr[j] else x_diff for i in range(n)] for j in range(n)])
    X = np.triu(X)

    # use the logit function to scale the result down to a valid probability
    logit = lambda t: 1 / (1 + np.exp(-t))
    return logit(X + Epsilon), list(color_arr)

# %%
def placeholder_func(G:nk.Graph, colors:list) -> float:
    # for now, just return a random p value
    return np.random.random()

def simulate(x_sim, x_diff, n, c, sigma_eps):
    P, colors = generate_prob_matrix(n, c, x_sim, x_diff, sigma_eps)
    G = generate_SBM_growth(n, P)
    pval = placeholder_func(nk.nxadapter.nx2nk(G), colors)

    return pval
    

# %% [markdown]
# ## Parameter Grid

# %%
param = {
    'x_sim': range(1, 11),
    'x_diff': range(1, 11),
    'n': range(10, 300, 20),
    'c': range(2, 10),
    'sigma_eps': range(0, 10)
    }

grid = ParameterSampler(param, n_iter=100, random_state=42)

# %% [markdown]
# ## Simulate

# %%
n_validations = 20
alpha = 0.05

results = {}

for params in grid:
    param_key = tuple(sorted(params.items()))

    # we will store the results of the tests themselves for each iteration, 
    # and then calculate the proportion that were correct
    results[param_key] = [0, []]
    
    for _ in range(n_validations):
        pval = simulate(**params)
        results[param_key][1].append(pval)
    
    results[param_key][0] = sum([elem < alpha for elem in results[param_key][1]]) / 20

# results will be the proportion of graphs correctly identified as significant
for key, value in results.items():
    print(f"Params: {key}; result: {value[0]}")


