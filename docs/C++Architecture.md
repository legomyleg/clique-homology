# Architecture & Data Pipeline

## Module Map
The codebase is divided into five primary modules, each handling a distinct domain of the computation:

* **`data`**: Handles input interpretation and parsing.
* **`main`**: The core execution driver (`main.cpp`).
* **`graph`**: Manages graph theory algorithms and transformations.
  * *Responsibilities:* Bron-Kerbosch (`BK`) clique finding, subgraph extraction, graph permutations, sorting, and `max_clique_size` calculation.
* **`topology`**: Manages algebraic topology computations.
  * *Responsibilities:* Matrix operations and boundary map construction.
* **`stats`**: Handles statistical analysis and significance testing.
  * *Responsibilities:* Permutation tests, single p-value (`pval`) calculation, and p-value vector (`pval_vec`) calculation.

---

## Pipeline Flow

**User Inputs:** `Graph_data`, `color_list`, `pval_option`, `confidence_level`, `N` (number of permutations).

### Phase 1: Observed Data Computation
**1. Input Parsing:** Parse the user input into a `Graph()` object, which encapsulates the adjacency matrix (`A`) and the `color_vector`.

**2. Subgraph Clique Extraction:** Iterate through the colored subgraphs (`Graph.colored_subgraphs()`). For each subgraph `G`, run the Bron-Kerbosch algorithm (`G.BK()`) to extract all cliques.

**3. Clique Mapping:** Organize the extracted cliques into an ordered map, keyed by the size of the cliques.

**4. Boundary Map Construction:** Pass the ordered map of cliques into the boundary map constructor.

**5. Matrix Generation:** Output the constructed boundary maps as a sequence (`vector<Matrix> M`).

**6. Rank & Nullity Calculation:** Iterate through each matrix `m` in `M`. Perform row reduction (`m.rr()`) to compute the rank and nullity. Store these in respective `ranks` and `nullities` vectors.

**7. Betti Number Calculation:** For each dimension `i`, compute the observed Betti number using the formula `betti[i] = nullities[i] - ranks[i]`. This yields the final `obs_betti` vector.

### Phase 2: Null Distribution Generation
**8. Graph Permutation:** Generate a permuted graph (`Graph().permuted_graph()`). Repeat Steps 1 through 7 on this permuted graph. Do this `N` times.

**9. Betti Matrix Assembly:** Aggregate the results of the `N` permutations into a `betti_matrix`. This matrix has `N` rows (one for each permutation) and `max_clique_size` columns.

### Phase 3: Statistical Analysis
Depending on the user's `pval_option`, the pipeline branches into one of two statistical evaluations:

**Path A: Single P-Value**
* **10.** Pass the `betti_matrix` and `obs_betti` into the `pval` function.
* **11.** Output a tuple containing: the aggregate `float` p-value, the `obs_betti` vector, and the null distribution.

**Path B: P-Value Vector**
* **12.** Pass the `betti_matrix` and `obs_betti` into the `pval_vec` function.
* **13.** Output a tuple containing: a vector of `float` p-values (one for each dimension) and a vector of confidence intervals.