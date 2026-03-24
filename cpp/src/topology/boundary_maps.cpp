#include <vector>
#include "matrix2.hpp"

using std::vector;

// size_map is vector of 
//  vector of same size cliques
//      a clique is a vector of integers
vector<Sparse_Binary_Matrix> construct_boundary_maps(const <vector<vector<vector<int>>>>& size_map) {
    vector<Sparse_Binary_Matrix> boundary_maps;
    // for matrices of each size, construct the boundary map
    for (unsigned int k = 1; k < size_map.size()-1; k++) {
        vector<vector<unsigned int>> new_data;
        auto& curr_cliques = size_map[k];
        auto& next_cliques = size_map[k+1];

        for (auto& next_clique : next_cliques) {
            
        }


    }

    return boundary_maps;
};