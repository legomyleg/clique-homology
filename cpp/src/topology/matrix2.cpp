// rewrite the binary matrix class to sparse implementation
// vector<vector<int>> a vector of rows, where each row contains the column indices that are non-zero

#include <vector>
#include <utility>
#include <stdexcept>
#include <limits>

using std::vector, std::pair;


unsigned int max_uint = std::numeric_limits<unsigned int>::max();

class Sparse_Binary_Matrix {
    private:
        vector<vector<unsigned int>> data;
        unsigned int rows;
        unsigned int cols;

        // these are computed after the matrix is row-reduced
        bool is_row_reduced = false;
        unsigned int rank = 0;
        unsigned int nullity = cols;

    public:
        // initialize empty sparse matrix of arbitrary dimension
        Sparse_Binary_Matrix(unsigned int r, unsigned int c) : rows(r), cols(c) {}

        // initialize sparse matrix based on sparse data input
        Sparse_Binary_Matrix(vector<vector<unsigned int>> new_data, unsigned int r, unsigned int c) 
        : rows(r),
          cols(c),
          data(new_data) {}

        void one(size_t i, size_t j) {
            if (i >= data.size()) {throw std::out_of_range("i is out of bounds");}
            // insert a 1 at position (i, j) in the matrix
            data[i].push_back(j);
            is_row_reduced = false;
        }

        pair<unsigned int, unsigned int> row_reduce() {
            if (is_row_reduced) {return {rank, nullity};}
            else {rank = 0; nullity = cols;}

            unsigned int curr_row = 0;

            // do row-reduction algorithm

            return {rank, nullity};

        }


};