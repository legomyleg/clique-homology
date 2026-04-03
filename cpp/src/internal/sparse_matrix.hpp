// rewrite the binary matrix class to sparse implementation
// vector<vector<int>> a vector of rows, where each row contains the column indices that are non-zero

// this is a sparse implmenetation of the binary matrix class
// basic functionality: one(i, j): insert a 1 at position (i, j)
//                      row_reduce(): row reduce the matrix, and return the rank and nullity as a pair

#include <vector>
#include <utility>
#include <stdexcept>
#include <limits>
#include <algorithm>

using std::vector, std::pair;

unsigned int max_uint = std::numeric_limits<unsigned int>::max();

class SparseBinaryMatrix {
    private:
        unsigned int rows;
        unsigned int cols;

        bool is_row_reduced = false;
        unsigned int rank;
        unsigned int nullity;
        
        // data is a vector of vectors, where the i-th vector are the column indices of the non-zero entries
        vector<vector<unsigned int>> data;

        // add two rows together: add row a to row b
        void add_rows(const std::vector<unsigned int>& a, vector<unsigned int>& b) {
            vector<unsigned int> result;
            result.reserve(a.size() + b.size()); 

            // This built-in function will XOR the sorted vectors in O(N) time
            std::set_symmetric_difference(
                a.begin(), a.end(), 
                b.begin(), b.end(), 
                std::back_inserter(result)
            );
        
            // Swap the newly built row into b
            b = std::move(result); 
        }

        // find the pivot column for a specific row
        unsigned int find_pivot(unsigned int row_index) const {
            if (data[row_index].empty()) {
                return max_uint; // Row is entirely zeros
            }
            // Because the row is sorted, the first element is always the leftmost pivot!
            return data[row_index].front();
        }

    public:
        // empty constructor
        SparseBinaryMatrix() : rows(0), cols(0), is_row_reduced(false), rank(0), nullity(0), data() {}
        
        // initialize empty sparse matrix of arbitrary dimension
        SparseBinaryMatrix(unsigned int r, unsigned int c) 
        : rows(r), cols(c), is_row_reduced(false), rank(0), nullity(c), data(r) {} // FIX: data size is exactly 'r'

        void one(unsigned int i, unsigned int j) {
            if (i >= rows || j >= cols) {throw std::out_of_range("Index out of bounds");}
            // insert a 1 at position (i, j) in the matrix
            data[i].push_back(j);
            is_row_reduced = false;
        }

        pair<unsigned int, unsigned int> row_reduce() {
            if (is_row_reduced) {
                return {rank, nullity};
            }

            // 1. SAFETY NET: Ensure every row is sorted and has no duplicate entries
            // set_symmetric_difference will fail silently if the vectors aren't sorted!
            for (auto& row : data) {
                std::sort(row.begin(), row.end());
                // Remove duplicates just in case .one() was called on the same spot twice
                row.erase(std::unique(row.begin(), row.end()), row.end());
            }

            // 2. Reset rank and nullity before calculation
            // Rank-Nullity Theorem: Rank + Nullity = Number of Columns
            rank = 0;
            nullity = cols;
            
            unsigned int curr_row = 0;

            // 3. Gaussian Elimination
            while (curr_row < rows) {
                unsigned int min_pivot = max_uint;
                unsigned int best_row = max_uint;

                // Find the row with the leftmost pivot
                for (unsigned int i = curr_row; i < rows; i++) {
                    unsigned int pivot = find_pivot(i);
                    if (pivot < min_pivot) {
                        min_pivot = pivot;
                        best_row = i;
                    }
                }

                // If min_pivot is still max_uint, all remaining rows are zeros. We are done!
                if (min_pivot == max_uint) {
                    break;
                }

                // Bring the best row to the current position
                if (curr_row != best_row) {
                    std::swap(data[curr_row], data[best_row]);
                }

                // We successfully locked in a pivot
                rank++;
                nullity--;

                // Zero out this column in all OTHER rows (producing Reduced Row Echelon Form)
                for (unsigned int i = 0; i < rows; i++) {
                    if (i != curr_row) {
                        // Binary search quickly checks if min_pivot exists in row i
                        if (std::binary_search(data[i].begin(), data[i].end(), min_pivot)) {
                            add_rows(data[curr_row], data[i]);
                        }
                    }
                }
                
                curr_row++;
            }

            is_row_reduced = true;
            return {rank, nullity};
        }
};