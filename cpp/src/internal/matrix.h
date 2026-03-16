#include <vector>
#include <cstdint>
#include <stdexcept>
#include <bit>
#include <utility>

using std::vector, std::pair, std::__countr_zero;

class Bin_Matrix {

    // matrix consisting of elements from the binary field {0, 1}
    private:
        size_t rows; // number of rows 
        size_t cols; // number of columns
        size_t col_ints; // how many integers do we need per row?

        vector<uint64_t> data; // 1D vector containing 64-bit integers

        // these are computed after row reduction
        bool row_reduced = false;
        size_t rank = 0;
        size_t nullity = cols;
       
        void add_rows(size_t target, size_t source) {
            // add source row to target row in place
            size_t target_start = target * col_ints;
            size_t source_start = source * col_ints;

            // XOR every pair of integers in the row
            for (size_t k = 0; k < col_ints; k++) {
                data[target_start + k] ^= data[source_start + k];
            }
        }

        void swap_rows(size_t row1, size_t row2) {
            if (row1 == row2) {return;}
            // swap the values in each row
            size_t start1 = row1 * col_ints;
            size_t start2 = row2 * col_ints;

            for (size_t k = 0; k < col_ints; k++) {
                std::swap(data[start1 + k], data[start2 + k]);
            }
        }

        size_t find_pivot(size_t i) const {
            // return the position of the pivot at row i
            auto start = i*col_ints;

            // loop through the integers in the given row
            for (size_t k = 0; k < col_ints; k++) {
                if (data[start + k] != 0) {
                    // return the number of trailing zeros plus the column that we started looking from
                    return (k*64) + (__countr_zero(data[start + k]));
                }
            }

            // the row is a zero row
            return -1;
        }

// ---------------------------------------------------------------------------------------------------------

    public:
        // constructor initializes to zero matrix of given dimensions
        Bin_Matrix(size_t r, size_t c) 
        : rows(r), 
          cols(c), 
          col_ints((c+63)/64),
          data((c+63)/64*r, 0) {}

        // return value at (i, j) 
        bool operator()(size_t i, size_t j) const { 
            if (i >= rows || j >= cols) {throw std::out_of_range("i,j out of bounds.");}

            // locate the bit corresponding to the i, j entry
            size_t chunk_i = (i*col_ints) + (j/64);
            size_t bit_offset = j % 64;

            // shift the bits, AND it against 1
            bool value = (data[chunk_i] >> bit_offset) & 1ULL;
            return value;

        }

        // make the value at (i, j) a one
        void one(size_t i, size_t j) {
            if (i >= rows || j >= cols) {throw std::out_of_range("i,j out of bounds.");}

            // useful for constructing the boundary maps
            size_t chunk_i = (i*col_ints) + (j/64);
            size_t bit_offset = j % 64;

            // equal to 1 only at the same position as bit_offset
            uint64_t mask = 1ULL << bit_offset;

            // make this bit a 1
            data[chunk_i] |= mask;
        }

        // make the value at (i, j) a zero
        void zero(size_t i, size_t j) {
            if (i >= rows || j >= cols) {throw std::out_of_range("i,j out of bounds.");}

            size_t chunk_i = (i*col_ints) + (j/64);
            size_t bit_offset = j % 64;

            // equal to 1 only at the same position as bit_offset
            uint64_t mask = 1ULL << bit_offset;

            // make this bit a zero
            // ~ flips the mask
            data[chunk_i] &= ~mask;
        }

        pair<size_t, size_t> row_reduce() {
            // don't recompute if unnecessary
            if (row_reduced) {return {rank, nullity};}

            size_t curr_row = 0;

            while (curr_row < rows) {
                size_t min_pivot = -1;
                size_t best_row = -1;

                // go looking for the best row to use (i.e. leftmost pivot)
                for (size_t i = curr_row; i < rows; i++) {
                    size_t pivot = find_pivot(i);
                    if (pivot < min_pivot) {
                        min_pivot = pivot;
                        best_row = i;
                    }
                }

                if (min_pivot == SIZE_MAX) {
                    // there are no valid pivot rows
                    break;
                }

                // we've found our pivot row
                swap_rows(curr_row, best_row);
                rank++;
                nullity--;

                // zero out all the other rows in this column
                for (size_t i = 0; i < rows; i++) {
                    if (i != curr_row && (*this)(i, min_pivot) == 1) {
                        add_rows(i, curr_row);
                    }
                }

                // move to next row to check
                curr_row++;
            }

            return {rank, nullity};

        }

};