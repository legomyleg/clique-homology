#include <vector>
#include <cstdint>
#include <stdexcept>
#include <bit>
#include <utility>
#include <iostream>
#include <string>

using std::vector, std::pair, 
std::__countr_zero, std::cout, std::endl, std::string;

// this matrix class stores the columns as 64-bit integers for fast bit-wise row operations
// initialize matrix as BinaryMatrix(nrows, ncols). Update the (i,j)th entry with .zero(i,j) or .one(i,j) methods
// Call .row_reduce() to row reduce the matrix. Returns [rank, nullity] pair. Can be called multiple times to retrieve
// ranks and nullities
// there is matrix operations support for the *, +, and == operators

class BinaryMatrix {

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
            // fast look up by looking at trailing zeros
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
        BinaryMatrix(size_t r, size_t c) 
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

        // perform matrix addition
        BinaryMatrix operator+(const BinaryMatrix& other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::invalid_argument("matrices must have matching dimensions.");
            }

            // iterate through each chunk and XOR them
            // make a copy of the matrix
            BinaryMatrix result = *this;
            for (size_t i = 0; i < data.size(); i++) {
                // XOR adds component-wise
                result.data[i] ^= other.data[i];
            }

            return result;
        }

        // perform matrix multiplication
        BinaryMatrix operator*(const BinaryMatrix& other) const {
            // A's columns MUST equal B's rows
            if (cols != other.rows) {
                throw std::invalid_argument("Inner matrix dimensions are unequal");
            }
        
            // initialize resulting matrix product
            BinaryMatrix result(rows, other.cols);
        
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {

                    // If A(i, j) is 1, we XOR the entire j-th row of B into the i-th row of Result
                    if ((*this)(i, j) == 1) {

                        size_t result_start = i * result.col_ints;
                        size_t b_start = j * other.col_ints;
                    
                        // Fast chunk-by-chunk XOR
                        for (size_t chunk = 0; chunk < result.col_ints; chunk++) {
                            result.data[result_start + chunk] ^= other.data[b_start + chunk];
                        }
                    }
                }
            }
            return result;
        }

        bool operator==(const BinaryMatrix& other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::invalid_argument("Inner matrix dimensions are unequal");
            }

            // test for equality component-wise
            for (size_t i = 0; i < data.size(); i++) {
                if (data[i] != other.data[i]) {
                    return false;
                }
            }

            return true;
        }

        // allow 1 by 1 matrices to be compared to booleans
        bool operator==(const bool& other) const {
            if (rows != 1 || cols != 1) {throw std::invalid_argument("Inner matrix dimensions are unequal");}
            
            return (*this)(0, 0) == other;
        }

        // access number of rows and columns, if needed
        size_t nrows() const {return rows;}
        size_t ncols() const {return cols;}

        BinaryMatrix row_i (size_t i) const {
            // return row i as a Binary Matrix
            BinaryMatrix output(1, cols);

            auto start = col_ints * i;
            auto end = col_ints * i + col_ints;
            output.data = vector<uint64_t>(data.begin() + start, data.begin() + end);

            return output;

        }

        void add_row(vector<bool>& new_row) {
            // add a new row to the matrix
            if (new_row.size() != cols) {throw std::invalid_argument("New row dimensions does not match.");}
            for (size_t k = 0; k < col_ints; k++) {
                // add a new row to fill
                data.push_back(0);
            }

            for (size_t j = 0; j < new_row.size(); j++) {
                // update the bits
                if (new_row[j]) {(*this)(data.size()-1, j);} 
            }
        }

        // make the value at (i, j) a one
        // useful for constructing the boundary maps
        void one(size_t i, size_t j) {
            if (i >= rows || j >= cols) {throw std::out_of_range("i,j out of bounds.");}

            // matrix is changed, it isn't row_reduced anymore
            row_reduced = false;

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

            // matrix is changed, it isn't row reduced anymore
            row_reduced = false;

            size_t chunk_i = (i*col_ints) + (j/64);
            size_t bit_offset = j % 64;

            // equal to 1 only at the same position as bit_offset
            uint64_t mask = 1ULL << bit_offset;

            // make this bit a zero
            // ~ flips the mask
            data[chunk_i] &= ~mask;
        }

        pair<size_t, size_t> row_reduce() {
            // row reduce the matrix to RREF, compute and return rank and nullity.


            // don't recompute if unnecessary
            // row_reduced is false by default and will reset to false whenever matrix is manipulated
            // you can row reduce, then modify the matrix, then row reduce again, recomputing rank and nullity
            if (row_reduced) {
                return {rank, nullity};
            } else {
                rank = 0;
                nullity = cols;
            }

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

            row_reduced = true;
            return {rank, nullity};

        }

        void print(bool verbose) {
            // print the matrix to the console
            if (verbose) {
                cout << "BinaryMatrix([" << endl;
                for (size_t i = 0; i < rows; i++) {

                    cout << "  "; // offset

                    for (size_t j = 0; j < cols; j++) {

                        // print the row
                        cout << (*this)(i, j);

                        if (i != rows-1 || j != cols-1) {
                            cout << " ";
                        }
                    }

                    cout << endl;
                }
                cout << "])" << endl;
                cout << "Rows: " << rows << ", Columns: " << cols << "." << endl;
            } else {
                // not verbose, just print numeric values
                for (size_t i = 0; i < rows; i++) {

                    for (size_t j = 0; j < cols; j++) {

                        // print the row
                        cout << (*this)(i, j);

                        if (i != rows-1 || j != cols-1) {
                            cout << " ";
                        }
                    }

                    cout << endl;
            }
        }
    }

};

// allow booleans to be compared to 1 by 1 matrices from the left
inline bool operator==(const bool& lhs, const BinaryMatrix& rhs) {
    // Reuse the existing member operator by flipping the operands
    return rhs == lhs;
}