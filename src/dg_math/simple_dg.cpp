/**
 * Simple DG linear algebra implementation
 * Provides basic matrix and vector operations for DG methods
 */

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace SimpleDG {

// Simple matrix class for DG methods
class Matrix {
public:
    std::vector<std::vector<double>> data;
    size_t rows, cols;
    
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }
    
    double& operator()(size_t i, size_t j) {
        return data[i][j];
    }
    
    const double& operator()(size_t i, size_t j) const {
        return data[i][j];
    }
    
    void zero() {
        for (auto& row : data) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }
};

// Simple vector class for DG methods
class Vector {
public:
    std::vector<double> data;
    size_t size;
    
    Vector(size_t s) : size(s) {
        data.resize(size, 0.0);
    }
    
    double& operator[](size_t i) {
        return data[i];
    }
    
    const double& operator[](size_t i) const {
        return data[i];
    }
    
    void zero() {
        std::fill(data.begin(), data.end(), 0.0);
    }
};

// Simple linear solver using Gaussian elimination
Vector solve_system(const Matrix& A, const Vector& b) {
    if (A.rows != A.cols || A.rows != b.size) {
        throw std::runtime_error("Matrix dimensions mismatch");
    }
    
    size_t n = A.rows;
    Vector x(n);
    
    // Create augmented matrix
    std::vector<std::vector<double>> aug(n, std::vector<double>(n + 1));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = A(i, j);
        }
        aug[i][n] = b[i];
    }
    
    // Forward elimination
    for (size_t k = 0; k < n; ++k) {
        // Find pivot
        size_t max_row = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(aug[i][k]) > std::abs(aug[max_row][k])) {
                max_row = i;
            }
        }
        
        // Swap rows
        if (max_row != k) {
            std::swap(aug[k], aug[max_row]);
        }
        
        // Check for singular matrix
        if (std::abs(aug[k][k]) < 1e-14) {
            throw std::runtime_error("Singular matrix");
        }
        
        // Eliminate column
        for (size_t i = k + 1; i < n; ++i) {
            double factor = aug[i][k] / aug[k][k];
            for (size_t j = k; j <= n; ++j) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }
    
    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        x[i] = aug[i][n];
        for (size_t j = i + 1; j < n; ++j) {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    
    return x;
}

// Matrix-vector multiplication
Vector multiply(const Matrix& A, const Vector& x) {
    if (A.cols != x.size) {
        throw std::runtime_error("Matrix-vector dimension mismatch");
    }
    
    Vector result(A.rows);
    for (size_t i = 0; i < A.rows; ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < A.cols; ++j) {
            result[i] += A(i, j) * x[j];
        }
    }
    
    return result;
}

// Vector addition
Vector add(const Vector& a, const Vector& b) {
    if (a.size != b.size) {
        throw std::runtime_error("Vector dimension mismatch");
    }
    
    Vector result(a.size);
    for (size_t i = 0; i < a.size; ++i) {
        result[i] = a[i] + b[i];
    }
    
    return result;
}

// Vector scaling
Vector scale(const Vector& v, double factor) {
    Vector result(v.size);
    for (size_t i = 0; i < v.size; ++i) {
        result[i] = v[i] * factor;
    }
    
    return result;
}

// Vector norm
double norm(const Vector& v) {
    double sum = 0.0;
    for (size_t i = 0; i < v.size; ++i) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

} // namespace SimpleDG
