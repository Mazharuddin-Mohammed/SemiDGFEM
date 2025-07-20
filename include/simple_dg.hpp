/**
 * Simple DG linear algebra header
 * Provides basic matrix and vector operations for DG methods
 */

#pragma once

#include <vector>
#include <stdexcept>

namespace SimpleDG {

// Simple matrix class for DG methods
class Matrix {
public:
    std::vector<std::vector<double>> data;
    size_t rows, cols;
    
    Matrix(size_t r, size_t c);
    
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;
    
    void zero();
};

// Simple vector class for DG methods
class Vector {
public:
    std::vector<double> data;
    size_t size;
    
    Vector(size_t s);
    
    double& operator[](size_t i);
    const double& operator[](size_t i) const;
    
    void zero();
};

// Linear algebra operations
Vector solve_system(const Matrix& A, const Vector& b);
Vector multiply(const Matrix& A, const Vector& x);
Vector add(const Vector& a, const Vector& b);
Vector scale(const Vector& v, double factor);
double norm(const Vector& v);

} // namespace SimpleDG
