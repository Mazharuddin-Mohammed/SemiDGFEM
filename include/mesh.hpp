#pragma once
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include "device.hpp"

namespace simulator {
enum class MeshType { Structured, Unstructured };

class Mesh {
public:
    Mesh(const Device& device, MeshType type);
    ~Mesh() = default;

    // Copy constructor and assignment operator
    Mesh(const Mesh& other) = default;
    Mesh& operator=(const Mesh& other) = default;

    // Move constructor and assignment operator
    Mesh(Mesh&& other) noexcept = default;
    Mesh& operator=(Mesh&& other) noexcept = default;

    void generate_gmsh_mesh(const std::string& filename);

    // Safe accessors with bounds checking
    std::vector<double> get_grid_points_x() const { return grid_x_; }
    std::vector<double> get_grid_points_y() const { return grid_y_; }
    std::vector<std::vector<int>> get_elements() const { return elements_; }

    // Additional utility methods
    size_t get_num_nodes() const { return grid_x_.size(); }
    size_t get_num_elements() const { return elements_.size(); }
    bool is_valid() const;
    void validate() const;

    void refine(const std::vector<bool>& refine_flags);

    // Mesh quality metrics
    double get_min_element_quality() const;
    double get_max_element_size() const;

private:
    const Device& device_;
    MeshType type_;
    std::vector<double> grid_x_, grid_y_;
    std::vector<std::vector<int>> elements_;

    // Helper methods
    void generate_structured_mesh();
    void validate_element_connectivity() const;
    double compute_element_quality(const std::vector<int>& element) const;
};

// C interface for Cython
extern "C" {
    Mesh* create_mesh(Device* device, int mesh_type);
    void destroy_mesh(Mesh* mesh);
    void mesh_generate_gmsh(Mesh* mesh, const char* filename);
    int mesh_get_num_nodes(Mesh* mesh);
    int mesh_get_num_elements(Mesh* mesh);
    void mesh_get_grid_points_x(Mesh* mesh, double* points, int size);
    void mesh_get_grid_points_y(Mesh* mesh, double* points, int size);
    void mesh_get_elements(Mesh* mesh, int* elements, int num_elements, int nodes_per_element);
}

} // namespace simulator