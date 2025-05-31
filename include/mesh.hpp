#pragma once
#include <vector>
#include <string>
#include "device.hpp"

namespace simulator {
enum class MeshType { Structured, Unstructured };

class Mesh {
public:
    Mesh(const Device& device, MeshType type);
    void generate_gmsh_mesh(const std::string& filename);
    std::vector<double> get_grid_points_x() const { return grid_x_; }
    std::vector<double> get_grid_points_y() const { return grid_y_; }
    std::vector<std::vector<int>> get_elements() const { return elements_; }
    void refine(const std::vector<bool>& refine_flags);
private:
    const Device& device_;
    MeshType type_;
    std::vector<double> grid_x_, grid_y_;
    std::vector<std::vector<int>> elements_;
};
} // namespace simulator