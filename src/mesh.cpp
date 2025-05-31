#include "mesh.hpp"
#include <gmsh.h>
#include <cmath>

namespace simulator {
Mesh::Mesh(const Device& device, MeshType type) : device_(device), type_(type) {
    auto extents = device_.get_extents();
    double Lx = extents[0], Ly = extents[1];
    int nx = 50, ny = 25; // Default grid
    if (type_ == MeshType::Structured) {
        for (int i = 0; i <= nx; ++i) {
            for (int j = 0; j <= ny; ++j) {
                grid_x_.push_back(i * Lx / nx);
                grid_y_.push_back(j * Ly / ny);
            }
        }
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int n1 = i * (ny + 1) + j;
                int n2 = n1 + 1;
                int n3 = (i + 1) * (ny + 1) + j;
                int n4 = n3 + 1;
                elements_.push_back({n1, n3, n2});
                elements_.push_back({n2, n3, n4});
            }
        }
    }
}

void Mesh::generate_gmsh_mesh(const std::string& filename) {
    if (type_ != MeshType::Unstructured) return;
    gmsh::initialize();
    gmsh::model::add("device");
    auto extents = device_.get_extents();
    double Lx = extents[0], Ly = extents[1];
    gmsh::model::geo::addPoint(0, 0, 0, Lx / 50);
    gmsh::model::geo::addPoint(Lx, 0, 0, Lx / 50);
    gmsh::model::geo::addPoint(Lx, Ly, 0, Lx / 50);
    gmsh::model::geo::addPoint(0, Ly, 0, Lx / 50);
    gmsh::model::geo::addLine(1, 2);
    gmsh::model::geo::addLine(2, 3);
    gmsh::model::geo::addLine(3, 4);
    gmsh::model::geo::addLine(4, 1);
    gmsh::model::geo::addCurveLoop({1, 2, 3, 4});
    gmsh::model::geo::addPlaneSurface({1});
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::generate(2);
    gmsh::write(filename);

    grid_x_.clear();
    grid_y_.clear();
    elements_.clear();
    std::vector<std::size_t> nodeTags;
    std::vector<double> coord, paramCoord;
    gmsh::model::mesh::getNodes(nodeTags, coord, paramCoord);
    for (size_t i = 0; i < nodeTags.size(); ++i) {
        grid_x_.push_back(coord[i * 3]);
        grid_y_.push_back(coord[i * 3 + 1]);
    }
    std::vector<int> elemTypes;
    std::vector<std::vector<std::size_t>> elemTags, elemNodeTags;
    gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, 2);
    for (const auto& nodes : elemNodeTags[0]) {
        elements_.push_back({static_cast<int>(nodes[0] - 1), static_cast<int>(nodes[1] - 1), static_cast<int>(nodes[2] - 1)});
    }
    gmsh::finalize();
}

void Mesh::refine(const std::vector<bool>& refine_flags) {
    if (type_ == MeshType::Unstructured) {
        generate_gmsh_mesh("device_refined.msh");
        return;
    }
    std::vector<double> new_grid_x, new_grid_y;
    std::vector<std::vector<int>> new_elements;
    int node_offset = grid_x_.size();
    for (size_t e = 0; e < elements_.size(); ++e) {
        if (!refine_flags[e]) {
            new_elements.push_back(elements_[e]);
            continue;
        }
        int i1 = elements_[e][0], i2 = elements_[e][1], i3 = elements_[e][2];
        double x1 = grid_x_[i1], y1 = grid_y_[i1];
        double x2 = grid_x_[i2], y2 = grid_y_[i2];
        double x3 = grid_x_[i3], y3 = grid_y_[i3];
        double x12 = (x1 + x2) / 2, y12 = (y1 + y2) / 2;
        double x23 = (x2 + x3) / 2, y23 = (y2 + y3) / 2;
        double x31 = (x3 + x1) / 2, y31 = (y3 + y1) / 2;
        int n12 = node_offset++;
        int n23 = node_offset++;
        int n31 = node_offset++;
        new_grid_x.insert(new_grid_x.end(), {x12, x23, x31});
        new_grid_y.insert(new_grid_y.end(), {y12, y23, y31});
        new_elements.push_back({i1, n12, n31});
        new_elements.push_back({n12, i2, n23});
        new_elements.push_back({n31, n23, i3});
        new_elements.push_back({n12, n23, n31});
    }
    grid_x_.insert(grid_x_.end(), new_grid_x.begin(), new_grid_x.end());
    grid_y_.insert(grid_y_.end(), new_grid_y.begin(), new_grid_y.end());
    elements_ = new_elements;
}
} // namespace simulator