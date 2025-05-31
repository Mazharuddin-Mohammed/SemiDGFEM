#include "mesh.hpp"
#include <gmsh.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace simulator {

Mesh::Mesh(const Device& device, MeshType type) : device_(device), type_(type) {
    if (!device.is_valid()) {
        throw std::invalid_argument("Invalid device provided to mesh constructor");
    }

    try {
        if (type_ == MeshType::Structured) {
            generate_structured_mesh();
        }
        validate();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create mesh: " + std::string(e.what()));
    }
}

void Mesh::generate_structured_mesh() {
    auto extents = device_.get_extents();
    double Lx = extents[0], Ly = extents[1];

    if (Lx <= 0.0 || Ly <= 0.0) {
        throw std::invalid_argument("Invalid device extents for mesh generation");
    }

    const int nx = 50, ny = 25; // Default grid

    // Clear existing data
    grid_x_.clear();
    grid_y_.clear();
    elements_.clear();

    // Reserve memory to avoid reallocations
    grid_x_.reserve((nx + 1) * (ny + 1));
    grid_y_.reserve((nx + 1) * (ny + 1));
    elements_.reserve(2 * nx * ny);

    // Generate grid points
    for (int i = 0; i <= nx; ++i) {
        for (int j = 0; j <= ny; ++j) {
            grid_x_.push_back(i * Lx / nx);
            grid_y_.push_back(j * Ly / ny);
        }
    }

    // Generate elements (triangles)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int n1 = i * (ny + 1) + j;
            int n2 = n1 + 1;
            int n3 = (i + 1) * (ny + 1) + j;
            int n4 = n3 + 1;

            // Validate node indices
            int max_node = (nx + 1) * (ny + 1) - 1;
            if (n1 > max_node || n2 > max_node || n3 > max_node || n4 > max_node) {
                throw std::runtime_error("Invalid node index in element generation");
            }

            elements_.push_back({n1, n3, n2});
            elements_.push_back({n2, n3, n4});
        }
    }
}

void Mesh::generate_gmsh_mesh(const std::string& filename) {
    if (type_ != MeshType::Unstructured) {
        throw std::invalid_argument("GMSH mesh generation only supported for unstructured meshes");
    }

    if (filename.empty()) {
        throw std::invalid_argument("Empty filename provided for mesh generation");
    }

    try {
        gmsh::initialize();
        gmsh::model::add("device");

        auto extents = device_.get_extents();
        double Lx = extents[0], Ly = extents[1];

        if (Lx <= 0.0 || Ly <= 0.0) {
            throw std::invalid_argument("Invalid device extents for GMSH mesh generation");
        }

        double mesh_size = std::min(Lx, Ly) / 50.0;

        // Add points
        int p1 = gmsh::model::geo::addPoint(0, 0, 0, mesh_size);
        int p2 = gmsh::model::geo::addPoint(Lx, 0, 0, mesh_size);
        int p3 = gmsh::model::geo::addPoint(Lx, Ly, 0, mesh_size);
        int p4 = gmsh::model::geo::addPoint(0, Ly, 0, mesh_size);

        // Add lines
        int l1 = gmsh::model::geo::addLine(p1, p2);
        int l2 = gmsh::model::geo::addLine(p2, p3);
        int l3 = gmsh::model::geo::addLine(p3, p4);
        int l4 = gmsh::model::geo::addLine(p4, p1);

        // Add curve loop and surface
        int cl = gmsh::model::geo::addCurveLoop({l1, l2, l3, l4});
        int s = gmsh::model::geo::addPlaneSurface({cl});

        gmsh::model::geo::synchronize();
        gmsh::model::mesh::generate(2);
        gmsh::write(filename);

        // Extract mesh data
        grid_x_.clear();
        grid_y_.clear();
        elements_.clear();

        std::vector<std::size_t> nodeTags;
        std::vector<double> coord, paramCoord;
        gmsh::model::mesh::getNodes(nodeTags, coord, paramCoord);

        if (coord.size() % 3 != 0) {
            throw std::runtime_error("Invalid coordinate data from GMSH");
        }

        size_t num_nodes = nodeTags.size();
        grid_x_.reserve(num_nodes);
        grid_y_.reserve(num_nodes);

        for (size_t i = 0; i < num_nodes; ++i) {
            if (i * 3 + 1 >= coord.size()) {
                throw std::runtime_error("Coordinate index out of bounds");
            }
            grid_x_.push_back(coord[i * 3]);
            grid_y_.push_back(coord[i * 3 + 1]);
        }

        std::vector<int> elemTypes;
        std::vector<std::vector<std::size_t>> elemTags, elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, 2);

        if (!elemNodeTags.empty()) {
            elements_.reserve(elemNodeTags[0].size() / 3);
            for (size_t i = 0; i < elemNodeTags[0].size(); i += 3) {
                if (i + 2 >= elemNodeTags[0].size()) {
                    throw std::runtime_error("Element node index out of bounds");
                }

                int n1 = static_cast<int>(elemNodeTags[0][i] - 1);
                int n2 = static_cast<int>(elemNodeTags[0][i + 1] - 1);
                int n3 = static_cast<int>(elemNodeTags[0][i + 2] - 1);

                // Validate node indices
                if (n1 < 0 || n2 < 0 || n3 < 0 ||
                    n1 >= static_cast<int>(num_nodes) ||
                    n2 >= static_cast<int>(num_nodes) ||
                    n3 >= static_cast<int>(num_nodes)) {
                    throw std::runtime_error("Invalid node index in element");
                }

                elements_.push_back({n1, n2, n3});
            }
        }

        gmsh::finalize();
        validate();

    } catch (const std::exception& e) {
        gmsh::finalize(); // Ensure cleanup
        throw std::runtime_error("GMSH mesh generation failed: " + std::string(e.what()));
    }
}

void Mesh::refine(const std::vector<bool>& refine_flags) {
    if (refine_flags.size() != elements_.size()) {
        throw std::invalid_argument("Refine flags size must match number of elements");
    }

    if (type_ == MeshType::Unstructured) {
        generate_gmsh_mesh("device_refined.msh");
        return;
    }

    std::vector<double> new_grid_x, new_grid_y;
    std::vector<std::vector<int>> new_elements;
    int node_offset = static_cast<int>(grid_x_.size());

    for (size_t e = 0; e < elements_.size(); ++e) {
        if (e >= refine_flags.size()) {
            throw std::runtime_error("Element index out of bounds in refinement");
        }

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

    validate();
}

bool Mesh::is_valid() const {
    if (grid_x_.size() != grid_y_.size()) return false;
    if (grid_x_.empty() || elements_.empty()) return false;

    // Check element connectivity
    for (const auto& element : elements_) {
        if (element.size() != 3) return false; // Triangular elements only
        for (int node_id : element) {
            if (node_id < 0 || node_id >= static_cast<int>(grid_x_.size())) {
                return false;
            }
        }
    }
    return true;
}

void Mesh::validate() const {
    if (!is_valid()) {
        throw std::runtime_error("Mesh validation failed");
    }
    validate_element_connectivity();
}

void Mesh::validate_element_connectivity() const {
    for (size_t e = 0; e < elements_.size(); ++e) {
        const auto& element = elements_[e];
        if (element.size() != 3) {
            throw std::runtime_error("Non-triangular element found at index " + std::to_string(e));
        }

        for (size_t i = 0; i < element.size(); ++i) {
            int node_id = element[i];
            if (node_id < 0 || node_id >= static_cast<int>(grid_x_.size())) {
                throw std::runtime_error("Invalid node index " + std::to_string(node_id) +
                                       " in element " + std::to_string(e));
            }
        }

        // Check for degenerate elements
        double quality = compute_element_quality(element);
        if (quality < 1e-12) {
            throw std::runtime_error("Degenerate element found at index " + std::to_string(e));
        }
    }
}

double Mesh::compute_element_quality(const std::vector<int>& element) const {
    if (element.size() != 3) return 0.0;

    int i1 = element[0], i2 = element[1], i3 = element[2];
    if (i1 >= static_cast<int>(grid_x_.size()) || i2 >= static_cast<int>(grid_x_.size()) ||
        i3 >= static_cast<int>(grid_x_.size())) {
        return 0.0;
    }

    double x1 = grid_x_[i1], y1 = grid_y_[i1];
    double x2 = grid_x_[i2], y2 = grid_y_[i2];
    double x3 = grid_x_[i3], y3 = grid_y_[i3];

    // Compute area using cross product
    double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

    // Compute perimeter
    double l1 = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    double l2 = std::sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
    double l3 = std::sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3));
    double perimeter = l1 + l2 + l3;

    // Quality metric: 4*sqrt(3)*area / perimeter^2 (ranges from 0 to 1)
    if (perimeter < 1e-12) return 0.0;
    return 4.0 * std::sqrt(3.0) * area / (perimeter * perimeter);
}

double Mesh::get_min_element_quality() const {
    if (elements_.empty()) return 0.0;

    double min_quality = 1.0;
    for (const auto& element : elements_) {
        double quality = compute_element_quality(element);
        min_quality = std::min(min_quality, quality);
    }
    return min_quality;
}

double Mesh::get_max_element_size() const {
    if (elements_.empty()) return 0.0;

    double max_size = 0.0;
    for (const auto& element : elements_) {
        if (element.size() != 3) continue;

        int i1 = element[0], i2 = element[1], i3 = element[2];
        if (i1 >= static_cast<int>(grid_x_.size()) || i2 >= static_cast<int>(grid_x_.size()) ||
            i3 >= static_cast<int>(grid_x_.size())) {
            continue;
        }

        double x1 = grid_x_[i1], y1 = grid_y_[i1];
        double x2 = grid_x_[i2], y2 = grid_y_[i2];
        double x3 = grid_x_[i3], y3 = grid_y_[i3];

        double l1 = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        double l2 = std::sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
        double l3 = std::sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3));

        double element_size = std::max({l1, l2, l3});
        max_size = std::max(max_size, element_size);
    }
    return max_size;
}

// C interface implementations
extern "C" {
    Mesh* create_mesh(Device* device, int mesh_type) {
        if (!device) return nullptr;
        try {
            return new Mesh(*device, static_cast<MeshType>(mesh_type));
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_mesh(Mesh* mesh) {
        delete mesh;
    }

    void mesh_generate_gmsh(Mesh* mesh, const char* filename) {
        if (!mesh || !filename) return;
        try {
            mesh->generate_gmsh_mesh(std::string(filename));
        } catch (...) {
            // Error handling in calling code
        }
    }

    int mesh_get_num_nodes(Mesh* mesh) {
        if (!mesh) return 0;
        return static_cast<int>(mesh->get_num_nodes());
    }

    int mesh_get_num_elements(Mesh* mesh) {
        if (!mesh) return 0;
        return static_cast<int>(mesh->get_num_elements());
    }

    void mesh_get_grid_points_x(Mesh* mesh, double* points, int size) {
        if (!mesh || !points || size <= 0) return;
        try {
            auto grid_x = mesh->get_grid_points_x();
            int copy_size = std::min(size, static_cast<int>(grid_x.size()));
            for (int i = 0; i < copy_size; ++i) {
                points[i] = grid_x[i];
            }
        } catch (...) {
            // Error handling in calling code
        }
    }

    void mesh_get_grid_points_y(Mesh* mesh, double* points, int size) {
        if (!mesh || !points || size <= 0) return;
        try {
            auto grid_y = mesh->get_grid_points_y();
            int copy_size = std::min(size, static_cast<int>(grid_y.size()));
            for (int i = 0; i < copy_size; ++i) {
                points[i] = grid_y[i];
            }
        } catch (...) {
            // Error handling in calling code
        }
    }

    void mesh_get_elements(Mesh* mesh, int* elements, int num_elements, int nodes_per_element) {
        if (!mesh || !elements || num_elements <= 0 || nodes_per_element <= 0) return;
        try {
            auto mesh_elements = mesh->get_elements();
            int copy_elements = std::min(num_elements, static_cast<int>(mesh_elements.size()));

            for (int e = 0; e < copy_elements; ++e) {
                int copy_nodes = std::min(nodes_per_element, static_cast<int>(mesh_elements[e].size()));
                for (int n = 0; n < copy_nodes; ++n) {
                    elements[e * nodes_per_element + n] = mesh_elements[e][n];
                }
                // Fill remaining nodes with -1 if element has fewer nodes
                for (int n = copy_nodes; n < nodes_per_element; ++n) {
                    elements[e * nodes_per_element + n] = -1;
                }
            }
        } catch (...) {
            // Error handling in calling code
        }
    }
}

} // namespace simulator