#include "advanced_mesh_refinement.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iomanip>

namespace simulator {
namespace amr {

// ============================================================================
// MeshRefinementExecutor Implementation
// ============================================================================

MeshRefinementExecutor::MeshRefinementExecutor() {}

MeshRefinementExecutor::RefinementResult MeshRefinementExecutor::execute_refinement(
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices,
    const std::vector<AdvancedRefinementInfo>& refinement_info) {
    
    RefinementResult result;
    
    try {
        // Initialize with original mesh
        result.new_elements = elements;
        result.new_vertices = vertices;
        result.element_levels.resize(elements.size(), 0);
        result.parent_mapping.resize(elements.size());
        result.vertex_mapping.resize(vertices.size());
        
        // Initialize mappings
        std::iota(result.parent_mapping.begin(), result.parent_mapping.end(), 0);
        std::iota(result.vertex_mapping.begin(), result.vertex_mapping.end(), 0);
        
        // Vertex map for avoiding duplicates
        std::unordered_map<std::string, int> vertex_map;
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertex_map[vertex_key(vertices[i])] = static_cast<int>(i);
        }
        
        // Process refinement requests
        std::vector<std::vector<int>> elements_to_add;
        std::vector<int> elements_to_remove;
        
        for (size_t e = 0; e < elements.size(); ++e) {
            if (e >= refinement_info.size() || !refinement_info[e].refine) {
                continue;
            }
            
            const auto& element = elements[e];
            const auto& ref_info = refinement_info[e];
            
            std::vector<std::vector<int>> new_elements;
            
            if (ref_info.direction == RefinementDirection::ISOTROPIC) {
                new_elements = refine_element_isotropic(element, result.new_vertices, 
                                                       result.new_vertices);
            } else {
                new_elements = refine_element_anisotropic(element, result.new_vertices,
                                                        ref_info.direction,
                                                        ref_info.anisotropy_ratio,
                                                        result.new_vertices);
            }
            
            // Add new elements
            for (const auto& new_elem : new_elements) {
                elements_to_add.push_back(new_elem);
                result.parent_mapping.push_back(static_cast<int>(e));
                result.element_levels.push_back(result.element_levels[e] + 1);
            }
            
            // Mark original element for removal
            elements_to_remove.push_back(static_cast<int>(e));
        }
        
        // Remove refined elements (in reverse order to maintain indices)
        std::sort(elements_to_remove.rbegin(), elements_to_remove.rend());
        for (int elem_idx : elements_to_remove) {
            result.new_elements.erase(result.new_elements.begin() + elem_idx);
            result.element_levels.erase(result.element_levels.begin() + elem_idx);
            result.parent_mapping.erase(result.parent_mapping.begin() + elem_idx);
        }
        
        // Add new elements
        result.new_elements.insert(result.new_elements.end(), 
                                 elements_to_add.begin(), elements_to_add.end());
        
        // Update vertex mapping
        result.vertex_mapping.resize(result.new_vertices.size());
        for (size_t i = vertices.size(); i < result.new_vertices.size(); ++i) {
            result.vertex_mapping[i] = -1; // New vertex
        }
        
        // Ensure mesh conformity
        ensure_mesh_conformity(result.new_elements, result.new_vertices);
        
        result.success = true;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = "Refinement failed: " + std::string(e.what());
    }
    
    return result;
}

std::vector<std::vector<int>> MeshRefinementExecutor::refine_element_isotropic(
    const std::vector<int>& element,
    const std::vector<std::array<double, 2>>& vertices,
    std::vector<std::array<double, 2>>& new_vertices) {
    
    if (element.size() != 3) {
        throw std::invalid_argument("Only triangular elements supported");
    }
    
    // Get element vertices
    auto v1 = vertices[element[0]];
    auto v2 = vertices[element[1]];
    auto v3 = vertices[element[2]];
    
    // Create midpoint vertices
    std::array<double, 2> m12 = {(v1[0] + v2[0]) / 2.0, (v1[1] + v2[1]) / 2.0};
    std::array<double, 2> m23 = {(v2[0] + v3[0]) / 2.0, (v2[1] + v3[1]) / 2.0};
    std::array<double, 2> m31 = {(v3[0] + v1[0]) / 2.0, (v3[1] + v1[1]) / 2.0};
    
    // Add new vertices
    std::unordered_map<std::string, int> vertex_map;
    for (size_t i = 0; i < new_vertices.size(); ++i) {
        vertex_map[vertex_key(new_vertices[i])] = static_cast<int>(i);
    }
    
    int idx_m12 = add_vertex_if_new(m12, new_vertices, vertex_map);
    int idx_m23 = add_vertex_if_new(m23, new_vertices, vertex_map);
    int idx_m31 = add_vertex_if_new(m31, new_vertices, vertex_map);
    
    // Create four new triangular elements
    std::vector<std::vector<int>> new_elements = {
        {element[0], idx_m12, idx_m31},  // Corner triangle 1
        {element[1], idx_m23, idx_m12},  // Corner triangle 2
        {element[2], idx_m31, idx_m23},  // Corner triangle 3
        {idx_m12, idx_m23, idx_m31}     // Central triangle
    };
    
    return new_elements;
}

std::vector<std::vector<int>> MeshRefinementExecutor::refine_element_anisotropic(
    const std::vector<int>& element,
    const std::vector<std::array<double, 2>>& vertices,
    RefinementDirection direction,
    double anisotropy_ratio,
    std::vector<std::array<double, 2>>& new_vertices) {
    
    if (element.size() != 3) {
        throw std::invalid_argument("Only triangular elements supported");
    }
    
    // Get element vertices
    auto v1 = vertices[element[0]];
    auto v2 = vertices[element[1]];
    auto v3 = vertices[element[2]];
    
    std::vector<std::vector<int>> new_elements;
    std::unordered_map<std::string, int> vertex_map;
    for (size_t i = 0; i < new_vertices.size(); ++i) {
        vertex_map[vertex_key(new_vertices[i])] = static_cast<int>(i);
    }
    
    if (direction == RefinementDirection::X_DIRECTION) {
        // Refine primarily in X direction
        // Find the edge most aligned with Y direction (to split in X)
        double edge12_y = std::abs(v2[1] - v1[1]);
        double edge23_y = std::abs(v3[1] - v2[1]);
        double edge31_y = std::abs(v1[1] - v3[1]);
        
        if (edge12_y >= edge23_y && edge12_y >= edge31_y) {
            // Split edge 1-2
            std::array<double, 2> mid = {(v1[0] + v2[0]) / 2.0, (v1[1] + v2[1]) / 2.0};
            int idx_mid = add_vertex_if_new(mid, new_vertices, vertex_map);
            new_elements = {{element[0], idx_mid, element[2]}, 
                          {idx_mid, element[1], element[2]}};
        } else if (edge23_y >= edge31_y) {
            // Split edge 2-3
            std::array<double, 2> mid = {(v2[0] + v3[0]) / 2.0, (v2[1] + v3[1]) / 2.0};
            int idx_mid = add_vertex_if_new(mid, new_vertices, vertex_map);
            new_elements = {{element[0], element[1], idx_mid}, 
                          {element[0], idx_mid, element[2]}};
        } else {
            // Split edge 3-1
            std::array<double, 2> mid = {(v3[0] + v1[0]) / 2.0, (v3[1] + v1[1]) / 2.0};
            int idx_mid = add_vertex_if_new(mid, new_vertices, vertex_map);
            new_elements = {{element[0], element[1], idx_mid}, 
                          {idx_mid, element[1], element[2]}};
        }
    } else if (direction == RefinementDirection::Y_DIRECTION) {
        // Refine primarily in Y direction
        // Find the edge most aligned with X direction (to split in Y)
        double edge12_x = std::abs(v2[0] - v1[0]);
        double edge23_x = std::abs(v3[0] - v2[0]);
        double edge31_x = std::abs(v1[0] - v3[0]);
        
        if (edge12_x >= edge23_x && edge12_x >= edge31_x) {
            // Split edge 1-2
            std::array<double, 2> mid = {(v1[0] + v2[0]) / 2.0, (v1[1] + v2[1]) / 2.0};
            int idx_mid = add_vertex_if_new(mid, new_vertices, vertex_map);
            new_elements = {{element[0], idx_mid, element[2]}, 
                          {idx_mid, element[1], element[2]}};
        } else if (edge23_x >= edge31_x) {
            // Split edge 2-3
            std::array<double, 2> mid = {(v2[0] + v3[0]) / 2.0, (v2[1] + v3[1]) / 2.0};
            int idx_mid = add_vertex_if_new(mid, new_vertices, vertex_map);
            new_elements = {{element[0], element[1], idx_mid}, 
                          {element[0], idx_mid, element[2]}};
        } else {
            // Split edge 3-1
            std::array<double, 2> mid = {(v3[0] + v1[0]) / 2.0, (v3[1] + v1[1]) / 2.0};
            int idx_mid = add_vertex_if_new(mid, new_vertices, vertex_map);
            new_elements = {{element[0], element[1], idx_mid}, 
                          {idx_mid, element[1], element[2]}};
        }
    } else {
        // Adaptive direction - use isotropic refinement as fallback
        return refine_element_isotropic(element, vertices, new_vertices);
    }
    
    return new_elements;
}

int MeshRefinementExecutor::add_vertex_if_new(
    const std::array<double, 2>& vertex,
    std::vector<std::array<double, 2>>& vertices,
    std::unordered_map<std::string, int>& vertex_map) {
    
    std::string key = vertex_key(vertex);
    auto it = vertex_map.find(key);
    
    if (it != vertex_map.end()) {
        return it->second;
    }
    
    int new_index = static_cast<int>(vertices.size());
    vertices.push_back(vertex);
    vertex_map[key] = new_index;
    return new_index;
}

std::string MeshRefinementExecutor::vertex_key(const std::array<double, 2>& vertex) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12) << vertex[0] << "," << vertex[1];
    return oss.str();
}

void MeshRefinementExecutor::ensure_mesh_conformity(
    std::vector<std::vector<int>>& elements,
    std::vector<std::array<double, 2>>& vertices) {
    
    // Simple conformity check - ensure all vertex indices are valid
    for (auto& element : elements) {
        for (int& vertex_idx : element) {
            if (vertex_idx < 0 || vertex_idx >= static_cast<int>(vertices.size())) {
                throw std::runtime_error("Invalid vertex index in refined mesh");
            }
        }
    }
    
    // Additional conformity checks could be added here
    // such as hanging node resolution, edge consistency, etc.
}

std::vector<double> MeshRefinementExecutor::transfer_solution(
    const std::vector<double>& old_solution,
    const RefinementResult& refinement_result,
    const std::string& transfer_method) {
    
    if (!refinement_result.success) {
        return old_solution;
    }
    
    std::vector<double> new_solution(refinement_result.new_vertices.size(), 0.0);
    
    if (transfer_method == "interpolation") {
        // Linear interpolation for new vertices
        for (size_t i = 0; i < refinement_result.vertex_mapping.size(); ++i) {
            int original_vertex = refinement_result.vertex_mapping[i];
            if (original_vertex >= 0 && original_vertex < static_cast<int>(old_solution.size())) {
                // Original vertex - copy solution
                new_solution[i] = old_solution[original_vertex];
            } else {
                // New vertex - interpolate from neighbors
                // Simple average of surrounding vertices (could be improved)
                double sum = 0.0;
                int count = 0;
                
                for (const auto& element : refinement_result.new_elements) {
                    bool contains_vertex = std::find(element.begin(), element.end(), 
                                                   static_cast<int>(i)) != element.end();
                    if (contains_vertex) {
                        for (int vertex_idx : element) {
                            if (vertex_idx != static_cast<int>(i) && 
                                vertex_idx < static_cast<int>(old_solution.size())) {
                                sum += old_solution[vertex_idx];
                                count++;
                            }
                        }
                    }
                }
                
                if (count > 0) {
                    new_solution[i] = sum / count;
                }
            }
        }
    } else {
        // Default: copy available values
        for (size_t i = 0; i < std::min(new_solution.size(), old_solution.size()); ++i) {
            new_solution[i] = old_solution[i];
        }
    }
    
    return new_solution;
}

} // namespace amr
} // namespace simulator
