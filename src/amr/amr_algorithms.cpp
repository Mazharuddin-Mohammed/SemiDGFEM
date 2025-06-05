#include "amr_algorithms.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <unordered_set>
#include <map>

namespace simulator {
namespace amr {

AMRController::AMRController(ErrorEstimatorType estimator_type,
                           RefinementStrategy strategy,
                           int max_refinement_levels)
    : estimator_type_(estimator_type), strategy_(strategy),
      max_refinement_levels_(max_refinement_levels),
      refine_fraction_(0.3), coarsen_fraction_(0.1),
      error_tolerance_(1e-3), min_element_size_(1e-6), max_element_size_(1e-2),
      anisotropic_enabled_(false), boundary_layer_detection_(true) {
}

void AMRController::set_refinement_parameters(double refine_fraction,
                                             double coarsen_fraction,
                                             double error_tolerance,
                                             double min_element_size,
                                             double max_element_size) {
    if (refine_fraction < 0.0 || refine_fraction > 1.0 ||
        coarsen_fraction < 0.0 || coarsen_fraction > 1.0) {
        throw std::invalid_argument("Refinement fractions must be between 0 and 1");
    }
    
    refine_fraction_ = refine_fraction;
    coarsen_fraction_ = coarsen_fraction;
    error_tolerance_ = error_tolerance;
    min_element_size_ = min_element_size;
    max_element_size_ = max_element_size;
}

std::vector<double> AMRController::compute_error_indicators(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices,
    const std::vector<std::array<double, 2>>& solution_gradient) {
    
    if (solution.empty() || elements.empty() || vertices.empty()) {
        throw std::invalid_argument("Empty input data for error estimation");
    }
    
    switch (estimator_type_) {
        case ErrorEstimatorType::GRADIENT_BASED:
            return compute_gradient_based_error(solution, elements, vertices);
        case ErrorEstimatorType::RESIDUAL_BASED:
            return compute_residual_based_error(solution, elements, vertices);
        case ErrorEstimatorType::KELLY:
            return compute_kelly_error(solution, elements, vertices);
        case ErrorEstimatorType::ZZ_SUPERCONVERGENT:
            return compute_zz_error(solution, elements, vertices);
        default:
            return compute_gradient_based_error(solution, elements, vertices);
    }
}

std::vector<double> AMRController::compute_gradient_based_error(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices) {
    
    std::vector<double> error_indicators(elements.size(), 0.0);
    
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& element = elements[e];
        if (element.size() < 3) continue;
        
        // Get element vertices
        std::array<std::array<double, 2>, 3> elem_vertices;
        for (int i = 0; i < 3; ++i) {
            if (element[i] >= static_cast<int>(vertices.size())) continue;
            elem_vertices[i] = vertices[element[i]];
        }
        
        // Compute element area
        double area = utils::compute_triangle_area(elem_vertices);
        if (area < 1e-12) continue;
        
        // Compute gradient using finite differences
        auto gradient = utils::compute_element_gradient(solution, element, vertices);
        double grad_magnitude = std::sqrt(gradient[0] * gradient[0] + gradient[1] * gradient[1]);
        
        // Element size (diameter)
        double h = std::sqrt(area);
        
        // Gradient-based error indicator: h * |∇u|
        error_indicators[e] = h * grad_magnitude;
        
        // Add curvature term for higher-order accuracy
        if (e > 0 && e < elements.size() - 1) {
            // Approximate second derivative using neighboring elements
            double curvature = 0.0;

            // Find neighboring elements that share vertices
            std::vector<size_t> neighbors;
            for (size_t other = 0; other < elements.size(); ++other) {
                if (other == e) continue;

                // Check if elements share at least one vertex
                int shared_vertices = 0;
                for (int v1 : elements[e]) {
                    for (int v2 : elements[other]) {
                        if (v1 == v2) shared_vertices++;
                    }
                }
                if (shared_vertices >= 2) { // Share an edge
                    neighbors.push_back(other);
                }
            }

            // Compute curvature using neighboring gradients
            for (size_t neighbor_idx : neighbors) {
                auto grad_neighbor = utils::compute_element_gradient(solution, elements[neighbor_idx], vertices);
                curvature += std::abs(gradient[0] - grad_neighbor[0]) + std::abs(gradient[1] - grad_neighbor[1]);
            }

            if (!neighbors.empty()) {
                curvature /= neighbors.size(); // Average curvature
                error_indicators[e] += h * h * curvature;
            }
        }
    }
    
    return error_indicators;
}

std::vector<double> AMRController::compute_kelly_error(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices) {
    
    std::vector<double> error_indicators(elements.size(), 0.0);
    auto neighbors = utils::compute_element_neighbors(elements);
    
    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& element = elements[e];
        if (element.size() < 3) continue;
        
        double element_error = 0.0;
        
        // Loop over element faces
        for (int face = 0; face < 3; ++face) {
            int v1 = element[face];
            int v2 = element[(face + 1) % 3];
            
            if (v1 >= static_cast<int>(vertices.size()) || v2 >= static_cast<int>(vertices.size())) {
                continue;
            }
            
            // Face length
            double face_length = std::sqrt(
                std::pow(vertices[v2][0] - vertices[v1][0], 2) +
                std::pow(vertices[v2][1] - vertices[v1][1], 2)
            );
            
            // Compute jump in normal derivative across face
            double jump = 0.0;
            
            // Find neighboring element sharing this face
            for (int neighbor_id : neighbors[e]) {
                if (neighbor_id >= 0 && neighbor_id < static_cast<int>(elements.size())) {
                    const auto& neighbor = elements[neighbor_id];
                    
                    // Check if neighbor shares this face
                    bool shares_face = false;
                    for (int nv : neighbor) {
                        if (nv == v1 || nv == v2) {
                            shares_face = true;
                            break;
                        }
                    }
                    
                    if (shares_face) {
                        // Compute gradients on both sides
                        auto grad_e = utils::compute_element_gradient(solution, element, vertices);
                        auto grad_n = utils::compute_element_gradient(solution, neighbor, vertices);
                        
                        // Face normal (outward pointing from element e)
                        // For 2D, normal to edge (v1->v2) is perpendicular vector
                        double edge_x = vertices[v2][0] - vertices[v1][0];
                        double edge_y = vertices[v2][1] - vertices[v1][1];

                        // Perpendicular vector (rotated 90 degrees)
                        double nx = -edge_y / face_length;
                        double ny = edge_x / face_length;

                        // Ensure normal points outward from element e
                        // Check orientation using third vertex of element
                        int third_vertex = -1;
                        for (int v : element) {
                            if (v != v1 && v != v2) {
                                third_vertex = v;
                                break;
                            }
                        }

                        if (third_vertex >= 0) {
                            // Vector from edge midpoint to third vertex
                            double mid_x = (vertices[v1][0] + vertices[v2][0]) / 2.0;
                            double mid_y = (vertices[v1][1] + vertices[v2][1]) / 2.0;
                            double to_third_x = vertices[third_vertex][0] - mid_x;
                            double to_third_y = vertices[third_vertex][1] - mid_y;

                            // If normal points toward third vertex, flip it
                            if (nx * to_third_x + ny * to_third_y > 0) {
                                nx = -nx;
                                ny = -ny;
                            }
                        }
                        
                        // Normal derivatives
                        double dn_e = grad_e[0] * nx + grad_e[1] * ny;
                        double dn_n = grad_n[0] * nx + grad_n[1] * ny;
                        
                        jump = std::abs(dn_e - dn_n);
                        break;
                    }
                }
            }
            
            // Kelly error contribution: h^(1/2) * |[∂u/∂n]|
            element_error += std::sqrt(face_length) * jump;
        }
        
        error_indicators[e] = element_error;
    }
    
    return error_indicators;
}

std::vector<ElementRefinement> AMRController::determine_refinement(
    const std::vector<double>& error_indicators,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices,
    const std::vector<int>& current_levels) {
    
    std::vector<ElementRefinement> refinement_decisions(elements.size());
    
    if (error_indicators.empty()) {
        return refinement_decisions;
    }
    
    // Initialize refinement decisions
    for (size_t e = 0; e < elements.size(); ++e) {
        refinement_decisions[e].element_id = static_cast<int>(e);
        refinement_decisions[e].refine = false;
        refinement_decisions[e].coarsen = false;
        refinement_decisions[e].error_indicator = error_indicators[e];
        refinement_decisions[e].refinement_level = 
            (current_levels.empty()) ? 0 : current_levels[e];
    }
    
    // Sort elements by error indicator
    std::vector<size_t> sorted_indices(elements.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](size_t a, size_t b) {
                  return error_indicators[a] > error_indicators[b];
              });
    
    // Apply refinement strategy
    switch (strategy_) {
        case RefinementStrategy::FIXED_FRACTION: {
            size_t num_refine = static_cast<size_t>(refine_fraction_ * elements.size());
            size_t num_coarsen = static_cast<size_t>(coarsen_fraction_ * elements.size());
            
            // Mark elements for refinement (highest error)
            for (size_t i = 0; i < std::min(num_refine, sorted_indices.size()); ++i) {
                size_t elem_idx = sorted_indices[i];
                if (refinement_decisions[elem_idx].refinement_level < max_refinement_levels_) {
                    // Check element size constraint
                    std::array<std::array<double, 2>, 3> elem_vertices;
                    for (int j = 0; j < 3; ++j) {
                        elem_vertices[j] = vertices[elements[elem_idx][j]];
                    }
                    double area = utils::compute_triangle_area(elem_vertices);
                    double h = std::sqrt(area);
                    
                    if (h > min_element_size_ * 2.0) {  // Ensure refined elements won't be too small
                        refinement_decisions[elem_idx].refine = true;
                        
                        // Detect anisotropy if enabled
                        if (anisotropic_enabled_) {
                            refinement_decisions[elem_idx].anisotropy = 
                                detect_anisotropy_direction(elem_idx, {}, elements, vertices);
                        }
                    }
                }
            }
            
            // Mark elements for coarsening (lowest error)
            for (size_t i = sorted_indices.size() - num_coarsen; i < sorted_indices.size(); ++i) {
                size_t elem_idx = sorted_indices[i];
                if (refinement_decisions[elem_idx].refinement_level > 0) {
                    // Check element size constraint
                    std::array<std::array<double, 2>, 3> elem_vertices;
                    for (int j = 0; j < 3; ++j) {
                        elem_vertices[j] = vertices[elements[elem_idx][j]];
                    }
                    double area = utils::compute_triangle_area(elem_vertices);
                    double h = std::sqrt(area);
                    
                    if (h < max_element_size_ / 2.0) {  // Ensure coarsened elements won't be too large
                        refinement_decisions[elem_idx].coarsen = true;
                    }
                }
            }
            break;
        }
        
        case RefinementStrategy::EQUILIBRATION: {
            // Target error per element
            double total_error = std::accumulate(error_indicators.begin(), error_indicators.end(), 0.0);
            double target_error_per_element = total_error / elements.size();
            
            for (size_t e = 0; e < elements.size(); ++e) {
                if (error_indicators[e] > 2.0 * target_error_per_element) {
                    refinement_decisions[e].refine = true;
                } else if (error_indicators[e] < 0.5 * target_error_per_element) {
                    refinement_decisions[e].coarsen = true;
                }
            }
            break;
        }
        
        default:
            // Default to fixed fraction strategy
            break;
    }
    
    return refinement_decisions;
}

void AMRController::set_anisotropic_refinement(bool enable, bool boundary_layer_detection) {
    anisotropic_enabled_ = enable;
    boundary_layer_detection_ = boundary_layer_detection;
}

MeshQuality AMRController::compute_mesh_quality(
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices) const {
    
    MeshQuality quality;
    quality.min_angle = 180.0;
    quality.max_angle = 0.0;
    quality.min_area = std::numeric_limits<double>::max();
    quality.max_area = 0.0;
    quality.aspect_ratio_min = std::numeric_limits<double>::max();
    quality.aspect_ratio_max = 0.0;
    quality.skewness_max = 0.0;
    quality.num_bad_elements = 0;
    
    for (const auto& element : elements) {
        if (element.size() < 3) continue;
        
        std::array<std::array<double, 2>, 3> elem_vertices;
        for (int i = 0; i < 3; ++i) {
            if (element[i] >= static_cast<int>(vertices.size())) continue;
            elem_vertices[i] = vertices[element[i]];
        }
        
        // Compute area
        double area = utils::compute_triangle_area(elem_vertices);
        quality.min_area = std::min(quality.min_area, area);
        quality.max_area = std::max(quality.max_area, area);
        
        // Compute aspect ratio
        double aspect_ratio = utils::compute_aspect_ratio(elem_vertices);
        quality.aspect_ratio_min = std::min(quality.aspect_ratio_min, aspect_ratio);
        quality.aspect_ratio_max = std::max(quality.aspect_ratio_max, aspect_ratio);
        
        // Compute skewness
        double skewness = utils::compute_skewness(elem_vertices);
        quality.skewness_max = std::max(quality.skewness_max, skewness);
        
        // Compute angles
        for (int i = 0; i < 3; ++i) {
            int i1 = i, i2 = (i + 1) % 3, i3 = (i + 2) % 3;
            
            double v1[2] = {elem_vertices[i2][0] - elem_vertices[i1][0], 
                           elem_vertices[i2][1] - elem_vertices[i1][1]};
            double v2[2] = {elem_vertices[i3][0] - elem_vertices[i1][0], 
                           elem_vertices[i3][1] - elem_vertices[i1][1]};
            
            double dot = v1[0] * v2[0] + v1[1] * v2[1];
            double mag1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1]);
            double mag2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1]);
            
            if (mag1 > 1e-12 && mag2 > 1e-12) {
                double angle = std::acos(std::clamp(dot / (mag1 * mag2), -1.0, 1.0)) * 180.0 / M_PI;
                quality.min_angle = std::min(quality.min_angle, angle);
                quality.max_angle = std::max(quality.max_angle, angle);
            }
        }
        
        // Check if element is "bad" (poor quality)
        if (quality.min_angle < 10.0 || quality.max_angle > 170.0 || 
            aspect_ratio > 10.0 || skewness > 0.8) {
            quality.num_bad_elements++;
        }
    }
    
    return quality;
}

// Utility function implementations
namespace utils {

double compute_triangle_area(const std::array<std::array<double, 2>, 3>& vertices) {
    const auto& v0 = vertices[0];
    const auto& v1 = vertices[1];
    const auto& v2 = vertices[2];

    return 0.5 * std::abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]));
}

double compute_aspect_ratio(const std::array<std::array<double, 2>, 3>& vertices) {
    // Compute edge lengths
    double edges[3];
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;
        edges[i] = std::sqrt(
            std::pow(vertices[j][0] - vertices[i][0], 2) +
            std::pow(vertices[j][1] - vertices[i][1], 2)
        );
    }

    // Find longest and shortest edges
    double max_edge = *std::max_element(edges, edges + 3);
    double min_edge = *std::min_element(edges, edges + 3);

    return (min_edge > 1e-12) ? max_edge / min_edge : std::numeric_limits<double>::max();
}

double compute_skewness(const std::array<std::array<double, 2>, 3>& vertices) {
    // Compute area and perimeter
    double area = compute_triangle_area(vertices);

    double perimeter = 0.0;
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;
        perimeter += std::sqrt(
            std::pow(vertices[j][0] - vertices[i][0], 2) +
            std::pow(vertices[j][1] - vertices[i][1], 2)
        );
    }

    // Skewness = 1 - (4 * sqrt(3) * area) / (perimeter^2)
    // Perfect equilateral triangle has skewness = 0
    if (perimeter > 1e-12) {
        return 1.0 - (4.0 * std::sqrt(3.0) * area) / (perimeter * perimeter);
    }
    return 1.0;
}

std::vector<std::vector<int>> compute_element_neighbors(
    const std::vector<std::vector<int>>& elements) {

    std::vector<std::vector<int>> neighbors(elements.size());

    // Build edge-to-element mapping
    std::map<std::pair<int, int>, std::vector<int>> edge_to_elements;

    for (size_t e = 0; e < elements.size(); ++e) {
        const auto& element = elements[e];
        if (element.size() < 3) continue;

        for (int i = 0; i < 3; ++i) {
            int v1 = element[i];
            int v2 = element[(i + 1) % 3];

            // Ensure consistent edge ordering
            if (v1 > v2) std::swap(v1, v2);

            edge_to_elements[{v1, v2}].push_back(static_cast<int>(e));
        }
    }

    // Find neighbors through shared edges
    for (const auto& edge_pair : edge_to_elements) {
        const auto& edge_elements = edge_pair.second;

        if (edge_elements.size() == 2) {
            // Internal edge - elements are neighbors
            int e1 = edge_elements[0];
            int e2 = edge_elements[1];
            neighbors[e1].push_back(e2);
            neighbors[e2].push_back(e1);
        }
    }

    return neighbors;
}

std::array<double, 2> compute_element_gradient(
    const std::vector<double>& solution,
    const std::vector<int>& element,
    const std::vector<std::array<double, 2>>& vertices) {

    if (element.size() < 3 || solution.empty()) {
        return {0.0, 0.0};
    }

    // Get element vertices and solution values
    std::array<std::array<double, 2>, 3> elem_vertices;
    std::array<double, 3> elem_solution;

    for (int i = 0; i < 3; ++i) {
        if (element[i] >= static_cast<int>(vertices.size()) ||
            element[i] >= static_cast<int>(solution.size())) {
            return {0.0, 0.0};
        }
        elem_vertices[i] = vertices[element[i]];
        elem_solution[i] = solution[element[i]];
    }

    // Compute gradient using linear interpolation
    // ∇u = (u1-u0) * ∇N1 + (u2-u0) * ∇N2
    // where ∇N1 and ∇N2 are gradients of linear basis functions

    double x1 = elem_vertices[1][0] - elem_vertices[0][0];
    double y1 = elem_vertices[1][1] - elem_vertices[0][1];
    double x2 = elem_vertices[2][0] - elem_vertices[0][0];
    double y2 = elem_vertices[2][1] - elem_vertices[0][1];

    double det = x1 * y2 - x2 * y1;

    if (std::abs(det) < 1e-12) {
        return {0.0, 0.0};
    }

    double u1 = elem_solution[1] - elem_solution[0];
    double u2 = elem_solution[2] - elem_solution[0];

    double grad_x = (y2 * u1 - y1 * u2) / det;
    double grad_y = (-x2 * u1 + x1 * u2) / det;

    return {grad_x, grad_y};
}

} // namespace utils

// Advanced AMR implementations
std::vector<double> AMRController::compute_residual_based_error(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& nodes) {

    std::vector<double> errors(elements.size(), 0.0);

    for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
        const auto& element = elements[elem_idx];
        if (element.size() < 3) continue;

        // Get element vertices
        auto v1 = nodes[element[0]];
        auto v2 = nodes[element[1]];
        auto v3 = nodes[element[2]];

        // Compute element area
        double area = 0.5 * std::abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                                     (v3[0] - v1[0]) * (v2[1] - v1[1]));

        if (area < 1e-12) continue;

        // Get solution values at vertices
        double u1 = (element[0] < solution.size()) ? solution[element[0]] : 0.0;
        double u2 = (element[1] < solution.size()) ? solution[element[1]] : 0.0;
        double u3 = (element[2] < solution.size()) ? solution[element[2]] : 0.0;

        // Compute gradient-based error estimate
        double grad_x = ((u2 - u1) * (v3[1] - v1[1]) - (u3 - u1) * (v2[1] - v1[1])) / (2.0 * area);
        double grad_y = ((u3 - u1) * (v2[0] - v1[0]) - (u2 - u1) * (v3[0] - v1[0])) / (2.0 * area);

        // Element diameter
        double h = std::sqrt(area);

        // Residual-based error estimate: η = h * ||∇u||
        errors[elem_idx] = h * std::sqrt(grad_x * grad_x + grad_y * grad_y);
    }

    return errors;
}

std::vector<double> AMRController::compute_zz_error(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& nodes) {

    std::vector<double> errors(elements.size(), 0.0);

    // Zienkiewicz-Zhu error estimator based on gradient recovery
    std::vector<std::array<double, 2>> recovered_gradients(nodes.size(), {0.0, 0.0});
    std::vector<double> node_weights(nodes.size(), 0.0);

    // Step 1: Compute element gradients and project to nodes
    for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
        const auto& element = elements[elem_idx];
        if (element.size() < 3) continue;

        // Get element vertices
        auto v1 = nodes[element[0]];
        auto v2 = nodes[element[1]];
        auto v3 = nodes[element[2]];

        // Compute element area
        double area = 0.5 * std::abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                                     (v3[0] - v1[0]) * (v2[1] - v1[1]));

        if (area < 1e-12) continue;

        // Get solution values at vertices
        double u1 = (element[0] < solution.size()) ? solution[element[0]] : 0.0;
        double u2 = (element[1] < solution.size()) ? solution[element[1]] : 0.0;
        double u3 = (element[2] < solution.size()) ? solution[element[2]] : 0.0;

        // Compute element gradient
        double grad_x = ((u2 - u1) * (v3[1] - v1[1]) - (u3 - u1) * (v2[1] - v1[1])) / (2.0 * area);
        double grad_y = ((u3 - u1) * (v2[0] - v1[0]) - (u2 - u1) * (v3[0] - v1[0])) / (2.0 * area);

        // Project to nodes (area-weighted averaging)
        for (int i = 0; i < 3; ++i) {
            if (element[i] < recovered_gradients.size()) {
                recovered_gradients[element[i]][0] += area * grad_x;
                recovered_gradients[element[i]][1] += area * grad_y;
                node_weights[element[i]] += area;
            }
        }
    }

    // Step 2: Normalize recovered gradients
    for (size_t i = 0; i < recovered_gradients.size(); ++i) {
        if (node_weights[i] > 1e-12) {
            recovered_gradients[i][0] /= node_weights[i];
            recovered_gradients[i][1] /= node_weights[i];
        }
    }

    // Step 3: Compute error for each element
    for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
        const auto& element = elements[elem_idx];
        if (element.size() < 3) continue;

        // Get element vertices and area
        auto v1 = nodes[element[0]];
        auto v2 = nodes[element[1]];
        auto v3 = nodes[element[2]];

        double area = 0.5 * std::abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                                     (v3[0] - v1[0]) * (v2[1] - v1[1]));

        if (area < 1e-12) continue;

        // Compute element gradient
        double u1 = (element[0] < solution.size()) ? solution[element[0]] : 0.0;
        double u2 = (element[1] < solution.size()) ? solution[element[1]] : 0.0;
        double u3 = (element[2] < solution.size()) ? solution[element[2]] : 0.0;

        double elem_grad_x = ((u2 - u1) * (v3[1] - v1[1]) - (u3 - u1) * (v2[1] - v1[1])) / (2.0 * area);
        double elem_grad_y = ((u3 - u1) * (v2[0] - v1[0]) - (u2 - u1) * (v3[0] - v1[0])) / (2.0 * area);

        // Average recovered gradient at element centroid
        double recovered_grad_x = 0.0, recovered_grad_y = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (element[i] < recovered_gradients.size()) {
                recovered_grad_x += recovered_gradients[element[i]][0];
                recovered_grad_y += recovered_gradients[element[i]][1];
            }
        }
        recovered_grad_x /= 3.0;
        recovered_grad_y /= 3.0;

        // ZZ error estimate: ||∇u_h - ∇u*||
        double diff_x = elem_grad_x - recovered_grad_x;
        double diff_y = elem_grad_y - recovered_grad_y;
        errors[elem_idx] = std::sqrt(area) * std::sqrt(diff_x * diff_x + diff_y * diff_y);
    }

    return errors;
}

std::array<double, 2> AMRController::detect_anisotropy_direction(
    int element_id,
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices) {

    if (element_id < 0 || element_id >= static_cast<int>(elements.size())) {
        return {1.0, 0.0}; // Default horizontal direction
    }

    const auto& element = elements[element_id];
    if (element.size() < 3) {
        return {1.0, 0.0};
    }

    // Get element vertices
    auto v1 = vertices[element[0]];
    auto v2 = vertices[element[1]];
    auto v3 = vertices[element[2]];

    // Compute element area
    double area = 0.5 * std::abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                                 (v3[0] - v1[0]) * (v2[1] - v1[1]));

    if (area < 1e-12) {
        return {1.0, 0.0};
    }

    // Get solution values at vertices
    double u1 = (element[0] < solution.size()) ? solution[element[0]] : 0.0;
    double u2 = (element[1] < solution.size()) ? solution[element[1]] : 0.0;
    double u3 = (element[2] < solution.size()) ? solution[element[2]] : 0.0;

    // Compute gradient
    double grad_x = ((u2 - u1) * (v3[1] - v1[1]) - (u3 - u1) * (v2[1] - v1[1])) / (2.0 * area);
    double grad_y = ((u3 - u1) * (v2[0] - v1[0]) - (u2 - u1) * (v3[0] - v1[0])) / (2.0 * area);

    // Compute Hessian approximation using neighboring elements
    double hxx = 0.0, hxy = 0.0, hyy = 0.0;
    int neighbor_count = 0;

    // Find neighboring elements (improved approach using vertex connectivity)
    std::vector<size_t> neighbor_elements;

    for (size_t other_id = 0; other_id < elements.size(); ++other_id) {
        if (other_id == static_cast<size_t>(element_id)) continue;

        const auto& other_element = elements[other_id];
        if (other_element.size() < 3) continue;

        // Check if elements share at least 2 vertices (i.e., share an edge)
        int shared_vertices = 0;
        for (int v1 : element) {
            for (int v2 : other_element) {
                if (v1 == v2) shared_vertices++;
            }
        }

        if (shared_vertices >= 2) {
            neighbor_elements.push_back(other_id);
        }
    }

    // Process neighboring elements for Hessian computation
    for (size_t other_id : neighbor_elements) {
        const auto& other_element = elements[other_id];

        // Check if elements share an edge (have 2 common vertices)
        int common_vertices = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (element[i] == other_element[j]) {
                    common_vertices++;
                    break;
                }
            }
        }

        if (common_vertices >= 2) {
            // Compute gradient difference for Hessian approximation
            auto ov1 = vertices[other_element[0]];
            auto ov2 = vertices[other_element[1]];
            auto ov3 = vertices[other_element[2]];

            double other_area = 0.5 * std::abs((ov2[0] - ov1[0]) * (ov3[1] - ov1[1]) -
                                               (ov3[0] - ov1[0]) * (ov2[1] - ov1[1]));

            if (other_area > 1e-12) {
                double ou1 = (other_element[0] < solution.size()) ? solution[other_element[0]] : 0.0;
                double ou2 = (other_element[1] < solution.size()) ? solution[other_element[1]] : 0.0;
                double ou3 = (other_element[2] < solution.size()) ? solution[other_element[2]] : 0.0;

                double other_grad_x = ((ou2 - ou1) * (ov3[1] - ov1[1]) - (ou3 - ou1) * (ov2[1] - ov1[1])) / (2.0 * other_area);
                double other_grad_y = ((ou3 - ou1) * (ov2[0] - ov1[0]) - (ou2 - ou1) * (ov3[0] - ov1[0])) / (2.0 * other_area);

                // Element centroids
                double cx1 = (v1[0] + v2[0] + v3[0]) / 3.0;
                double cy1 = (v1[1] + v2[1] + v3[1]) / 3.0;
                double cx2 = (ov1[0] + ov2[0] + ov3[0]) / 3.0;
                double cy2 = (ov1[1] + ov2[1] + ov3[1]) / 3.0;

                double dx = cx2 - cx1;
                double dy = cy2 - cy1;
                double dist = std::sqrt(dx*dx + dy*dy);

                if (dist > 1e-12) {
                    // Approximate second derivatives
                    double dgx = other_grad_x - grad_x;
                    double dgy = other_grad_y - grad_y;

                    hxx += dgx * dx / (dist * dist);
                    hxy += 0.5 * (dgx * dy + dgy * dx) / (dist * dist);
                    hyy += dgy * dy / (dist * dist);
                    neighbor_count++;
                }
            }
        }
    }

    if (neighbor_count > 0) {
        hxx /= neighbor_count;
        hxy /= neighbor_count;
        hyy /= neighbor_count;
    }

    // Compute eigenvalues of Hessian to determine anisotropy direction
    double trace = hxx + hyy;
    double det = hxx * hyy - hxy * hxy;
    double discriminant = trace * trace - 4.0 * det;

    if (discriminant >= 0) {
        double lambda1 = 0.5 * (trace + std::sqrt(discriminant));
        double lambda2 = 0.5 * (trace - std::sqrt(discriminant));

        // Direction of maximum curvature (eigenvector of larger eigenvalue)
        if (std::abs(lambda1) > std::abs(lambda2)) {
            if (std::abs(hxy) > 1e-12) {
                double norm = std::sqrt(hxy*hxy + (lambda1 - hyy)*(lambda1 - hyy));
                return {hxy / norm, (lambda1 - hyy) / norm};
            } else if (std::abs(hxx - lambda1) > 1e-12) {
                return {1.0, 0.0};
            } else {
                return {0.0, 1.0};
            }
        }
    }

    // Default: direction of gradient (steepest descent)
    double grad_norm = std::sqrt(grad_x*grad_x + grad_y*grad_y);
    if (grad_norm > 1e-12) {
        return {grad_x / grad_norm, grad_y / grad_norm};
    }

    return {1.0, 0.0}; // Default horizontal direction
}

std::unordered_map<int, std::vector<int>> AMRController::perform_refinement(
    const std::vector<ElementRefinement>& refinement_decisions,
    std::vector<std::vector<int>>& elements,
    std::vector<std::array<double, 2>>& vertices,
    std::vector<double>& solution) {

    std::unordered_map<int, std::vector<int>> parent_to_children;

    // Track new vertices and elements
    std::vector<std::array<double, 2>> new_vertices;
    std::vector<std::vector<int>> new_elements;
    std::vector<double> new_solution;

    // Copy existing vertices and solution
    new_vertices = vertices;
    new_solution = solution;

    // Process each element for refinement
    for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
        const auto& decision = refinement_decisions[elem_idx];

        if (decision.refine && elements[elem_idx].size() == 3) {
            // Refine triangular element by subdividing into 4 triangles
            const auto& element = elements[elem_idx];

            // Get vertex coordinates
            auto v0 = vertices[element[0]];
            auto v1 = vertices[element[1]];
            auto v2 = vertices[element[2]];

            // Create midpoint vertices
            std::array<double, 2> m01 = {(v0[0] + v1[0]) / 2.0, (v0[1] + v1[1]) / 2.0};
            std::array<double, 2> m12 = {(v1[0] + v2[0]) / 2.0, (v1[1] + v2[1]) / 2.0};
            std::array<double, 2> m20 = {(v2[0] + v0[0]) / 2.0, (v2[1] + v0[1]) / 2.0};

            // Add new vertices
            int idx_m01 = new_vertices.size();
            new_vertices.push_back(m01);
            int idx_m12 = new_vertices.size();
            new_vertices.push_back(m12);
            int idx_m20 = new_vertices.size();
            new_vertices.push_back(m20);

            // Create 4 child elements
            std::vector<int> child_indices;

            // Child 0: corner at v0
            new_elements.push_back({element[0], idx_m01, idx_m20});
            child_indices.push_back(new_elements.size() - 1);

            // Child 1: corner at v1
            new_elements.push_back({element[1], idx_m12, idx_m01});
            child_indices.push_back(new_elements.size() - 1);

            // Child 2: corner at v2
            new_elements.push_back({element[2], idx_m20, idx_m12});
            child_indices.push_back(new_elements.size() - 1);

            // Child 3: central triangle
            new_elements.push_back({idx_m01, idx_m12, idx_m20});
            child_indices.push_back(new_elements.size() - 1);

            // Store parent-child relationship
            parent_to_children[elem_idx] = child_indices;

            // Interpolate solution to new vertices (simple averaging)
            if (elem_idx < solution.size()) {
                double parent_value = solution[elem_idx];
                // Extend solution for new elements
                for (int i = 0; i < 4; ++i) {
                    new_solution.push_back(parent_value);
                }
            }
        } else {
            // Keep element unchanged
            new_elements.push_back(elements[elem_idx]);
        }
    }

    // Update mesh data
    vertices = std::move(new_vertices);
    elements = std::move(new_elements);
    solution = std::move(new_solution);

    return parent_to_children;
}

} // namespace amr
} // namespace simulator
