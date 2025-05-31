#include "amr_algorithms.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <unordered_set>

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
            // Simplified curvature estimation
            if (e > 0) {
                auto grad_prev = utils::compute_element_gradient(solution, elements[e-1], vertices);
                curvature += std::abs(gradient[0] - grad_prev[0]) + std::abs(gradient[1] - grad_prev[1]);
            }
            error_indicators[e] += h * h * curvature;
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
                        
                        // Face normal (simplified)
                        double nx = -(vertices[v2][1] - vertices[v1][1]) / face_length;
                        double ny = (vertices[v2][0] - vertices[v1][0]) / face_length;
                        
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

} // namespace amr
} // namespace simulator
