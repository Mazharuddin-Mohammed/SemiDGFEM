#include "advanced_mesh_refinement.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <limits>

namespace simulator {
namespace amr {

// ============================================================================
// AdvancedAMRController Implementation
// ============================================================================

AdvancedAMRController::AdvancedAMRController()
    : estimator_type_(AdvancedErrorEstimatorType::GRADIENT_RECOVERY),
      refine_fraction_(0.3), coarsen_fraction_(0.1), error_tolerance_(1e-3),
      anisotropic_enabled_(false), max_anisotropy_ratio_(10.0),
      boundary_layer_threshold_(0.1), feature_threshold_(0.5),
      gradient_threshold_(1e-3), physics_type_("semiconductor") {
}

void AdvancedAMRController::set_error_estimator(AdvancedErrorEstimatorType type) {
    estimator_type_ = type;
}

void AdvancedAMRController::set_refinement_parameters(double refine_fraction, 
                                                    double coarsen_fraction,
                                                    double error_tolerance) {
    if (refine_fraction < 0.0 || refine_fraction > 1.0 ||
        coarsen_fraction < 0.0 || coarsen_fraction > 1.0) {
        throw std::invalid_argument("Refinement fractions must be between 0 and 1");
    }
    
    refine_fraction_ = refine_fraction;
    coarsen_fraction_ = coarsen_fraction;
    error_tolerance_ = error_tolerance;
}

void AdvancedAMRController::set_anisotropic_parameters(bool enable, 
                                                     double max_anisotropy_ratio,
                                                     double boundary_layer_threshold) {
    anisotropic_enabled_ = enable;
    max_anisotropy_ratio_ = std::max(1.0, max_anisotropy_ratio);
    boundary_layer_threshold_ = boundary_layer_threshold;
}

void AdvancedAMRController::set_feature_detection_parameters(double feature_threshold,
                                                           double gradient_threshold) {
    feature_threshold_ = feature_threshold;
    gradient_threshold_ = gradient_threshold;
}

void AdvancedAMRController::set_physics_parameters(const std::string& physics_type,
                                                 const std::vector<std::string>& solution_variables) {
    physics_type_ = physics_type;
    solution_variables_ = solution_variables;
}

std::vector<double> AdvancedAMRController::estimate_error(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices,
    const std::vector<std::string>& variable_names) {
    
    if (solution.empty() || elements.empty() || vertices.empty()) {
        return std::vector<double>(elements.size(), 0.0);
    }
    
    switch (estimator_type_) {
        case AdvancedErrorEstimatorType::GRADIENT_RECOVERY:
            return compute_gradient_recovery_error(solution, elements, vertices);
            
        case AdvancedErrorEstimatorType::HIERARCHICAL_BASIS:
            return compute_hierarchical_error(solution, elements, vertices);
            
        case AdvancedErrorEstimatorType::DUAL_WEIGHTED_RESIDUAL:
            return compute_dual_weighted_residual_error(solution, elements, vertices);
            
        case AdvancedErrorEstimatorType::FEATURE_DETECTION: {
            auto feature_errors = detect_features(solution, elements, vertices);
            return feature_errors;
        }
        
        case AdvancedErrorEstimatorType::PHYSICS_BASED:
            return compute_physics_based_error(solution, elements, vertices);
            
        case AdvancedErrorEstimatorType::HYBRID: {
            // Combine multiple error estimators
            auto gradient_errors = compute_gradient_recovery_error(solution, elements, vertices);
            auto feature_errors = detect_features(solution, elements, vertices);
            auto physics_errors = compute_physics_based_error(solution, elements, vertices);
            
            std::vector<double> combined_errors(elements.size());
            for (size_t i = 0; i < elements.size(); ++i) {
                combined_errors[i] = 0.4 * gradient_errors[i] + 
                                   0.3 * feature_errors[i] + 
                                   0.3 * physics_errors[i];
            }
            return combined_errors;
        }
        
        default:
            return compute_gradient_recovery_error(solution, elements, vertices);
    }
}

std::vector<AdvancedRefinementInfo> AdvancedAMRController::determine_refinement(
    const std::vector<double>& error_indicators,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices,
    const std::vector<int>& current_levels) {
    
    std::vector<AdvancedRefinementInfo> refinement_info(elements.size());
    
    if (error_indicators.empty()) {
        return refinement_info;
    }
    
    // Sort elements by error
    std::vector<size_t> sorted_indices(elements.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&error_indicators](size_t a, size_t b) {
                  return error_indicators[a] > error_indicators[b];
              });
    
    // Compute thresholds
    double max_error = *std::max_element(error_indicators.begin(), error_indicators.end());
    double mean_error = std::accumulate(error_indicators.begin(), error_indicators.end(), 0.0) 
                       / error_indicators.size();
    
    double refine_threshold = std::max(error_tolerance_, 0.5 * max_error);
    double coarsen_threshold = 0.1 * mean_error;
    
    // Detect boundary layers if enabled
    std::vector<bool> boundary_layer_elements;
    if (anisotropic_enabled_) {
        // For simplicity, assume boundary nodes are those with indices < 10% of total
        std::vector<int> boundary_nodes;
        size_t boundary_count = std::max(size_t(1), vertices.size() / 10);
        for (size_t i = 0; i < boundary_count; ++i) {
            boundary_nodes.push_back(static_cast<int>(i));
        }
        boundary_layer_elements = detect_boundary_layers(elements, vertices, boundary_nodes);
    }
    
    // Determine refinement for each element
    for (size_t e = 0; e < elements.size(); ++e) {
        refinement_info[e].error_indicator = error_indicators[e];
        
        // Refinement decision
        if (error_indicators[e] > refine_threshold) {
            refinement_info[e].refine = true;
            
            // Anisotropic analysis
            if (anisotropic_enabled_) {
                refinement_info[e].direction = analyze_anisotropy(
                    std::vector<double>(solution.begin(), solution.end()),
                    elements[e], vertices);
                
                if (refinement_info[e].direction != RefinementDirection::ISOTROPIC) {
                    refinement_info[e].anisotropy_ratio = compute_anisotropy_ratio(
                        refinement_info[e].gradient_direction, elements[e], vertices);
                    stats_.anisotropic_refinements++;
                }
                
                // Check for boundary layer
                if (e < boundary_layer_elements.size() && boundary_layer_elements[e]) {
                    refinement_info[e].is_boundary_layer = true;
                    refinement_info[e].direction = RefinementDirection::Y_DIRECTION; // Normal to boundary
                    refinement_info[e].anisotropy_ratio = std::min(max_anisotropy_ratio_, 5.0);
                }
            }
            
            stats_.total_elements_refined++;
        }
        // Coarsening decision
        else if (error_indicators[e] < coarsen_threshold) {
            refinement_info[e].coarsen = true;
            stats_.total_elements_coarsened++;
        }
        
        // Feature detection
        if (estimator_type_ == AdvancedErrorEstimatorType::FEATURE_DETECTION ||
            estimator_type_ == AdvancedErrorEstimatorType::HYBRID) {
            auto gradient = compute_element_gradient(
                std::vector<double>(solution.begin(), solution.end()),
                elements[e], vertices);
            double gradient_magnitude = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);
            refinement_info[e].feature_strength = gradient_magnitude;
            
            if (gradient_magnitude > feature_threshold_) {
                refinement_info[e].refine = true;
                refinement_info[e].direction = RefinementDirection::ADAPTIVE;
            }
        }
    }
    
    return refinement_info;
}

MeshQualityMetrics AdvancedAMRController::analyze_mesh_quality(
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices) {
    
    MeshQualityMetrics metrics;
    
    if (elements.empty() || vertices.empty()) {
        return metrics;
    }
    
    std::vector<double> angles, aspect_ratios, element_sizes, qualities;
    angles.reserve(elements.size() * 3);
    aspect_ratios.reserve(elements.size());
    element_sizes.reserve(elements.size());
    qualities.reserve(elements.size());
    
    for (const auto& element : elements) {
        if (element.size() != 3) continue; // Only triangular elements
        
        // Get vertices
        auto v1 = vertices[element[0]];
        auto v2 = vertices[element[1]];
        auto v3 = vertices[element[2]];
        
        // Compute edge lengths
        double a = std::sqrt(std::pow(v2[0] - v3[0], 2) + std::pow(v2[1] - v3[1], 2));
        double b = std::sqrt(std::pow(v1[0] - v3[0], 2) + std::pow(v1[1] - v3[1], 2));
        double c = std::sqrt(std::pow(v1[0] - v2[0], 2) + std::pow(v1[1] - v2[1], 2));
        
        // Compute angles using law of cosines
        double angle1 = std::acos(std::max(-1.0, std::min(1.0, (b*b + c*c - a*a) / (2*b*c))));
        double angle2 = std::acos(std::max(-1.0, std::min(1.0, (a*a + c*c - b*b) / (2*a*c))));
        double angle3 = M_PI - angle1 - angle2;
        
        angles.push_back(angle1 * 180.0 / M_PI);
        angles.push_back(angle2 * 180.0 / M_PI);
        angles.push_back(angle3 * 180.0 / M_PI);
        
        // Compute aspect ratio
        double max_edge = std::max({a, b, c});
        double min_edge = std::min({a, b, c});
        double aspect_ratio = max_edge / min_edge;
        aspect_ratios.push_back(aspect_ratio);
        
        // Compute element size (area)
        double area = 0.5 * std::abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - 
                                    (v3[0] - v1[0]) * (v2[1] - v1[1]));
        element_sizes.push_back(area);
        
        // Compute quality (ratio of inscribed to circumscribed circle radii)
        double perimeter = a + b + c;
        double inradius = area / (0.5 * perimeter);
        double circumradius = (a * b * c) / (4.0 * area);
        double quality = (circumradius > 0) ? inradius / circumradius : 0.0;
        qualities.push_back(quality);
    }
    
    // Compute metrics
    if (!angles.empty()) {
        metrics.min_angle = *std::min_element(angles.begin(), angles.end());
        metrics.max_angle = *std::max_element(angles.begin(), angles.end());
    }
    
    if (!aspect_ratios.empty()) {
        metrics.min_aspect_ratio = *std::min_element(aspect_ratios.begin(), aspect_ratios.end());
        metrics.max_aspect_ratio = *std::max_element(aspect_ratios.begin(), aspect_ratios.end());
        
        // Count poor quality elements (aspect ratio > 5)
        metrics.num_poor_quality_elements = std::count_if(aspect_ratios.begin(), aspect_ratios.end(),
                                                         [](double ar) { return ar > 5.0; });
    }
    
    if (!element_sizes.empty()) {
        metrics.min_element_size = *std::min_element(element_sizes.begin(), element_sizes.end());
        metrics.max_element_size = *std::max_element(element_sizes.begin(), element_sizes.end());
    }
    
    if (!qualities.empty()) {
        metrics.average_quality = std::accumulate(qualities.begin(), qualities.end(), 0.0) / qualities.size();
        
        // Mesh regularity (standard deviation of quality)
        double mean_quality = metrics.average_quality;
        double variance = 0.0;
        for (double q : qualities) {
            variance += (q - mean_quality) * (q - mean_quality);
        }
        variance /= qualities.size();
        metrics.mesh_regularity = 1.0 / (1.0 + std::sqrt(variance));
    }
    
    return metrics;
}

std::vector<double> AdvancedAMRController::detect_features(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices) {

    std::vector<double> feature_indicators(elements.size(), 0.0);

    for (size_t e = 0; e < elements.size(); ++e) {
        auto gradient = compute_element_gradient(solution, elements[e], vertices);
        double gradient_magnitude = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);

        // Compute second derivatives (Hessian) for curvature detection
        double curvature = 0.0;
        if (e > 0 && e < elements.size() - 1) {
            auto prev_gradient = compute_element_gradient(solution, elements[e-1], vertices);
            auto next_gradient = compute_element_gradient(solution, elements[e+1], vertices);

            double grad_change_x = next_gradient[0] - prev_gradient[0];
            double grad_change_y = next_gradient[1] - prev_gradient[1];
            curvature = std::sqrt(grad_change_x*grad_change_x + grad_change_y*grad_change_y);
        }

        // Feature strength combines gradient magnitude and curvature
        feature_indicators[e] = gradient_magnitude + 0.5 * curvature;
    }

    return feature_indicators;
}

std::vector<bool> AdvancedAMRController::detect_boundary_layers(
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices,
    const std::vector<int>& boundary_nodes) {

    std::vector<bool> is_boundary_layer(elements.size(), false);
    std::unordered_set<int> boundary_set(boundary_nodes.begin(), boundary_nodes.end());

    for (size_t e = 0; e < elements.size(); ++e) {
        // Check if element has boundary nodes
        int boundary_node_count = 0;
        for (int node : elements[e]) {
            if (boundary_set.count(node)) {
                boundary_node_count++;
            }
        }

        // Element is in boundary layer if it has at least one boundary node
        if (boundary_node_count > 0) {
            // Additional check: element size should be small in normal direction
            double element_size = compute_element_size(elements[e], vertices);
            if (element_size < boundary_layer_threshold_) {
                is_boundary_layer[e] = true;
            }
        }
    }

    return is_boundary_layer;
}

std::vector<double> AdvancedAMRController::compute_gradient_recovery_error(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices) {

    std::vector<double> errors(elements.size(), 0.0);

    // Zienkiewicz-Zhu gradient recovery
    std::vector<std::array<double, 2>> recovered_gradients(vertices.size(), {0.0, 0.0});
    std::vector<double> node_weights(vertices.size(), 0.0);

    // Compute element gradients and project to nodes
    for (size_t e = 0; e < elements.size(); ++e) {
        auto element_gradient = compute_element_gradient(solution, elements[e], vertices);
        double element_area = compute_element_size(elements[e], vertices);

        // Distribute gradient to element nodes
        for (int node : elements[e]) {
            recovered_gradients[node][0] += element_gradient[0] * element_area;
            recovered_gradients[node][1] += element_gradient[1] * element_area;
            node_weights[node] += element_area;
        }
    }

    // Normalize by weights
    for (size_t n = 0; n < vertices.size(); ++n) {
        if (node_weights[n] > 0) {
            recovered_gradients[n][0] /= node_weights[n];
            recovered_gradients[n][1] /= node_weights[n];
        }
    }

    // Compute error as difference between element and recovered gradients
    for (size_t e = 0; e < elements.size(); ++e) {
        auto element_gradient = compute_element_gradient(solution, elements[e], vertices);

        // Average recovered gradient over element
        std::array<double, 2> avg_recovered = {0.0, 0.0};
        for (int node : elements[e]) {
            avg_recovered[0] += recovered_gradients[node][0];
            avg_recovered[1] += recovered_gradients[node][1];
        }
        avg_recovered[0] /= 3.0;
        avg_recovered[1] /= 3.0;

        // Error norm
        double error_x = element_gradient[0] - avg_recovered[0];
        double error_y = element_gradient[1] - avg_recovered[1];
        errors[e] = std::sqrt(error_x*error_x + error_y*error_y);
    }

    return errors;
}

std::vector<double> AdvancedAMRController::compute_physics_based_error(
    const std::vector<double>& solution,
    const std::vector<std::vector<int>>& elements,
    const std::vector<std::array<double, 2>>& vertices) {

    std::vector<double> errors(elements.size(), 0.0);

    if (physics_type_ == "semiconductor") {
        // Semiconductor-specific error indicators
        for (size_t e = 0; e < elements.size(); ++e) {
            auto gradient = compute_element_gradient(solution, elements[e], vertices);
            double gradient_magnitude = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);

            // High gradients indicate depletion regions, junctions, etc.
            double element_size = compute_element_size(elements[e], vertices);
            double characteristic_length = std::sqrt(element_size);

            // Error based on gradient and characteristic length
            errors[e] = gradient_magnitude * characteristic_length;

            // Additional physics-based criteria
            if (solution.size() >= elements[e].size()) {
                // Check for rapid solution changes (e.g., potential drops)
                double max_solution = -std::numeric_limits<double>::max();
                double min_solution = std::numeric_limits<double>::max();

                for (int node : elements[e]) {
                    if (node < static_cast<int>(solution.size())) {
                        max_solution = std::max(max_solution, solution[node]);
                        min_solution = std::min(min_solution, solution[node]);
                    }
                }

                double solution_variation = max_solution - min_solution;
                errors[e] += 0.5 * solution_variation / characteristic_length;
            }
        }
    }

    return errors;
}

// Utility function implementations
std::array<double, 2> AdvancedAMRController::compute_element_gradient(
    const std::vector<double>& solution,
    const std::vector<int>& element,
    const std::vector<std::array<double, 2>>& vertices) {

    if (element.size() != 3 || solution.size() <= static_cast<size_t>(*std::max_element(element.begin(), element.end()))) {
        return {0.0, 0.0};
    }

    // Get element vertices and solution values
    auto v1 = vertices[element[0]];
    auto v2 = vertices[element[1]];
    auto v3 = vertices[element[2]];

    double u1 = solution[element[0]];
    double u2 = solution[element[1]];
    double u3 = solution[element[2]];

    // Compute gradient using linear basis functions
    double det = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]);

    if (std::abs(det) < 1e-12) {
        return {0.0, 0.0};
    }

    double grad_x = ((v3[1] - v1[1]) * (u2 - u1) - (v2[1] - v1[1]) * (u3 - u1)) / det;
    double grad_y = ((v2[0] - v1[0]) * (u3 - u1) - (v3[0] - v1[0]) * (u2 - u1)) / det;

    return {grad_x, grad_y};
}

double AdvancedAMRController::compute_element_size(
    const std::vector<int>& element,
    const std::vector<std::array<double, 2>>& vertices) {

    if (element.size() != 3) return 0.0;

    auto v1 = vertices[element[0]];
    auto v2 = vertices[element[1]];
    auto v3 = vertices[element[2]];

    // Triangle area
    double area = 0.5 * std::abs((v2[0] - v1[0]) * (v3[1] - v1[1]) -
                                (v3[0] - v1[0]) * (v2[1] - v1[1]));
    return area;
}

RefinementDirection AdvancedAMRController::analyze_anisotropy(
    const std::vector<double>& solution,
    const std::vector<int>& element,
    const std::vector<std::array<double, 2>>& vertices) {

    auto gradient = compute_element_gradient(solution, element, vertices);
    double grad_magnitude = std::sqrt(gradient[0]*gradient[0] + gradient[1]*gradient[1]);

    if (grad_magnitude < gradient_threshold_) {
        return RefinementDirection::ISOTROPIC;
    }

    // Determine dominant gradient direction
    double angle = std::atan2(gradient[1], gradient[0]);
    double abs_angle = std::abs(angle);

    if (abs_angle < M_PI/4 || abs_angle > 3*M_PI/4) {
        return RefinementDirection::X_DIRECTION;
    } else {
        return RefinementDirection::Y_DIRECTION;
    }
}

double AdvancedAMRController::compute_anisotropy_ratio(
    const std::array<double, 2>& gradient_direction,
    const std::vector<int>& element,
    const std::vector<std::array<double, 2>>& vertices) {

    double grad_magnitude = std::sqrt(gradient_direction[0]*gradient_direction[0] +
                                    gradient_direction[1]*gradient_direction[1]);

    if (grad_magnitude < gradient_threshold_) {
        return 1.0;
    }

    // Simple anisotropy ratio based on gradient magnitude
    double ratio = std::min(max_anisotropy_ratio_, 1.0 + grad_magnitude * 10.0);
    return ratio;
}

} // namespace amr
} // namespace simulator
