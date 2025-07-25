#pragma once

#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace simulator {
namespace amr {

/**
 * @brief Advanced error estimator types
 */
enum class AdvancedErrorEstimatorType {
    GRADIENT_RECOVERY,      ///< Zienkiewicz-Zhu gradient recovery
    HIERARCHICAL_BASIS,     ///< Hierarchical basis error estimation
    DUAL_WEIGHTED_RESIDUAL, ///< Goal-oriented error estimation
    FEATURE_DETECTION,      ///< Feature-based refinement
    PHYSICS_BASED,          ///< Physics-specific error indicators
    HYBRID                  ///< Combination of multiple estimators
};

/**
 * @brief Anisotropic refinement directions
 */
enum class RefinementDirection {
    ISOTROPIC,     ///< Uniform refinement in all directions
    X_DIRECTION,   ///< Refinement primarily in X direction
    Y_DIRECTION,   ///< Refinement primarily in Y direction
    ADAPTIVE       ///< Direction determined by error analysis
};

/**
 * @brief Element refinement information
 */
struct AdvancedRefinementInfo {
    bool refine = false;
    bool coarsen = false;
    RefinementDirection direction = RefinementDirection::ISOTROPIC;
    double anisotropy_ratio = 1.0;  ///< Aspect ratio for anisotropic refinement
    int target_level = 0;
    double error_indicator = 0.0;
    std::array<double, 2> gradient_direction = {0.0, 0.0};
    bool is_boundary_layer = false;
    double feature_strength = 0.0;
};

/**
 * @brief Mesh quality metrics
 */
struct MeshQualityMetrics {
    double min_angle = 0.0;
    double max_angle = 0.0;
    double min_aspect_ratio = 0.0;
    double max_aspect_ratio = 0.0;
    double min_element_size = 0.0;
    double max_element_size = 0.0;
    double average_quality = 0.0;
    size_t num_poor_quality_elements = 0;
    double mesh_regularity = 0.0;
};

/**
 * @brief Advanced adaptive mesh refinement controller
 */
class AdvancedAMRController {
public:
    AdvancedAMRController();
    
    // Configuration
    void set_error_estimator(AdvancedErrorEstimatorType type);
    void set_refinement_parameters(double refine_fraction, double coarsen_fraction,
                                 double error_tolerance);
    void set_anisotropic_parameters(bool enable, double max_anisotropy_ratio,
                                  double boundary_layer_threshold);
    void set_feature_detection_parameters(double feature_threshold, 
                                        double gradient_threshold);
    void set_physics_parameters(const std::string& physics_type,
                              const std::vector<std::string>& solution_variables);
    
    // Error estimation
    std::vector<double> estimate_error(const std::vector<double>& solution,
                                     const std::vector<std::vector<int>>& elements,
                                     const std::vector<std::array<double, 2>>& vertices,
                                     const std::vector<std::string>& variable_names = {});
    
    // Refinement decision
    std::vector<AdvancedRefinementInfo> determine_refinement(
        const std::vector<double>& error_indicators,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices,
        const std::vector<int>& current_levels = {});
    
    // Mesh quality analysis
    MeshQualityMetrics analyze_mesh_quality(
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    // Feature detection
    std::vector<double> detect_features(const std::vector<double>& solution,
                                      const std::vector<std::vector<int>>& elements,
                                      const std::vector<std::array<double, 2>>& vertices);
    
    // Boundary layer detection
    std::vector<bool> detect_boundary_layers(
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices,
        const std::vector<int>& boundary_nodes);
    
    // Load balancing for parallel execution
    std::vector<int> compute_load_balancing(
        const std::vector<std::vector<int>>& elements,
        const std::vector<AdvancedRefinementInfo>& refinement_info,
        int num_processors);
    
    // Statistics and monitoring
    struct RefinementStatistics {
        size_t total_elements_refined = 0;
        size_t total_elements_coarsened = 0;
        size_t anisotropic_refinements = 0;
        double average_error_reduction = 0.0;
        double mesh_quality_improvement = 0.0;
        double computational_cost_increase = 0.0;
    };
    
    RefinementStatistics get_refinement_statistics() const { return stats_; }
    void reset_statistics() { stats_ = RefinementStatistics{}; }
    
private:
    // Configuration parameters
    AdvancedErrorEstimatorType estimator_type_;
    double refine_fraction_;
    double coarsen_fraction_;
    double error_tolerance_;
    bool anisotropic_enabled_;
    double max_anisotropy_ratio_;
    double boundary_layer_threshold_;
    double feature_threshold_;
    double gradient_threshold_;
    std::string physics_type_;
    std::vector<std::string> solution_variables_;
    
    // Statistics
    mutable RefinementStatistics stats_;
    
    // Error estimation methods
    std::vector<double> compute_gradient_recovery_error(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    std::vector<double> compute_hierarchical_error(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    std::vector<double> compute_dual_weighted_residual_error(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    std::vector<double> compute_physics_based_error(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    // Anisotropic refinement analysis
    RefinementDirection analyze_anisotropy(
        const std::vector<double>& solution,
        const std::vector<int>& element,
        const std::vector<std::array<double, 2>>& vertices);
    
    double compute_anisotropy_ratio(
        const std::array<double, 2>& gradient_direction,
        const std::vector<int>& element,
        const std::vector<std::array<double, 2>>& vertices);
    
    // Utility functions
    std::array<double, 2> compute_element_gradient(
        const std::vector<double>& solution,
        const std::vector<int>& element,
        const std::vector<std::array<double, 2>>& vertices);
    
    double compute_element_size(const std::vector<int>& element,
                              const std::vector<std::array<double, 2>>& vertices);
    
    double compute_element_quality(const std::vector<int>& element,
                                 const std::vector<std::array<double, 2>>& vertices);
    
    bool is_near_boundary(const std::vector<int>& element,
                         const std::vector<int>& boundary_nodes);
};

/**
 * @brief Mesh refinement executor
 */
class MeshRefinementExecutor {
public:
    MeshRefinementExecutor();
    
    // Execute refinement
    struct RefinementResult {
        std::vector<std::vector<int>> new_elements;
        std::vector<std::array<double, 2>> new_vertices;
        std::vector<int> element_levels;
        std::vector<int> parent_mapping;  ///< Maps new elements to original parents
        std::vector<int> vertex_mapping;  ///< Maps new vertices to original vertices
        bool success = false;
        std::string error_message;
    };
    
    RefinementResult execute_refinement(
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices,
        const std::vector<AdvancedRefinementInfo>& refinement_info);
    
    // Coarsening
    RefinementResult execute_coarsening(
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices,
        const std::vector<AdvancedRefinementInfo>& refinement_info);
    
    // Solution transfer
    std::vector<double> transfer_solution(
        const std::vector<double>& old_solution,
        const RefinementResult& refinement_result,
        const std::string& transfer_method = "interpolation");
    
private:
    // Refinement operations
    std::vector<std::vector<int>> refine_element_isotropic(
        const std::vector<int>& element,
        const std::vector<std::array<double, 2>>& vertices,
        std::vector<std::array<double, 2>>& new_vertices);
    
    std::vector<std::vector<int>> refine_element_anisotropic(
        const std::vector<int>& element,
        const std::vector<std::array<double, 2>>& vertices,
        RefinementDirection direction,
        double anisotropy_ratio,
        std::vector<std::array<double, 2>>& new_vertices);
    
    // Coarsening operations
    bool can_coarsen_elements(const std::vector<int>& element_group,
                            const std::vector<AdvancedRefinementInfo>& refinement_info);
    
    std::vector<int> coarsen_element_group(
        const std::vector<int>& element_group,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    // Utility functions
    int add_vertex_if_new(const std::array<double, 2>& vertex,
                         std::vector<std::array<double, 2>>& vertices,
                         std::unordered_map<std::string, int>& vertex_map);
    
    std::string vertex_key(const std::array<double, 2>& vertex);
    
    void ensure_mesh_conformity(std::vector<std::vector<int>>& elements,
                               std::vector<std::array<double, 2>>& vertices);
};

} // namespace amr
} // namespace simulator
