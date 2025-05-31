#pragma once

#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace simulator {
namespace amr {

/**
 * @brief Error estimator types for adaptive mesh refinement
 */
enum class ErrorEstimatorType {
    GRADIENT_BASED,      ///< Gradient-based error estimation
    RESIDUAL_BASED,      ///< Residual-based error estimation
    RECOVERY_BASED,      ///< Recovery-based error estimation
    HIERARCHICAL,        ///< Hierarchical basis error estimation
    KELLY,              ///< Kelly error estimator
    ZZ_SUPERCONVERGENT  ///< Zienkiewicz-Zhu superconvergent patch recovery
};

/**
 * @brief Refinement strategy types
 */
enum class RefinementStrategy {
    FIXED_FRACTION,     ///< Refine fixed fraction of elements
    FIXED_NUMBER,       ///< Refine fixed number of elements
    EQUILIBRATION,      ///< Error equilibration strategy
    ADAPTIVE_THRESHOLD, ///< Adaptive threshold based on error distribution
    ANISOTROPIC        ///< Anisotropic refinement for boundary layers
};

/**
 * @brief Element refinement information
 */
struct ElementRefinement {
    int element_id;                    ///< Global element ID
    bool refine;                       ///< Whether to refine this element
    bool coarsen;                      ///< Whether to coarsen this element
    double error_indicator;            ///< Local error indicator
    std::array<double, 2> anisotropy; ///< Anisotropy direction (for anisotropic refinement)
    int refinement_level;              ///< Current refinement level
};

/**
 * @brief Mesh quality metrics
 */
struct MeshQuality {
    double min_angle;          ///< Minimum angle in mesh
    double max_angle;          ///< Maximum angle in mesh
    double min_area;           ///< Minimum element area
    double max_area;           ///< Maximum element area
    double aspect_ratio_min;   ///< Minimum aspect ratio
    double aspect_ratio_max;   ///< Maximum aspect ratio
    double skewness_max;       ///< Maximum skewness
    int num_bad_elements;      ///< Number of poor quality elements
};

/**
 * @brief Advanced AMR controller class
 */
class AMRController {
public:
    /**
     * @brief Constructor
     * 
     * @param estimator_type Type of error estimator to use
     * @param strategy Refinement strategy
     * @param max_refinement_levels Maximum allowed refinement levels
     */
    AMRController(ErrorEstimatorType estimator_type = ErrorEstimatorType::GRADIENT_BASED,
                  RefinementStrategy strategy = RefinementStrategy::FIXED_FRACTION,
                  int max_refinement_levels = 5);

    /**
     * @brief Set refinement parameters
     * 
     * @param refine_fraction Fraction of elements to refine (0.0-1.0)
     * @param coarsen_fraction Fraction of elements to coarsen (0.0-1.0)
     * @param error_tolerance Global error tolerance
     * @param min_element_size Minimum allowed element size
     * @param max_element_size Maximum allowed element size
     */
    void set_refinement_parameters(double refine_fraction = 0.3,
                                  double coarsen_fraction = 0.1,
                                  double error_tolerance = 1e-3,
                                  double min_element_size = 1e-6,
                                  double max_element_size = 1e-2);

    /**
     * @brief Compute error indicators for all elements
     * 
     * @param solution Solution vector
     * @param elements Element connectivity
     * @param vertices Vertex coordinates
     * @param solution_gradient Gradient of solution (optional)
     * @return Vector of error indicators per element
     */
    std::vector<double> compute_error_indicators(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices,
        const std::vector<std::array<double, 2>>& solution_gradient = {});

    /**
     * @brief Determine refinement flags based on error indicators
     * 
     * @param error_indicators Error indicators per element
     * @param elements Element connectivity
     * @param vertices Vertex coordinates
     * @param current_levels Current refinement levels per element
     * @return Vector of refinement decisions
     */
    std::vector<ElementRefinement> determine_refinement(
        const std::vector<double>& error_indicators,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices,
        const std::vector<int>& current_levels = {});

    /**
     * @brief Perform mesh refinement/coarsening
     * 
     * @param refinement_decisions Refinement decisions per element
     * @param elements Element connectivity (modified)
     * @param vertices Vertex coordinates (modified)
     * @param solution Solution vector (interpolated to new mesh)
     * @return Mapping from old to new element indices
     */
    std::unordered_map<int, std::vector<int>> perform_refinement(
        const std::vector<ElementRefinement>& refinement_decisions,
        std::vector<std::vector<int>>& elements,
        std::vector<std::array<double, 2>>& vertices,
        std::vector<double>& solution);

    /**
     * @brief Compute mesh quality metrics
     * 
     * @param elements Element connectivity
     * @param vertices Vertex coordinates
     * @return Mesh quality metrics
     */
    MeshQuality compute_mesh_quality(
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices) const;

    /**
     * @brief Enable/disable anisotropic refinement
     * 
     * @param enable Whether to enable anisotropic refinement
     * @param boundary_layer_detection Whether to detect boundary layers
     */
    void set_anisotropic_refinement(bool enable, bool boundary_layer_detection = true);

    /**
     * @brief Get refinement statistics
     * 
     * @return Statistics about last refinement operation
     */
    struct RefinementStats {
        int elements_refined;
        int elements_coarsened;
        int new_vertices_created;
        double total_error_before;
        double total_error_after;
        double refinement_efficiency;
        MeshQuality quality_before;
        MeshQuality quality_after;
    };
    
    RefinementStats get_last_refinement_stats() const { return last_stats_; }

private:
    ErrorEstimatorType estimator_type_;
    RefinementStrategy strategy_;
    int max_refinement_levels_;
    
    // Refinement parameters
    double refine_fraction_;
    double coarsen_fraction_;
    double error_tolerance_;
    double min_element_size_;
    double max_element_size_;
    
    // Anisotropic refinement settings
    bool anisotropic_enabled_;
    bool boundary_layer_detection_;
    
    // Statistics
    RefinementStats last_stats_;
    
    // Private methods for different error estimators
    std::vector<double> compute_gradient_based_error(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    std::vector<double> compute_residual_based_error(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    std::vector<double> compute_kelly_error(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    std::vector<double> compute_zz_error(
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    // Anisotropic refinement detection
    std::array<double, 2> detect_anisotropy_direction(
        int element_id,
        const std::vector<double>& solution,
        const std::vector<std::vector<int>>& elements,
        const std::vector<std::array<double, 2>>& vertices);
    
    // Mesh operations
    void refine_triangle(int element_id,
                        std::vector<std::vector<int>>& elements,
                        std::vector<std::array<double, 2>>& vertices,
                        std::unordered_map<int, std::vector<int>>& element_mapping);
    
    void coarsen_elements(const std::vector<int>& element_ids,
                         std::vector<std::vector<int>>& elements,
                         std::vector<std::array<double, 2>>& vertices);
    
    // Solution interpolation
    void interpolate_solution(const std::vector<double>& old_solution,
                             std::vector<double>& new_solution,
                             const std::unordered_map<int, std::vector<int>>& element_mapping,
                             const std::vector<std::vector<int>>& old_elements,
                             const std::vector<std::vector<int>>& new_elements,
                             const std::vector<std::array<double, 2>>& vertices);
};

/**
 * @brief Utility functions for AMR
 */
namespace utils {

/**
 * @brief Compute element area
 */
double compute_triangle_area(const std::array<std::array<double, 2>, 3>& vertices);

/**
 * @brief Compute element aspect ratio
 */
double compute_aspect_ratio(const std::array<std::array<double, 2>, 3>& vertices);

/**
 * @brief Compute element skewness
 */
double compute_skewness(const std::array<std::array<double, 2>, 3>& vertices);

/**
 * @brief Find element neighbors
 */
std::vector<std::vector<int>> compute_element_neighbors(
    const std::vector<std::vector<int>>& elements);

/**
 * @brief Compute solution gradient at element centroid
 */
std::array<double, 2> compute_element_gradient(
    const std::vector<double>& solution,
    const std::vector<int>& element,
    const std::vector<std::array<double, 2>>& vertices);

} // namespace utils

} // namespace amr
} // namespace simulator
