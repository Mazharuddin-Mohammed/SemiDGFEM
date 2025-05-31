#include "driftdiffusion.hpp"
#include "mesh.hpp"
#include "amr_algorithms.hpp"
#include "performance_optimization.hpp"
#include <petscksp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <functional>

namespace simulator {

// PETSc objects structure for proper resource management
struct DriftDiffusion::PETScObjects {
    Mat A_n = nullptr, A_p = nullptr;
    Vec x_n = nullptr, x_p = nullptr;
    Vec b_n = nullptr, b_p = nullptr;
    KSP ksp_n = nullptr, ksp_p = nullptr;
    bool initialized = false;

    ~PETScObjects() {
        cleanup();
    }

    void cleanup() {
        if (ksp_n) { KSPDestroy(&ksp_n); ksp_n = nullptr; }
        if (ksp_p) { KSPDestroy(&ksp_p); ksp_p = nullptr; }
        if (A_n) { MatDestroy(&A_n); A_n = nullptr; }
        if (A_p) { MatDestroy(&A_p); A_p = nullptr; }
        if (x_n) { VecDestroy(&x_n); x_n = nullptr; }
        if (x_p) { VecDestroy(&x_p); x_p = nullptr; }
        if (b_n) { VecDestroy(&b_n); b_n = nullptr; }
        if (b_p) { VecDestroy(&b_p); b_p = nullptr; }
        initialized = false;
    }
};

DriftDiffusion::DriftDiffusion(const Device& device, Method method, MeshType mesh_type, int order)
    : device_(device), method_(method), mesh_type_(mesh_type), order_(order),
      poisson_(std::make_unique<Poisson>(device, method, mesh_type)),
      petsc_objects_(std::make_unique<PETScObjects>()),
      convergence_residual_(0.0) {

    if (!device.is_valid()) {
        throw std::invalid_argument("Invalid device provided to DriftDiffusion constructor");
    }

    if (method != Method::DG) {
        throw std::invalid_argument("Only DG method is currently supported");
    }

    validate_order();

    try {
        initialize_petsc();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize DriftDiffusion solver: " + std::string(e.what()));
    }
}

DriftDiffusion::~DriftDiffusion() {
    cleanup_petsc_objects();
}

DriftDiffusion::DriftDiffusion(const DriftDiffusion& other)
    : device_(other.device_), method_(other.method_), mesh_type_(other.mesh_type_),
      order_(other.order_), Nd_(other.Nd_), Na_(other.Na_), Et_(other.Et_),
      poisson_(std::make_unique<Poisson>(*other.poisson_)),
      petsc_objects_(std::make_unique<PETScObjects>()),
      convergence_residual_(other.convergence_residual_) {
    initialize_petsc();
}

DriftDiffusion& DriftDiffusion::operator=(const DriftDiffusion& other) {
    if (this != &other) {
        cleanup_petsc_objects();
        // Note: device_ is a const reference and cannot be reassigned
        method_ = other.method_;
        mesh_type_ = other.mesh_type_;
        order_ = other.order_;
        Nd_ = other.Nd_;
        Na_ = other.Na_;
        Et_ = other.Et_;
        poisson_ = std::make_unique<Poisson>(*other.poisson_);
        petsc_objects_ = std::make_unique<PETScObjects>();
        convergence_residual_ = other.convergence_residual_;
        initialize_petsc();
    }
    return *this;
}

DriftDiffusion::DriftDiffusion(DriftDiffusion&& other) noexcept
    : device_(other.device_), method_(other.method_), mesh_type_(other.mesh_type_),
      order_(other.order_), Nd_(std::move(other.Nd_)), Na_(std::move(other.Na_)),
      Et_(std::move(other.Et_)), poisson_(std::move(other.poisson_)),
      petsc_objects_(std::move(other.petsc_objects_)),
      convergence_residual_(other.convergence_residual_) {
}

DriftDiffusion& DriftDiffusion::operator=(DriftDiffusion&& other) noexcept {
    if (this != &other) {
        cleanup_petsc_objects();
        // Note: device_ is a const reference and cannot be reassigned
        method_ = other.method_;
        mesh_type_ = other.mesh_type_;
        order_ = other.order_;
        Nd_ = std::move(other.Nd_);
        Na_ = std::move(other.Na_);
        Et_ = std::move(other.Et_);
        poisson_ = std::move(other.poisson_);
        petsc_objects_ = std::move(other.petsc_objects_);
        convergence_residual_ = other.convergence_residual_;
    }
    return *this;
}

void DriftDiffusion::set_doping(const std::vector<double>& Nd, const std::vector<double>& Na) {
    if (Nd.empty() || Na.empty()) {
        throw std::invalid_argument("Doping arrays cannot be empty");
    }

    if (Nd.size() != Na.size()) {
        throw std::invalid_argument("Nd and Na arrays must have the same size");
    }

    if (std::any_of(Nd.begin(), Nd.end(), [](double x) { return x < 0 || !std::isfinite(x); }) ||
        std::any_of(Na.begin(), Na.end(), [](double x) { return x < 0 || !std::isfinite(x); })) {
        throw std::invalid_argument("Doping concentrations must be non-negative and finite");
    }

    Nd_ = Nd;
    Na_ = Na;
}

void DriftDiffusion::set_trap_level(const std::vector<double>& Et) {
    if (Et.empty()) {
        throw std::invalid_argument("Trap level array cannot be empty");
    }

    if (std::any_of(Et.begin(), Et.end(), [](double x) { return !std::isfinite(x); })) {
        throw std::invalid_argument("All trap levels must be finite");
    }

    Et_ = Et;
}

// Helper methods implementation
void DriftDiffusion::initialize_petsc() {
    static bool petsc_initialized = false;
    if (!petsc_initialized) {
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        petsc_initialized = true;
    }
    petsc_objects_->initialized = true;
}

void DriftDiffusion::cleanup_petsc_objects() {
    if (petsc_objects_) {
        petsc_objects_->cleanup();
    }
}

bool DriftDiffusion::is_valid() const {
    return device_.is_valid() && poisson_ && poisson_->is_valid() &&
           petsc_objects_ && petsc_objects_->initialized;
}

void DriftDiffusion::validate() const {
    if (!is_valid()) {
        throw std::runtime_error("DriftDiffusion solver is in invalid state");
    }
}

void DriftDiffusion::validate_order() const {
    if (order_ < 2 || order_ > 3) {
        throw std::invalid_argument("Order must be 2 (P2) or 3 (P3)");
    }
}

void DriftDiffusion::validate_inputs(const std::vector<double>& bc) const {
    if (bc.size() != 4) {
        throw std::invalid_argument("2D drift-diffusion solver requires exactly 4 boundary conditions");
    }

    for (size_t i = 0; i < bc.size(); ++i) {
        if (!std::isfinite(bc[i])) {
            throw std::invalid_argument("Boundary condition " + std::to_string(i) + " is not finite");
        }
    }
}

void DriftDiffusion::validate_doping() const {
    if (Nd_.empty() || Na_.empty()) {
        throw std::runtime_error("Doping profiles must be set before solving");
    }

    if (Nd_.size() != Na_.size()) {
        throw std::runtime_error("Nd and Na arrays must have the same size");
    }
}

size_t DriftDiffusion::get_dof_count() const {
    if (!is_valid()) return 0;

    try {
        Mesh mesh(device_, mesh_type_);
        return mesh.get_num_nodes();
    } catch (...) {
        return 0;
    }
}

double DriftDiffusion::get_convergence_residual() const {
    return convergence_residual_;
}

std::map<std::string, std::vector<double>> DriftDiffusion::solve_drift_diffusion(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {

    validate();
    validate_inputs(bc);
    validate_doping();

    if (mesh_type_ == MeshType::Structured) {
        return solve_structured_drift_diffusion(bc, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol);
    } else {
        return solve_unstructured_drift_diffusion(bc, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol);
    }
}

std::map<std::string, std::vector<double>> DriftDiffusion::solve_structured_drift_diffusion(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {

    try {
        Mesh mesh(device_, mesh_type_);
        auto grid_x = mesh.get_grid_points_x();
        auto grid_y = mesh.get_grid_points_y();
        auto elements = mesh.get_elements();

        if (grid_x.empty() || grid_y.empty() || elements.empty()) {
            throw std::runtime_error("Invalid mesh data");
        }

        int n_nodes = static_cast<int>(grid_x.size());

        // Initialize solution vectors
        std::vector<double> V(n_nodes, 0.0);
        std::vector<double> n(n_nodes, 1e10 * 1e6);  // Intrinsic carrier concentration
        std::vector<double> p(n_nodes, 1e10 * 1e6);
        std::vector<double> Jn(n_nodes, 0.0);
        std::vector<double> Jp(n_nodes, 0.0);

        // Physical constants
        const double q = 1.602e-19;        // Elementary charge (C)
        const double kT = 0.0259;          // Thermal voltage at 300K (V)
        const double mu_n = 1000e-4;       // Electron mobility (m²/V·s)
        const double mu_p = 400e-4;        // Hole mobility (m²/V·s)
        const double ni = 1e10 * 1e6;      // Intrinsic carrier concentration (1/m³)

        // Validate doping array sizes
        if (Nd_.size() != static_cast<size_t>(n_nodes) || Na_.size() != static_cast<size_t>(n_nodes)) {
            throw std::runtime_error("Doping array size must match number of mesh nodes");
        }

        // Initial Poisson solve
        V = poisson_->solve_2d_self_consistent(bc, n, p, Nd_, Na_, poisson_max_iter, poisson_tol);

        // Store previous solution for convergence checking
        std::vector<double> V_old = V;
        std::vector<double> n_old = n;
        std::vector<double> p_old = p;

        // Main iteration loop
        for (int step = 0; step < max_steps; ++step) {
            // Update carrier densities based on current potential
            compute_carrier_densities(V, n, p);

            // Compute electric field
            std::vector<double> Ex(n_nodes, 0.0), Ey(n_nodes, 0.0);

            // Calculate electric field using finite differences
            for (int i = 0; i < n_nodes; ++i) {
                // Find neighboring nodes for gradient calculation
                // This is a simplified approach - in practice, you'd use proper DG gradients
                if (i > 0 && i < n_nodes - 1) {
                    Ex[i] = -(V[i + 1] - V[i - 1]) / (grid_x[i + 1] - grid_x[i - 1]);
                }

                // For y-direction, we need to be more careful with 2D indexing
                int nx = static_cast<int>(std::sqrt(n_nodes)); // Assuming square grid
                int ix = i % nx;
                int iy = i / nx;

                if (iy > 0 && iy < nx - 1) {
                    int i_up = (iy + 1) * nx + ix;
                    int i_down = (iy - 1) * nx + ix;
                    if (i_up < n_nodes && i_down >= 0) {
                        Ey[i] = -(V[i_up] - V[i_down]) / (grid_y[i_up] - grid_y[i_down]);
                    }
                }
            }

            // Compute current densities
            compute_current_densities(V, n, p, Jn, Jp);

            // Advanced Adaptive Mesh Refinement
            if (use_amr) {
                performance::PROFILE_SCOPE("AMR_Processing");

                try {
                    // Create advanced AMR controller
                    amr::AMRController amr_controller(
                        amr::ErrorEstimatorType::KELLY,
                        amr::RefinementStrategy::EQUILIBRATION,
                        5  // max refinement levels
                    );

                    // Set refinement parameters
                    amr_controller.set_refinement_parameters(
                        0.2,    // refine_fraction
                        0.05,   // coarsen_fraction
                        1e-4,   // error_tolerance
                        1e-7,   // min_element_size
                        1e-3    // max_element_size
                    );

                    // Enable anisotropic refinement for boundary layers
                    amr_controller.set_anisotropic_refinement(true, true);

                    // Convert grid points to vertices format
                    std::vector<std::array<double, 2>> vertices_amr(grid_x.size());
                    for (size_t i = 0; i < grid_x.size(); ++i) {
                        vertices_amr[i] = {grid_x[i], grid_y[i]};
                    }

                    // Compute error indicators using Kelly estimator
                    auto error_indicators = amr_controller.compute_error_indicators(
                        V, elements, vertices_amr
                    );

                    // Determine refinement decisions
                    auto refinement_decisions = amr_controller.determine_refinement(
                        error_indicators, elements, vertices_amr
                    );

                    // Check if refinement is needed
                    bool needs_refinement = std::any_of(refinement_decisions.begin(),
                                                       refinement_decisions.end(),
                                                       [](const amr::ElementRefinement& r) {
                                                           return r.refine || r.coarsen;
                                                       });

                    if (needs_refinement) {
                        // Perform mesh refinement
                        auto element_mapping = amr_controller.perform_refinement(
                            refinement_decisions, elements, vertices_amr, V
                        );

                        // Update mesh data
                        grid_x.resize(vertices_amr.size());
                        grid_y.resize(vertices_amr.size());
                        for (size_t i = 0; i < vertices_amr.size(); ++i) {
                            grid_x[i] = vertices_amr[i][0];
                            grid_y[i] = vertices_amr[i][1];
                        }

                        n_nodes = static_cast<int>(grid_x.size());

                        // Interpolate carrier densities to new mesh
                        std::vector<double> n_new(n_nodes, ni);
                        std::vector<double> p_new(n_nodes, ni);
                        std::vector<double> Jn_new(n_nodes, 0.0);
                        std::vector<double> Jp_new(n_nodes, 0.0);

                        // Simple interpolation (could be improved with proper DG projection)
                        for (int i = 0; i < n_nodes; ++i) {
                            if (i < static_cast<int>(n.size())) {
                                n_new[i] = n[i];
                                p_new[i] = p[i];
                                Jn_new[i] = (i < static_cast<int>(Jn.size())) ? Jn[i] : 0.0;
                                Jp_new[i] = (i < static_cast<int>(Jp.size())) ? Jp[i] : 0.0;
                            }
                        }

                        // Update solution vectors
                        n = std::move(n_new);
                        p = std::move(p_new);
                        Jn = std::move(Jn_new);
                        Jp = std::move(Jp_new);

                        // Get refinement statistics
                        auto stats = amr_controller.get_last_refinement_stats();
                        std::cout << "AMR: Refined " << stats.elements_refined
                                 << " elements, coarsened " << stats.elements_coarsened
                                 << " elements" << std::endl;

                        // Check mesh quality
                        auto quality = amr_controller.compute_mesh_quality(elements, vertices_amr);
                        if (quality.num_bad_elements > 0) {
                            std::cout << "Warning: " << quality.num_bad_elements
                                     << " poor quality elements detected" << std::endl;
                        }
                    }

                } catch (const std::exception& e) {
                    std::cerr << "Advanced AMR failed: " << e.what() << std::endl;
                    // Fallback to simple refinement
                    std::vector<bool> refine_flags(elements.size(), false);
                    for (size_t e = 0; e < elements.size(); ++e) {
                        if (elements[e].size() >= 3) {
                            int i1 = elements[e][0], i2 = elements[e][1], i3 = elements[e][2];
                            if (i1 >= 0 && i1 < n_nodes && i2 >= 0 && i2 < n_nodes && i3 >= 0 && i3 < n_nodes) {
                                double grad_x = std::abs(V[i2] - V[i1]) / std::max(std::abs(grid_x[i2] - grid_x[i1]), 1e-12);
                                double grad_y = std::abs(V[i3] - V[i1]) / std::max(std::abs(grid_y[i3] - grid_y[i1]), 1e-12);
                                if (grad_x > 1e5 || grad_y > 1e5) {
                                    refine_flags[e] = true;
                                }
                            }
                        }
                    }
                    mesh.refine(refine_flags);
                }
            }

            // Update potential with self-consistent Poisson solver
            try {
                V = poisson_->solve_2d_self_consistent(bc, n, p, Nd_, Na_, poisson_max_iter, poisson_tol);
            } catch (const std::exception& e) {
                std::cerr << "Poisson solve failed: " << e.what() << std::endl;
                break;
            }

            // Check convergence
            if (check_convergence(V_old, V, 1e-6)) {
                convergence_residual_ = 0.0;
                for (size_t i = 0; i < V.size(); ++i) {
                    convergence_residual_ += std::abs(V[i] - V_old[i]);
                }
                convergence_residual_ /= V.size();
                break;
            }

            // Update old solutions
            V_old = V;
            n_old = n;
            p_old = p;
        }

        // Prepare results
        std::map<std::string, std::vector<double>> results;
        results["potential"] = V;
        results["n"] = n;
        results["p"] = p;
        results["Jn"] = Jn;
        results["Jp"] = Jp;

        return results;

    } catch (const std::exception& e) {
        throw std::runtime_error("Structured drift-diffusion solve failed: " + std::string(e.what()));
    }
}

std::map<std::string, std::vector<double>> DriftDiffusion::solve_structured_drift_diffusion_unstructured_fallback(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {

    // For now, delegate to structured solver
    // In a full implementation, this would use unstructured DG methods
    return solve_structured_drift_diffusion(bc, Vg, max_steps, use_amr, poisson_max_iter, poisson_tol);
}

void DriftDiffusion::compute_carrier_densities(const std::vector<double>& V,
                                              std::vector<double>& n,
                                              std::vector<double>& p) const {
    performance::PROFILE_FUNCTION();

    const double kT = 0.0259;          // Thermal voltage at 300K (V)
    const double ni = 1e10 * 1e6;      // Intrinsic carrier concentration (1/m³)
    const double inv_kT = 1.0 / kT;    // Precompute inverse for efficiency

    const size_t n_points = V.size();

    // Use SIMD-optimized computation if available
    if (performance::simd::VectorOps::has_avx2() && n_points >= 4) {
        performance::PROFILE_SCOPE("SIMD_Carrier_Densities");

        // Prepare aligned arrays for SIMD operations
        performance::memory::AlignedVector<double> phi_array(n_points);
        performance::memory::AlignedVector<double> n_intrinsic(n_points);
        performance::memory::AlignedVector<double> p_intrinsic(n_points);

        // Compute electrostatic potential relative to trap levels
        #pragma omp parallel for simd
        for (size_t i = 0; i < n_points; ++i) {
            double Et = (Et_.size() > i) ? Et_[i] : 0.0;
            phi_array[i] = (V[i] - Et) * inv_kT;
        }

        // Vectorized exponential computation
        #pragma omp parallel for simd
        for (size_t i = 0; i < n_points; ++i) {
            n_intrinsic[i] = ni * std::exp(phi_array[i]);
            p_intrinsic[i] = ni * std::exp(-phi_array[i]);
        }

        // Apply doping effects with vectorization
        #pragma omp parallel for
        for (size_t i = 0; i < n_points; ++i) {
            if (i < Nd_.size() && i < Na_.size()) {
                n[i] = n_intrinsic[i];
                p[i] = p_intrinsic[i];

                // Apply doping effects
                if (Nd_[i] > Na_[i]) {
                    n[i] += (Nd_[i] - Na_[i]);
                } else if (Na_[i] > Nd_[i]) {
                    p[i] += (Na_[i] - Nd_[i]);
                }

                // Ensure positive concentrations
                n[i] = std::max(n[i], ni);
                p[i] = std::max(p[i], ni);
            }
        }
    } else {
        // Standard computation for smaller arrays or when SIMD unavailable
        #pragma omp parallel for
        for (size_t i = 0; i < n_points; ++i) {
            if (i < Nd_.size() && i < Na_.size()) {
                // Calculate quasi-Fermi levels and carrier densities
                double Et = (Et_.size() > i) ? Et_[i] : 0.0;
                double phi = (V[i] - Et) * inv_kT;

                // Boltzmann statistics (valid for non-degenerate semiconductors)
                n[i] = ni * std::exp(phi);
                p[i] = ni * std::exp(-phi);

                // Apply doping effects
                if (Nd_[i] > Na_[i]) {
                    n[i] += (Nd_[i] - Na_[i]);
                } else if (Na_[i] > Nd_[i]) {
                    p[i] += (Na_[i] - Nd_[i]);
                }

                // Ensure positive concentrations
                n[i] = std::max(n[i], ni);
                p[i] = std::max(p[i], ni);
            }
        }
    }
}

void DriftDiffusion::compute_current_densities(const std::vector<double>& V,
                                              const std::vector<double>& n,
                                              const std::vector<double>& p,
                                              std::vector<double>& Jn,
                                              std::vector<double>& Jp) const {
    const double q = 1.602e-19;        // Elementary charge (C)
    const double kT = 0.0259;          // Thermal voltage at 300K (V)
    const double mu_n = 1000e-4;       // Electron mobility (m²/V·s)
    const double mu_p = 400e-4;        // Hole mobility (m²/V·s)

    // Simplified current density calculation
    // In a full implementation, this would use proper DG gradients
    for (size_t i = 0; i < V.size(); ++i) {
        if (i > 0 && i < V.size() - 1) {
            // Electric field (simplified finite difference)
            double Ex = -(V[i + 1] - V[i - 1]) / 2.0;  // Assuming unit spacing

            // Electron current density (drift + diffusion)
            double Dn = mu_n * kT / q;  // Einstein relation
            double grad_n = (n[i + 1] - n[i - 1]) / 2.0;
            Jn[i] = q * mu_n * n[i] * Ex + q * Dn * grad_n;

            // Hole current density (drift + diffusion)
            double Dp = mu_p * kT / q;  // Einstein relation
            double grad_p = (p[i + 1] - p[i - 1]) / 2.0;
            Jp[i] = q * mu_p * p[i] * Ex - q * Dp * grad_p;
        } else {
            Jn[i] = 0.0;
            Jp[i] = 0.0;
        }
    }
}

bool DriftDiffusion::check_convergence(const std::vector<double>& V_old,
                                      const std::vector<double>& V_new,
                                      double tolerance) const {
    if (V_old.size() != V_new.size()) return false;

    double max_change = 0.0;
    for (size_t i = 0; i < V_old.size(); ++i) {
        max_change = std::max(max_change, std::abs(V_new[i] - V_old[i]));
    }

    return max_change < tolerance;
}

// Missing function implementations
double DriftDiffusion::compute_residual_norm(const std::vector<double>& V_old, const std::vector<double>& V_new) const {
    if (V_old.size() != V_new.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }

    double norm = 0.0;
    for (size_t i = 0; i < V_old.size(); ++i) {
        double diff = V_new[i] - V_old[i];
        norm += diff * diff;
    }
    return std::sqrt(norm);
}

bool DriftDiffusion::check_unstructured_convergence(const std::vector<double>& V_old, const std::vector<double>& V_new, double tolerance) const {
    double residual = compute_residual_norm(V_old, V_new);
    return residual < tolerance;
}

void DriftDiffusion::solve_unstructured_continuity_equations(
    const std::vector<double>& V,
    std::vector<double>& n,
    std::vector<double>& p,
    std::vector<double>& Jn,
    std::vector<double>& Jp,
    const std::vector<std::vector<int>>& elements,
    int dofs_per_element) {

    // Placeholder implementation - delegate to structured solver for now
    compute_current_densities(V, n, p, Jn, Jp);
}

void DriftDiffusion::perform_unstructured_amr(
    const Mesh& mesh,
    const std::vector<double>& V,
    std::vector<std::vector<int>>& elements,
    int dofs_per_element) {

    // Placeholder implementation - AMR not implemented yet
    // This would typically refine elements based on error estimates
}

std::map<std::string, std::vector<double>> DriftDiffusion::convert_dg_to_nodal(
    const std::vector<double>& V,
    const std::vector<double>& n,
    const std::vector<double>& p,
    const std::vector<double>& Jn,
    const std::vector<double>& Jp,
    const std::vector<std::vector<int>>& elements,
    const std::vector<double>& grid_x,
    const std::vector<double>& grid_y,
    int dofs_per_element) const {

    // Placeholder implementation - return input data as-is for now
    std::map<std::string, std::vector<double>> result;
    result["potential"] = V;
    result["n"] = n;
    result["p"] = p;
    result["Jn"] = Jn;
    result["Jp"] = Jp;
    return result;
}

// Note: solve_unstructured_poisson and compute_unstructured_carrier_densities
// are implemented in the unstructured file to avoid multiple definitions

} // namespace simulator

// C interface implementation
extern "C" {
    simulator::DriftDiffusion* create_drift_diffusion(simulator::Device* device, int method, int mesh_type, int order) {
        if (!device) return nullptr;
        try {
            return new simulator::DriftDiffusion(*device, static_cast<simulator::Method>(method),
                                               static_cast<simulator::MeshType>(mesh_type), order);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_drift_diffusion(simulator::DriftDiffusion* dd) {
        delete dd;
    }

    int drift_diffusion_is_valid(simulator::DriftDiffusion* dd) {
        if (!dd) return 0;
        return dd->is_valid() ? 1 : 0;
    }

    int drift_diffusion_set_doping(simulator::DriftDiffusion* dd, double* Nd, double* Na, int size) {
        if (!dd || !Nd || !Na || size <= 0) return -1;
        try {
            std::vector<double> Nd_vec(Nd, Nd + size);
            std::vector<double> Na_vec(Na, Na + size);
            dd->set_doping(Nd_vec, Na_vec);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int drift_diffusion_set_trap_level(simulator::DriftDiffusion* dd, double* Et, int size) {
        if (!dd || !Et || size <= 0) return -1;
        try {
            std::vector<double> Et_vec(Et, Et + size);
            dd->set_trap_level(Et_vec);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int drift_diffusion_solve(simulator::DriftDiffusion* dd, double* bc, int bc_size, double Vg,
                             int max_steps, int use_amr, int poisson_max_iter, double poisson_tol,
                             double* V, double* n, double* p, double* Jn, double* Jp, int size) {
        if (!dd || !bc || !V || !n || !p || !Jn || !Jp || bc_size != 4 || size <= 0) return -1;

        try {
            std::vector<double> bc_vec(bc, bc + bc_size);
            auto results = dd->solve_drift_diffusion(bc_vec, Vg, max_steps, use_amr != 0,
                                                    poisson_max_iter, poisson_tol);

            // Copy results to output arrays
            auto& V_result = results["potential"];
            auto& n_result = results["n"];
            auto& p_result = results["p"];
            auto& Jn_result = results["Jn"];
            auto& Jp_result = results["Jp"];

            int copy_size = std::min(size, static_cast<int>(V_result.size()));
            for (int i = 0; i < copy_size; ++i) {
                V[i] = V_result[i];
                n[i] = n_result[i];
                p[i] = p_result[i];
                Jn[i] = (i < static_cast<int>(Jn_result.size())) ? Jn_result[i] : 0.0;
                Jp[i] = (i < static_cast<int>(Jp_result.size())) ? Jp_result[i] : 0.0;
            }

            return 0;
        } catch (...) {
            return -1;
        }
    }

    size_t drift_diffusion_get_dof_count(simulator::DriftDiffusion* dd) {
        if (!dd) return 0;
        return dd->get_dof_count();
    }

    double drift_diffusion_get_convergence_residual(simulator::DriftDiffusion* dd) {
        if (!dd) return 0.0;
        return dd->get_convergence_residual();
    }

    int drift_diffusion_get_order(simulator::DriftDiffusion* dd) {
        if (!dd) return 0;
        return dd->get_order();
    }
}