#include "driftdiffusion.hpp"
#include "mesh.hpp"
#include "dg_assembly.hpp"
#include "dg_basis_functions.hpp"
#include <petscksp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <functional>

namespace simulator {

// Implementation of unstructured DG methods for drift-diffusion solver
std::map<std::string, std::vector<double>> DriftDiffusion::solve_unstructured_drift_diffusion(
    const std::vector<double>& bc, double Vg, int max_steps, bool use_amr,
    int poisson_max_iter, double poisson_tol) {

    try {
        Mesh mesh(device_, mesh_type_);

        // Generate unstructured mesh
        mesh.generate_gmsh_mesh("device_unstructured_dd.msh");

        auto grid_x = mesh.get_grid_points_x();
        auto grid_y = mesh.get_grid_points_y();
        auto elements = mesh.get_elements();

        if (grid_x.empty() || grid_y.empty() || elements.empty()) {
            throw std::runtime_error("Invalid unstructured mesh data");
        }

        int n_nodes = static_cast<int>(grid_x.size());
        int n_elements = static_cast<int>(elements.size());

        // Create DG assembly object
        dg::DGAssembly dg_assembler(order_, 50.0);
        int dofs_per_element = dg_assembler.get_dofs_per_element();
        int n_dofs = n_elements * dofs_per_element;

        // Initialize solution vectors
        std::vector<double> V(n_dofs, 0.0);
        std::vector<double> n(n_dofs, 1e10 * 1e6);  // Intrinsic carrier concentration
        std::vector<double> p(n_dofs, 1e10 * 1e6);
        std::vector<double> Jn(n_dofs, 0.0);
        std::vector<double> Jp(n_dofs, 0.0);

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
        V = solve_unstructured_poisson(bc, n, p);

        // Store previous solution for convergence checking
        std::vector<double> V_old = V;
        std::vector<double> n_old = n;
        std::vector<double> p_old = p;

        // Main iteration loop
        for (int step = 0; step < max_steps; ++step) {
            // Update carrier densities
            compute_unstructured_carrier_densities(V, n, p, elements, dofs_per_element);

            // Solve continuity equations for electrons and holes
            solve_unstructured_continuity_equations(V, n, p, Jn, Jp, elements, dofs_per_element);

            // Adaptive mesh refinement
            if (use_amr) {
                try {
                    perform_unstructured_amr(mesh, V, elements, dofs_per_element);

                    // Update mesh data after refinement
                    grid_x = mesh.get_grid_points_x();
                    grid_y = mesh.get_grid_points_y();
                    elements = mesh.get_elements();
                    n_nodes = static_cast<int>(grid_x.size());
                    n_elements = static_cast<int>(elements.size());
                    n_dofs = n_elements * dofs_per_element;

                    // Resize solution vectors
                    V.resize(n_dofs, 0.0);
                    n.resize(n_dofs, ni);
                    p.resize(n_dofs, ni);
                    Jn.resize(n_dofs, 0.0);
                    Jp.resize(n_dofs, 0.0);
                } catch (const std::exception& e) {
                    std::cerr << "Unstructured AMR failed: " << e.what() << std::endl;
                }
            }

            // Update potential with self-consistent Poisson solver
            try {
                V = solve_unstructured_poisson(bc, n, p);
            } catch (const std::exception& e) {
                std::cerr << "Unstructured Poisson solve failed: " << e.what() << std::endl;
                break;
            }

            // Check convergence
            if (check_unstructured_convergence(V_old, V, 1e-6)) {
                convergence_residual_ = compute_residual_norm(V_old, V);
                break;
            }

            // Update old solutions
            V_old = V;
            n_old = n;
            p_old = p;
        }

        // Convert DG solution to nodal values for output
        auto nodal_results = convert_dg_to_nodal(V, n, p, Jn, Jp, elements, grid_x, grid_y, dofs_per_element);

        return nodal_results;

    } catch (const std::exception& e) {
        throw std::runtime_error("Unstructured drift-diffusion solve failed: " + std::string(e.what()));
    }
}

// Helper method implementations for unstructured DG
std::vector<double> DriftDiffusion::solve_unstructured_poisson(
    const std::vector<double>& bc,
    const std::vector<double>& n,
    const std::vector<double>& p) {

    // Use the Poisson solver with updated charge density
    std::vector<double> rho(n.size());
    const double q = 1.602e-19;

    for (size_t i = 0; i < n.size(); ++i) {
        double Nd_val = (i < Nd_.size()) ? Nd_[i] : 0.0;
        double Na_val = (i < Na_.size()) ? Na_[i] : 0.0;
        rho[i] = q * (p[i] - n[i] + Nd_val - Na_val);
    }

    poisson_->set_charge_density(rho);
    return poisson_->solve_2d(bc);
}

void DriftDiffusion::compute_unstructured_carrier_densities(
    const std::vector<double>& V,
    std::vector<double>& n,
    std::vector<double>& p,
    const std::vector<std::vector<int>>& elements,
    int dofs_per_element) const {

    const double kT = 0.0259;          // Thermal voltage at 300K (V)
    const double ni = 1e10 * 1e6;      // Intrinsic carrier concentration (1/m³)

    // For DG methods, carrier densities are computed at DOF locations
    for (size_t e = 0; e < elements.size(); ++e) {
        for (int dof = 0; dof < dofs_per_element; ++dof) {
            int global_dof = e * dofs_per_element + dof;

            if (global_dof < static_cast<int>(V.size())) {
                // Calculate quasi-Fermi levels and carrier densities
                double Et = (Et_.size() > global_dof) ? Et_[global_dof] : 0.0;
                double phi = V[global_dof] - Et;

                // Boltzmann statistics
                n[global_dof] = ni * std::exp(phi / kT);
                p[global_dof] = ni * std::exp(-phi / kT);

                // Apply doping effects
                if (global_dof < static_cast<int>(Nd_.size()) && global_dof < static_cast<int>(Na_.size())) {
                    if (Nd_[global_dof] > Na_[global_dof]) {
                        n[global_dof] += (Nd_[global_dof] - Na_[global_dof]);
                    } else if (Na_[global_dof] > Nd_[global_dof]) {
                        p[global_dof] += (Na_[global_dof] - Nd_[global_dof]);
                    }
                }

                // Ensure positive concentrations
                n[global_dof] = std::max(n[global_dof], ni);
                p[global_dof] = std::max(p[global_dof], ni);
            }
        }
    }
}

} // namespace simulator