#include "quantum_transport.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>

namespace simulator {
namespace quantum {

// Physical constants
constexpr double HBAR = 1.054571817e-34;  // Reduced Planck constant (J·s)
constexpr double ME = 9.1093837015e-31;   // Electron mass (kg)
constexpr double Q = 1.602176634e-19;     // Elementary charge (C)
constexpr double KB = 1.380649e-23;       // Boltzmann constant (J/K)
constexpr double EV_TO_J = 1.602176634e-19; // eV to Joules conversion

QuantumTransportSolver::QuantumTransportSolver() {
    // Initialize default parameters
    params_.transport_type = QuantumTransportType::BALLISTIC;
    params_.confinement_type = ConfinementType::NONE;
    params_.effective_mass = 0.067;
    params_.temperature = 300.0;
    params_.fermi_level = 0.0;
    params_.max_states = 100;
    params_.energy_cutoff = 2.0;
    params_.convergence_tolerance = 1e-8;
    params_.max_iterations = 1000;
}

QuantumTransportSolver::~QuantumTransportSolver() = default;

void QuantumTransportSolver::set_parameters(const QuantumTransportParameters& params) {
    params_ = params;
}

void QuantumTransportSolver::set_mesh(const std::vector<std::array<double, 2>>& vertices,
                                    const std::vector<std::array<int, 3>>& elements) {
    vertices_ = vertices;
    elements_ = elements;
    
    // Initialize matrices
    int num_nodes = vertices_.size();
    hamiltonian_matrix_.assign(num_nodes, std::vector<double>(num_nodes, 0.0));
    overlap_matrix_.assign(num_nodes, std::vector<double>(num_nodes, 0.0));
    green_function_matrix_.assign(num_nodes, std::vector<std::complex<double>>(num_nodes, 0.0));
}

void QuantumTransportSolver::set_potential(const std::vector<double>& potential) {
    potential_ = potential;
}

void QuantumTransportSolver::set_material_properties(const std::vector<double>& effective_mass,
                                                   const std::vector<double>& band_offset) {
    effective_mass_ = effective_mass;
    band_offset_ = band_offset;
}

bool QuantumTransportSolver::solve_schrodinger_equation() {
    if (vertices_.empty() || elements_.empty()) {
        std::cerr << "Error: Mesh not set" << std::endl;
        return false;
    }
    
    // Build Hamiltonian and overlap matrices
    build_hamiltonian_matrix();
    build_overlap_matrix();
    
    // Apply boundary conditions
    apply_boundary_conditions();
    
    // Solve eigenvalue problem
    return solve_eigenvalue_problem();
}

bool QuantumTransportSolver::calculate_quantum_states() {
    if (!solve_schrodinger_equation()) {
        return false;
    }
    
    // Update occupation numbers based on Fermi-Dirac statistics
    update_occupation_numbers();
    
    results_.converged = true;
    return true;
}

std::vector<QuantumState> QuantumTransportSolver::get_bound_states() const {
    std::vector<QuantumState> bound_states;
    
    for (const auto& state : results_.states) {
        if (state.energy < params_.energy_cutoff) {
            bound_states.push_back(state);
        }
    }
    
    return bound_states;
}

std::vector<QuantumState> QuantumTransportSolver::get_scattering_states(double energy) const {
    std::vector<QuantumState> scattering_states;
    
    // Calculate scattering states at given energy
    // This is a simplified implementation
    for (int i = 0; i < 10; ++i) {
        QuantumState state;
        state.energy = energy;
        state.n = i;
        state.wavefunction.resize(vertices_.size());
        
        // Simple plane wave approximation
        double k = sqrt(2.0 * params_.effective_mass * ME * energy * EV_TO_J) / HBAR;
        for (size_t j = 0; j < vertices_.size(); ++j) {
            double x = vertices_[j][0];
            state.wavefunction[j] = std::complex<double>(cos(k * x), sin(k * x));
        }
        
        scattering_states.push_back(state);
    }
    
    return scattering_states;
}

bool QuantumTransportSolver::calculate_transmission_coefficients() {
    results_.transmission_coefficients.clear();
    results_.reflection_coefficients.clear();
    
    // Energy grid for transmission calculation
    int num_energies = 100;
    double energy_min = 0.0;
    double energy_max = params_.energy_cutoff;
    double energy_step = (energy_max - energy_min) / (num_energies - 1);
    
    results_.energy_grid.resize(num_energies);
    results_.transmission_function.resize(num_energies);
    
    for (int i = 0; i < num_energies; ++i) {
        double energy = energy_min + i * energy_step;
        results_.energy_grid[i] = energy;
        
        // Calculate transmission coefficient using transfer matrix method
        double transmission = calculate_transmission_at_energy(energy);
        results_.transmission_function[i] = transmission;
        
        if (i < 10) { // Store first 10 for compatibility
            results_.transmission_coefficients.push_back(transmission);
            results_.reflection_coefficients.push_back(1.0 - transmission);
        }
    }
    
    return true;
}

double QuantumTransportSolver::calculate_transmission_at_energy(double energy) const {
    if (potential_.empty()) {
        return 1.0; // Perfect transmission for no potential
    }
    
    // Simple WKB approximation for transmission
    double barrier_height = *std::max_element(potential_.begin(), potential_.end());
    
    if (energy > barrier_height) {
        return 1.0; // Above barrier
    }
    
    // Below barrier - tunneling
    double barrier_width = params_.barrier_width;
    double kappa = sqrt(2.0 * params_.effective_mass * ME * (barrier_height - energy) * EV_TO_J) / HBAR;
    
    return exp(-2.0 * kappa * barrier_width);
}

bool QuantumTransportSolver::calculate_current_density() {
    if (results_.states.empty()) {
        return false;
    }
    
    int num_nodes = vertices_.size();
    results_.current_density.assign(num_nodes, std::vector<double>(2, 0.0));
    
    // Calculate current density from quantum states
    for (const auto& state : results_.states) {
        if (state.occupation > 1e-10) {
            // Calculate current contribution from this state
            for (int i = 0; i < num_nodes; ++i) {
                // Simplified current calculation
                double psi_real = state.wavefunction[i].real();
                double psi_imag = state.wavefunction[i].imag();
                
                // Current density components (simplified)
                results_.current_density[i][0] += state.occupation * psi_imag * psi_real;
                results_.current_density[i][1] += state.occupation * psi_real * psi_imag;
            }
        }
    }
    
    return true;
}

bool QuantumTransportSolver::calculate_charge_density() {
    if (results_.states.empty()) {
        return false;
    }
    
    int num_nodes = vertices_.size();
    results_.charge_density.assign(num_nodes, std::vector<double>(1, 0.0));
    
    // Calculate charge density from quantum states
    for (const auto& state : results_.states) {
        for (int i = 0; i < num_nodes; ++i) {
            double psi_magnitude_sq = std::norm(state.wavefunction[i]);
            results_.charge_density[i][0] += state.occupation * psi_magnitude_sq;
        }
    }
    
    return true;
}

void QuantumTransportSolver::add_scattering_mechanism(ScatteringType type) {
    auto it = std::find(params_.scattering_mechanisms.begin(), 
                       params_.scattering_mechanisms.end(), type);
    if (it == params_.scattering_mechanisms.end()) {
        params_.scattering_mechanisms.push_back(type);
    }
}

bool QuantumTransportSolver::calculate_scattering_rates() {
    // Calculate scattering rates for each mechanism
    for (auto mechanism : params_.scattering_mechanisms) {
        switch (mechanism) {
            case ScatteringType::ACOUSTIC_PHONON:
                calculate_acoustic_phonon_scattering();
                break;
            case ScatteringType::OPTICAL_PHONON:
                calculate_optical_phonon_scattering();
                break;
            case ScatteringType::IONIZED_IMPURITY:
                calculate_impurity_scattering();
                break;
            default:
                break;
        }
    }
    
    return true;
}

void QuantumTransportSolver::calculate_acoustic_phonon_scattering() {
    // Simplified acoustic phonon scattering calculation
    double deformation_potential = params_.acoustic_deformation_potential;
    double sound_velocity = 5000.0; // m/s (typical for semiconductors)
    double density = 5320.0; // kg/m³ (typical for GaAs)
    
    for (auto& state : results_.states) {
        // Calculate scattering rate (simplified)
        double scattering_rate = (deformation_potential * deformation_potential * KB * params_.temperature) /
                               (HBAR * density * sound_velocity * sound_velocity);
        
        // Store in state (this is simplified - normally would be state-dependent)
        // For now, just modify occupation slightly
        state.occupation *= (1.0 - scattering_rate * 1e-12);
    }
}

void QuantumTransportSolver::calculate_optical_phonon_scattering() {
    // Simplified optical phonon scattering calculation
    double phonon_energy = 0.036; // eV (typical LO phonon energy for GaAs)
    
    for (auto& state : results_.states) {
        if (state.energy > phonon_energy) {
            // Emission possible
            double emission_rate = 1e12; // 1/s (typical)
            state.occupation *= (1.0 - emission_rate * 1e-12);
        }
    }
}

void QuantumTransportSolver::calculate_impurity_scattering() {
    // Simplified impurity scattering calculation
    double impurity_density = 1e16; // cm⁻³
    
    for (auto& state : results_.states) {
        double scattering_rate = impurity_density * 1e6 * 1e-15; // Simplified
        state.occupation *= (1.0 - scattering_rate * 1e-12);
    }
}

bool QuantumTransportSolver::solve_boltzmann_equation() {
    // Simplified Boltzmann equation solution
    // In practice, this would involve iterative solution of the BTE
    
    calculate_scattering_rates();
    
    // Update transport properties
    results_.mobility = calculate_quantum_mobility();
    results_.conductance = calculate_quantum_conductance();
    results_.resistance = (results_.conductance > 0) ? 1.0 / results_.conductance : 1e12;
    
    return true;
}

double QuantumTransportSolver::calculate_quantum_mobility() const {
    // Simplified mobility calculation
    double total_scattering_rate = 0.0;
    double total_occupation = 0.0;
    
    for (const auto& state : results_.states) {
        if (state.occupation > 1e-10) {
            total_scattering_rate += state.occupation / 1e12; // Simplified
            total_occupation += state.occupation;
        }
    }
    
    if (total_occupation > 0 && total_scattering_rate > 0) {
        return Q / (params_.effective_mass * ME * total_scattering_rate / total_occupation);
    }
    
    return 0.1; // Default mobility (m²/V·s)
}

double QuantumTransportSolver::calculate_quantum_conductance() const {
    // Quantum conductance calculation
    double conductance = 0.0;
    
    for (size_t i = 0; i < results_.transmission_function.size(); ++i) {
        double energy = results_.energy_grid[i];
        double transmission = results_.transmission_function[i];
        
        // Fermi-Dirac distribution
        double fermi_factor = 1.0 / (1.0 + exp((energy - params_.fermi_level) / 
                                              (KB * params_.temperature / EV_TO_J)));
        
        conductance += transmission * fermi_factor;
    }
    
    // Quantum of conductance
    double g0 = 2.0 * Q * Q / (2.0 * M_PI * HBAR); // Factor of 2 for spin
    
    return g0 * conductance / results_.transmission_function.size();
}

QuantumTransportResults QuantumTransportSolver::get_results() const {
    return results_;
}

std::vector<double> QuantumTransportSolver::get_wavefunction(int state_index) const {
    if (state_index < 0 || state_index >= static_cast<int>(results_.states.size())) {
        return {};
    }
    
    const auto& state = results_.states[state_index];
    std::vector<double> wavefunction_real;
    wavefunction_real.reserve(state.wavefunction.size());
    
    for (const auto& psi : state.wavefunction) {
        wavefunction_real.push_back(psi.real());
    }
    
    return wavefunction_real;
}

void QuantumTransportSolver::build_hamiltonian_matrix() {
    int num_nodes = vertices_.size();
    
    // Initialize matrix
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            hamiltonian_matrix_[i][j] = 0.0;
        }
    }
    
    // Assemble kinetic and potential energy terms
    assemble_kinetic_energy_matrix();
    assemble_potential_energy_matrix();
}

void QuantumTransportSolver::assemble_kinetic_energy_matrix() {
    // Simplified finite difference kinetic energy matrix
    int num_nodes = vertices_.size();
    double hbar_sq_over_2m = HBAR * HBAR / (2.0 * params_.effective_mass * ME);
    
    for (const auto& element : elements_) {
        // Get element nodes
        int n1 = element[0];
        int n2 = element[1];
        int n3 = element[2];
        
        // Calculate element area
        double x1 = vertices_[n1][0], y1 = vertices_[n1][1];
        double x2 = vertices_[n2][0], y2 = vertices_[n2][1];
        double x3 = vertices_[n3][0], y3 = vertices_[n3][1];
        
        double area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
        
        if (area > 1e-15) {
            // Add kinetic energy contribution (simplified)
            double kinetic_contrib = hbar_sq_over_2m / area;
            
            hamiltonian_matrix_[n1][n1] += kinetic_contrib;
            hamiltonian_matrix_[n2][n2] += kinetic_contrib;
            hamiltonian_matrix_[n3][n3] += kinetic_contrib;
            
            hamiltonian_matrix_[n1][n2] -= 0.5 * kinetic_contrib;
            hamiltonian_matrix_[n2][n1] -= 0.5 * kinetic_contrib;
            hamiltonian_matrix_[n2][n3] -= 0.5 * kinetic_contrib;
            hamiltonian_matrix_[n3][n2] -= 0.5 * kinetic_contrib;
            hamiltonian_matrix_[n3][n1] -= 0.5 * kinetic_contrib;
            hamiltonian_matrix_[n1][n3] -= 0.5 * kinetic_contrib;
        }
    }
}

void QuantumTransportSolver::assemble_potential_energy_matrix() {
    // Add potential energy terms to diagonal
    int num_nodes = vertices_.size();
    
    for (int i = 0; i < num_nodes && i < static_cast<int>(potential_.size()); ++i) {
        hamiltonian_matrix_[i][i] += potential_[i] * EV_TO_J;
    }
}

void QuantumTransportSolver::build_overlap_matrix() {
    int num_nodes = vertices_.size();
    
    // Initialize overlap matrix (identity for now - simplified)
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            overlap_matrix_[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void QuantumTransportSolver::apply_boundary_conditions() {
    // Apply boundary conditions (simplified - zero at boundaries)
    int num_nodes = vertices_.size();
    
    for (int i = 0; i < num_nodes; ++i) {
        // Check if node is on boundary (simplified check)
        double x = vertices_[i][0];
        double y = vertices_[i][1];
        
        bool on_boundary = (x < 1e-10 || y < 1e-10); // Simplified boundary detection
        
        if (on_boundary) {
            // Zero boundary condition
            for (int j = 0; j < num_nodes; ++j) {
                hamiltonian_matrix_[i][j] = 0.0;
                hamiltonian_matrix_[j][i] = 0.0;
            }
            hamiltonian_matrix_[i][i] = 1.0;
        }
    }
}

bool QuantumTransportSolver::solve_eigenvalue_problem() {
    // Simplified eigenvalue solver (in practice, would use LAPACK or similar)
    int num_nodes = vertices_.size();
    results_.states.clear();
    
    // For demonstration, create some mock eigenvalues and eigenvectors
    int num_states = std::min(params_.max_states, num_nodes);
    
    for (int n = 0; n < num_states; ++n) {
        QuantumState state;
        state.n = n;
        state.energy = 0.1 * (n + 1); // Mock energy levels
        state.wavefunction.resize(num_nodes);
        
        // Mock wavefunction (sine wave)
        for (int i = 0; i < num_nodes; ++i) {
            double x = vertices_[i][0];
            double psi = sin(M_PI * (n + 1) * x);
            state.wavefunction[i] = std::complex<double>(psi, 0.0);
        }
        
        normalize_wavefunction(state);
        results_.states.push_back(state);
    }
    
    return true;
}

void QuantumTransportSolver::normalize_wavefunction(QuantumState& state) {
    double norm = 0.0;
    for (const auto& psi : state.wavefunction) {
        norm += std::norm(psi);
    }
    
    if (norm > 1e-15) {
        double normalization = 1.0 / sqrt(norm);
        for (auto& psi : state.wavefunction) {
            psi *= normalization;
        }
    }
}

void QuantumTransportSolver::update_occupation_numbers() {
    for (auto& state : results_.states) {
        // Fermi-Dirac distribution
        double fermi_factor = 1.0 / (1.0 + exp((state.energy - params_.fermi_level) / 
                                              (KB * params_.temperature / EV_TO_J)));
        state.occupation = fermi_factor;
    }
}

// QuantumConfinementCalculator implementation
QuantumConfinementCalculator::QuantumConfinementCalculator()
    : confinement_type_(ConfinementType::NONE),
      dimensions_({1e-8, 1e-8, 1e-8}),
      effective_mass_(0.067),
      band_offset_(0.0) {
}

void QuantumConfinementCalculator::set_confinement_type(ConfinementType type) {
    confinement_type_ = type;
}

void QuantumConfinementCalculator::set_dimensions(const std::array<double, 3>& dimensions) {
    dimensions_ = dimensions;
}

void QuantumConfinementCalculator::set_material_parameters(double effective_mass, double band_offset) {
    effective_mass_ = effective_mass;
    band_offset_ = band_offset;
}

std::vector<double> QuantumConfinementCalculator::calculate_energy_levels(int max_levels) const {
    switch (confinement_type_) {
        case ConfinementType::QUANTUM_WELL:
            return solve_infinite_well_1d(max_levels);
        case ConfinementType::QUANTUM_WIRE:
            return solve_infinite_well_2d(max_levels);
        case ConfinementType::QUANTUM_DOT:
            return solve_infinite_well_3d(max_levels);
        default:
            return {};
    }
}

std::vector<std::vector<double>> QuantumConfinementCalculator::calculate_wavefunctions(int max_levels) const {
    std::vector<std::vector<double>> wavefunctions;

    // Simplified wavefunction calculation
    int num_points = 100;

    for (int n = 1; n <= max_levels; ++n) {
        std::vector<double> wavefunction(num_points);

        for (int i = 0; i < num_points; ++i) {
            double x = static_cast<double>(i) / (num_points - 1);

            switch (confinement_type_) {
                case ConfinementType::QUANTUM_WELL:
                    wavefunction[i] = sqrt(2.0) * sin(n * M_PI * x);
                    break;
                case ConfinementType::QUANTUM_WIRE:
                case ConfinementType::QUANTUM_DOT:
                    wavefunction[i] = sqrt(2.0) * sin(n * M_PI * x) * sin(n * M_PI * x);
                    break;
                default:
                    wavefunction[i] = 1.0;
                    break;
            }
        }

        wavefunctions.push_back(wavefunction);
    }

    return wavefunctions;
}

double QuantumConfinementCalculator::calculate_density_of_states(double energy) const {
    // Simplified DOS calculation
    switch (confinement_type_) {
        case ConfinementType::QUANTUM_WELL: {
            // 2D DOS (step function)
            double dos_2d = effective_mass_ * ME / (M_PI * HBAR * HBAR);
            return (energy > get_ground_state_energy()) ? dos_2d : 0.0;
        }
        case ConfinementType::QUANTUM_WIRE: {
            // 1D DOS (1/sqrt(E))
            double ground_energy = get_ground_state_energy();
            if (energy > ground_energy) {
                return sqrt(2.0 * effective_mass_ * ME) / (M_PI * HBAR * sqrt(energy - ground_energy));
            }
            return 0.0;
        }
        case ConfinementType::QUANTUM_DOT: {
            // 0D DOS (delta functions)
            auto energy_levels = calculate_energy_levels(10);
            double dos = 0.0;
            double broadening = 0.01; // eV

            for (double level_energy : energy_levels) {
                // Lorentzian broadening
                dos += (broadening / M_PI) / ((energy - level_energy) * (energy - level_energy) + broadening * broadening);
            }
            return dos;
        }
        default:
            return 0.0;
    }
}

double QuantumConfinementCalculator::get_ground_state_energy() const {
    auto energy_levels = calculate_energy_levels(1);
    return energy_levels.empty() ? 0.0 : energy_levels[0];
}

double QuantumConfinementCalculator::get_level_spacing() const {
    auto energy_levels = calculate_energy_levels(2);
    if (energy_levels.size() < 2) {
        return 0.0;
    }
    return energy_levels[1] - energy_levels[0];
}

bool QuantumConfinementCalculator::is_quantum_confined() const {
    double thermal_energy = KB * 300.0 / EV_TO_J; // kT at room temperature
    double level_spacing = get_level_spacing();
    return level_spacing > thermal_energy;
}

std::vector<double> QuantumConfinementCalculator::solve_infinite_well_1d(int max_levels) const {
    std::vector<double> energy_levels;
    double length = dimensions_[0];

    for (int n = 1; n <= max_levels; ++n) {
        double energy = (n * n * M_PI * M_PI * HBAR * HBAR) /
                       (2.0 * effective_mass_ * ME * length * length);
        energy_levels.push_back(energy / EV_TO_J + band_offset_); // Convert to eV
    }

    return energy_levels;
}

std::vector<double> QuantumConfinementCalculator::solve_infinite_well_2d(int max_levels) const {
    std::vector<double> energy_levels;
    double lx = dimensions_[0];
    double ly = dimensions_[1];

    for (int nx = 1; nx <= max_levels; ++nx) {
        for (int ny = 1; ny <= max_levels; ++ny) {
            if (energy_levels.size() >= static_cast<size_t>(max_levels)) break;

            double energy = (M_PI * M_PI * HBAR * HBAR / (2.0 * effective_mass_ * ME)) *
                           (nx * nx / (lx * lx) + ny * ny / (ly * ly));
            energy_levels.push_back(energy / EV_TO_J + band_offset_); // Convert to eV
        }
        if (energy_levels.size() >= static_cast<size_t>(max_levels)) break;
    }

    std::sort(energy_levels.begin(), energy_levels.end());
    if (energy_levels.size() > static_cast<size_t>(max_levels)) {
        energy_levels.resize(max_levels);
    }

    return energy_levels;
}

std::vector<double> QuantumConfinementCalculator::solve_infinite_well_3d(int max_levels) const {
    std::vector<double> energy_levels;
    double lx = dimensions_[0];
    double ly = dimensions_[1];
    double lz = dimensions_[2];

    for (int nx = 1; nx <= max_levels; ++nx) {
        for (int ny = 1; ny <= max_levels; ++ny) {
            for (int nz = 1; nz <= max_levels; ++nz) {
                if (energy_levels.size() >= static_cast<size_t>(max_levels)) break;

                double energy = (M_PI * M_PI * HBAR * HBAR / (2.0 * effective_mass_ * ME)) *
                               (nx * nx / (lx * lx) + ny * ny / (ly * ly) + nz * nz / (lz * lz));
                energy_levels.push_back(energy / EV_TO_J + band_offset_); // Convert to eV
            }
            if (energy_levels.size() >= static_cast<size_t>(max_levels)) break;
        }
        if (energy_levels.size() >= static_cast<size_t>(max_levels)) break;
    }

    std::sort(energy_levels.begin(), energy_levels.end());
    if (energy_levels.size() > static_cast<size_t>(max_levels)) {
        energy_levels.resize(max_levels);
    }

    return energy_levels;
}

std::vector<double> QuantumConfinementCalculator::solve_harmonic_oscillator(int max_levels) const {
    std::vector<double> energy_levels;

    // Harmonic oscillator frequency (simplified)
    double omega = sqrt(2.0 * band_offset_ * EV_TO_J / (effective_mass_ * ME * dimensions_[0] * dimensions_[0]));

    for (int n = 0; n < max_levels; ++n) {
        double energy = HBAR * omega * (n + 0.5);
        energy_levels.push_back(energy / EV_TO_J); // Convert to eV
    }

    return energy_levels;
}

// TunnelingCalculator implementation
TunnelingCalculator::TunnelingCalculator()
    : min_energy_(0.0), max_energy_(2.0), num_energy_points_(100) {
}

void TunnelingCalculator::set_barrier_profile(const std::vector<double>& position,
                                             const std::vector<double>& potential) {
    position_ = position;
    potential_ = potential;
}

void TunnelingCalculator::set_energy_range(double min_energy, double max_energy, int num_points) {
    min_energy_ = min_energy;
    max_energy_ = max_energy;
    num_energy_points_ = num_points;
}

std::vector<double> TunnelingCalculator::calculate_transmission_coefficients() const {
    std::vector<double> transmission_coefficients;
    transmission_coefficients.reserve(num_energy_points_);

    double energy_step = (max_energy_ - min_energy_) / (num_energy_points_ - 1);

    for (int i = 0; i < num_energy_points_; ++i) {
        double energy = min_energy_ + i * energy_step;
        double transmission = solve_schrodinger_1d(energy);
        transmission_coefficients.push_back(transmission);
    }

    return transmission_coefficients;
}

std::vector<double> TunnelingCalculator::calculate_reflection_coefficients() const {
    auto transmission = calculate_transmission_coefficients();
    std::vector<double> reflection_coefficients;
    reflection_coefficients.reserve(transmission.size());

    for (double T : transmission) {
        reflection_coefficients.push_back(1.0 - T);
    }

    return reflection_coefficients;
}

double TunnelingCalculator::calculate_tunneling_current(double voltage, double temperature) const {
    if (position_.empty() || potential_.empty()) {
        return 0.0;
    }

    // Calculate current using Landauer formula
    double current = 0.0;
    double energy_step = (max_energy_ - min_energy_) / (num_energy_points_ - 1);

    auto transmission = calculate_transmission_coefficients();

    for (int i = 0; i < num_energy_points_; ++i) {
        double energy = min_energy_ + i * energy_step;
        double T = transmission[i];

        // Fermi-Dirac distributions for source and drain
        double kT = KB * temperature / EV_TO_J;
        double f_source = 1.0 / (1.0 + exp((energy - voltage/2.0) / kT));
        double f_drain = 1.0 / (1.0 + exp((energy + voltage/2.0) / kT));

        current += T * (f_source - f_drain) * energy_step;
    }

    // Convert to current (simplified)
    double g0 = 2.0 * Q * Q / (2.0 * M_PI * HBAR); // Quantum of conductance
    return g0 * current;
}

double TunnelingCalculator::get_barrier_height() const {
    if (potential_.empty()) {
        return 0.0;
    }

    return *std::max_element(potential_.begin(), potential_.end());
}

double TunnelingCalculator::get_barrier_width() const {
    if (position_.size() < 2) {
        return 0.0;
    }

    // Find barrier region (simplified)
    double max_potential = get_barrier_height();
    double threshold = 0.5 * max_potential;

    double start_pos = 0.0, end_pos = 0.0;
    bool in_barrier = false;

    for (size_t i = 0; i < potential_.size(); ++i) {
        if (potential_[i] > threshold && !in_barrier) {
            start_pos = position_[i];
            in_barrier = true;
        } else if (potential_[i] <= threshold && in_barrier) {
            end_pos = position_[i];
            break;
        }
    }

    return end_pos - start_pos;
}

double TunnelingCalculator::get_wkb_transmission(double energy) const {
    if (energy >= get_barrier_height()) {
        return 1.0; // Above barrier
    }

    // WKB approximation
    double integral = 0.0;
    double effective_mass = 0.067 * ME; // Default effective mass

    for (size_t i = 1; i < position_.size(); ++i) {
        double dx = position_[i] - position_[i-1];
        double V_avg = 0.5 * (potential_[i] + potential_[i-1]);

        if (V_avg > energy) {
            double k_local = sqrt(2.0 * effective_mass * (V_avg - energy) * EV_TO_J) / HBAR;
            integral += k_local * dx;
        }
    }

    return exp(-2.0 * integral);
}

std::vector<std::complex<double>> TunnelingCalculator::calculate_transfer_matrix(double energy) const {
    // Simplified transfer matrix calculation
    std::vector<std::complex<double>> transfer_matrix(4, std::complex<double>(0.0, 0.0));

    // Identity matrix as starting point
    transfer_matrix[0] = std::complex<double>(1.0, 0.0); // M11
    transfer_matrix[1] = std::complex<double>(0.0, 0.0); // M12
    transfer_matrix[2] = std::complex<double>(0.0, 0.0); // M21
    transfer_matrix[3] = std::complex<double>(1.0, 0.0); // M22

    double effective_mass = 0.067 * ME;

    for (size_t i = 1; i < position_.size(); ++i) {
        double dx = position_[i] - position_[i-1];
        double V = potential_[i];

        std::complex<double> k;
        if (energy > V) {
            k = std::complex<double>(sqrt(2.0 * effective_mass * (energy - V) * EV_TO_J) / HBAR, 0.0);
        } else {
            k = std::complex<double>(0.0, sqrt(2.0 * effective_mass * (V - energy) * EV_TO_J) / HBAR);
        }

        // Propagation matrix for this segment
        std::complex<double> exp_ikx = std::exp(std::complex<double>(0.0, 1.0) * k * dx);
        std::complex<double> exp_minus_ikx = std::exp(std::complex<double>(0.0, -1.0) * k * dx);

        // Update transfer matrix (simplified)
        std::complex<double> new_m11 = transfer_matrix[0] * exp_ikx;
        std::complex<double> new_m22 = transfer_matrix[3] * exp_minus_ikx;

        transfer_matrix[0] = new_m11;
        transfer_matrix[3] = new_m22;
    }

    return transfer_matrix;
}

double TunnelingCalculator::solve_schrodinger_1d(double energy) const {
    if (position_.empty() || potential_.empty()) {
        return 1.0;
    }

    // Use WKB approximation for simplicity
    return get_wkb_transmission(energy);
}

} // namespace quantum
} // namespace simulator
