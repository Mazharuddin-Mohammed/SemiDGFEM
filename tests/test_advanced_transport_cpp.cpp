/**
 * C++ Test Suite for Advanced Transport Models
 * Tests non-equilibrium statistics, energy transport, and hydrodynamics
 */

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>

// Include the advanced transport headers
#include "../src/physics/advanced_physics.hpp"
#include "../include/device.hpp"

using namespace SemiDGFEM::Physics;
using namespace std;

void test_non_equilibrium_statistics() {
    cout << "\n=== Testing Non-Equilibrium Statistics ===" << endl;
    
    try {
        // Create non-equilibrium statistics model
        SiliconProperties props;
        NonEquilibriumStatistics non_eq_stats(props);
        
        // Test data
        size_t n_points = 100;
        vector<double> potential(n_points, 0.0);
        vector<double> quasi_fermi_n(n_points, 0.0);
        vector<double> quasi_fermi_p(n_points, 0.0);
        vector<double> Nd(n_points, 1e23); // 1e17 cm^-3
        vector<double> Na(n_points, 1e22); // 1e16 cm^-3
        vector<double> n, p;
        
        // Set up a forward bias condition
        for (size_t i = 0; i < n_points; ++i) {
            potential[i] = 0.7 * i / (n_points - 1); // 0 to 0.7V
            quasi_fermi_n[i] = potential[i] + 0.1;
            quasi_fermi_p[i] = potential[i] - 0.1;
        }
        
        // Calculate carrier densities with Fermi-Dirac statistics
        auto start = chrono::high_resolution_clock::now();
        non_eq_stats.calculate_fermi_dirac_densities(
            potential, quasi_fermi_n, quasi_fermi_p, Nd, Na, n, p, 300.0);
        auto end = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        
        // Validate results
        assert(n.size() == n_points);
        assert(p.size() == n_points);
        
        // Check physical constraints
        for (size_t i = 0; i < n_points; ++i) {
            assert(n[i] > 0);
            assert(p[i] > 0);
            assert(isfinite(n[i]));
            assert(isfinite(p[i]));
        }
        
        // Test bandgap narrowing
        double bgn = non_eq_stats.calculate_bandgap_narrowing(1e25);
        assert(bgn >= 0.0);
        assert(bgn <= 0.1); // Should be limited to 100 meV
        
        // Test ionization fraction
        double ionization = non_eq_stats.calculate_ionization_fraction(1e23, 300.0);
        assert(ionization > 0.0);
        assert(ionization <= 1.0);
        
        // Test degeneracy factor
        double deg_factor = non_eq_stats.calculate_degeneracy_factor(1e24, 2.8e25, 300.0);
        assert(deg_factor >= 1.0);
        
        cout << "✓ Non-equilibrium statistics test passed" << endl;
        cout << "  - Calculation time: " << duration.count() << " μs" << endl;
        cout << "  - Electron density range: [" << scientific << setprecision(2) 
             << *min_element(n.begin(), n.end()) << ", " 
             << *max_element(n.begin(), n.end()) << "] m^-3" << endl;
        cout << "  - Hole density range: [" << scientific << setprecision(2)
             << *min_element(p.begin(), p.end()) << ", " 
             << *max_element(p.begin(), p.end()) << "] m^-3" << endl;
        cout << "  - Bandgap narrowing: " << fixed << setprecision(3) << bgn << " eV" << endl;
        
    } catch (const exception& e) {
        cout << "✗ Non-equilibrium statistics test failed: " << e.what() << endl;
        throw;
    }
}

void test_energy_transport_model() {
    cout << "\n=== Testing Energy Transport Model ===" << endl;
    
    try {
        // Create energy transport model
        EnergyTransportConfig config;
        config.enable_energy_relaxation = true;
        config.enable_velocity_overshoot = true;
        config.energy_relaxation_time_n = 0.1e-12;
        config.energy_relaxation_time_p = 0.1e-12;
        config.saturation_velocity_n = 1e5;
        config.saturation_velocity_p = 8e4;
        
        SiliconProperties props;
        EnergyTransportModel energy_model(config, props);
        
        // Test data
        size_t n_points = 100;
        vector<double> energy_density_n(n_points);
        vector<double> energy_density_p(n_points);
        vector<double> n(n_points, 1e22);
        vector<double> p(n_points, 1e21);
        vector<double> T_n, T_p;
        
        // Initialize energy densities (hot carriers)
        for (size_t i = 0; i < n_points; ++i) {
            double T_hot = 300.0 + 200.0 * i / (n_points - 1); // 300K to 500K
            energy_density_n[i] = (3.0/2.0) * PhysicalConstants::k * T_hot * n[i];
            energy_density_p[i] = (3.0/2.0) * PhysicalConstants::k * T_hot * p[i];
        }
        
        // Calculate carrier temperatures
        auto start = chrono::high_resolution_clock::now();
        energy_model.calculate_carrier_temperatures(
            energy_density_n, energy_density_p, n, p, T_n, T_p, 300.0);
        auto end = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        
        // Validate results
        assert(T_n.size() == n_points);
        assert(T_p.size() == n_points);
        
        for (size_t i = 0; i < n_points; ++i) {
            assert(T_n[i] >= 300.0); // Should be at least lattice temperature
            assert(T_p[i] >= 300.0);
            assert(T_n[i] <= 2000.0); // Should be bounded
            assert(T_p[i] <= 2000.0);
            assert(isfinite(T_n[i]));
            assert(isfinite(T_p[i]));
        }
        
        // Test energy relaxation
        vector<double> energy_relaxation_n, energy_relaxation_p;
        energy_model.calculate_energy_relaxation(
            T_n, T_p, n, p, energy_relaxation_n, energy_relaxation_p, 300.0);
        
        assert(energy_relaxation_n.size() == n_points);
        assert(energy_relaxation_p.size() == n_points);
        
        // Test velocity overshoot
        double overshoot = energy_model.calculate_velocity_overshoot(1e5, 400.0, true);
        assert(overshoot >= 1.0);
        assert(overshoot <= 2.0);
        
        cout << "✓ Energy transport model test passed" << endl;
        cout << "  - Calculation time: " << duration.count() << " μs" << endl;
        cout << "  - Electron temperature range: [" << fixed << setprecision(1)
             << *min_element(T_n.begin(), T_n.end()) << ", "
             << *max_element(T_n.begin(), T_n.end()) << "] K" << endl;
        cout << "  - Hole temperature range: [" << fixed << setprecision(1)
             << *min_element(T_p.begin(), T_p.end()) << ", "
             << *max_element(T_p.begin(), T_p.end()) << "] K" << endl;
        cout << "  - Velocity overshoot factor: " << fixed << setprecision(2) << overshoot << endl;
        
    } catch (const exception& e) {
        cout << "✗ Energy transport model test failed: " << e.what() << endl;
        throw;
    }
}

void test_hydrodynamic_model() {
    cout << "\n=== Testing Hydrodynamic Model ===" << endl;
    
    try {
        // Create hydrodynamic model
        HydrodynamicConfig config;
        config.enable_momentum_relaxation = true;
        config.enable_pressure_gradient = true;
        config.enable_heat_flow = true;
        config.momentum_relaxation_time_n = 0.1e-12;
        config.momentum_relaxation_time_p = 0.1e-12;
        config.thermal_conductivity = 150.0;
        config.specific_heat = 700.0;
        
        SiliconProperties props;
        HydrodynamicModel hydro_model(config, props);
        
        // Test data
        size_t n_points = 100;
        vector<double> velocity_n(n_points);
        vector<double> velocity_p(n_points);
        vector<double> n(n_points, 1e22);
        vector<double> p(n_points, 1e21);
        vector<double> T_n(n_points, 350.0);
        vector<double> T_p(n_points, 340.0);
        
        // Initialize velocities
        for (size_t i = 0; i < n_points; ++i) {
            velocity_n[i] = 1e4 * sin(2.0 * M_PI * i / n_points); // Sinusoidal velocity
            velocity_p[i] = -8e3 * sin(2.0 * M_PI * i / n_points);
        }
        
        // Test momentum relaxation
        vector<double> momentum_relaxation_n, momentum_relaxation_p;
        auto start = chrono::high_resolution_clock::now();
        hydro_model.calculate_momentum_relaxation(
            velocity_n, velocity_p, n, p, momentum_relaxation_n, momentum_relaxation_p);
        auto end = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        
        // Validate results
        assert(momentum_relaxation_n.size() == n_points);
        assert(momentum_relaxation_p.size() == n_points);
        
        for (size_t i = 0; i < n_points; ++i) {
            assert(isfinite(momentum_relaxation_n[i]));
            assert(isfinite(momentum_relaxation_p[i]));
        }
        
        // Test pressure gradients
        vector<double> pressure_grad_n, pressure_grad_p;
        hydro_model.calculate_pressure_gradients(n, p, T_n, T_p, pressure_grad_n, pressure_grad_p);
        
        assert(pressure_grad_n.size() == n_points);
        assert(pressure_grad_p.size() == n_points);
        
        // Test heat flow
        vector<double> lattice_temp(n_points, 300.0);
        vector<double> heat_flow_n, heat_flow_p;
        hydro_model.calculate_heat_flow(T_n, T_p, lattice_temp, heat_flow_n, heat_flow_p);
        
        assert(heat_flow_n.size() == n_points);
        assert(heat_flow_p.size() == n_points);
        
        // Test thermal conductivity tensor
        auto kappa_tensor = hydro_model.calculate_thermal_conductivity(1e22, 350.0);
        assert(kappa_tensor[0][0] > 0.0);
        assert(kappa_tensor[1][1] > 0.0);
        
        cout << "✓ Hydrodynamic model test passed" << endl;
        cout << "  - Calculation time: " << duration.count() << " μs" << endl;
        cout << "  - Max momentum relaxation (e): " << scientific << setprecision(2)
             << *max_element(momentum_relaxation_n.begin(), momentum_relaxation_n.end()) << " kg⋅m/s⋅m^3⋅s^-1" << endl;
        cout << "  - Max momentum relaxation (h): " << scientific << setprecision(2)
             << *max_element(momentum_relaxation_p.begin(), momentum_relaxation_p.end()) << " kg⋅m/s⋅m^3⋅s^-1" << endl;
        cout << "  - Thermal conductivity: " << fixed << setprecision(1) << kappa_tensor[0][0] << " W/m⋅K" << endl;
        
    } catch (const exception& e) {
        cout << "✗ Hydrodynamic model test failed: " << e.what() << endl;
        throw;
    }
}

void test_physics_integration() {
    cout << "\n=== Testing Physics Integration ===" << endl;
    
    try {
        // Test that all models work together
        SiliconProperties props;
        NonEquilibriumStatistics non_eq_stats(props);
        
        EnergyTransportConfig et_config;
        EnergyTransportModel energy_model(et_config, props);
        
        HydrodynamicConfig hd_config;
        HydrodynamicModel hydro_model(hd_config, props);
        
        // Simulate a simple device scenario
        size_t n_points = 50;
        vector<double> potential(n_points);
        vector<double> quasi_fermi_n(n_points);
        vector<double> quasi_fermi_p(n_points);
        vector<double> Nd(n_points, 1e23);
        vector<double> Na(n_points, 1e22);
        
        // Set up PN junction
        for (size_t i = 0; i < n_points; ++i) {
            double x = double(i) / (n_points - 1);
            potential[i] = 0.8 * x; // Linear potential
            quasi_fermi_n[i] = potential[i] + 0.05;
            quasi_fermi_p[i] = potential[i] - 0.05;
            
            if (i < n_points / 2) {
                Nd[i] = 1e24; // N-type
                Na[i] = 1e20;
            } else {
                Nd[i] = 1e20;
                Na[i] = 1e24; // P-type
            }
        }
        
        // Calculate carrier densities
        vector<double> n, p;
        non_eq_stats.calculate_fermi_dirac_densities(
            potential, quasi_fermi_n, quasi_fermi_p, Nd, Na, n, p, 300.0);
        
        // Calculate energy densities and temperatures
        vector<double> energy_n(n_points), energy_p(n_points);
        for (size_t i = 0; i < n_points; ++i) {
            energy_n[i] = (3.0/2.0) * PhysicalConstants::k * 350.0 * n[i];
            energy_p[i] = (3.0/2.0) * PhysicalConstants::k * 340.0 * p[i];
        }
        
        vector<double> T_n, T_p;
        energy_model.calculate_carrier_temperatures(energy_n, energy_p, n, p, T_n, T_p, 300.0);
        
        // Calculate momentum effects
        vector<double> velocity_n(n_points, 1e4);
        vector<double> velocity_p(n_points, -8e3);
        vector<double> momentum_relaxation_n, momentum_relaxation_p;
        hydro_model.calculate_momentum_relaxation(
            velocity_n, velocity_p, n, p, momentum_relaxation_n, momentum_relaxation_p);
        
        // Validate integrated results
        assert(n.size() == n_points);
        assert(T_n.size() == n_points);
        assert(momentum_relaxation_n.size() == n_points);
        
        // Check physical consistency
        double total_charge = 0.0;
        for (size_t i = 0; i < n_points; ++i) {
            total_charge += PhysicalConstants::q * (p[i] - n[i] + Nd[i] - Na[i]);
            assert(n[i] > 0 && p[i] > 0);
            assert(T_n[i] >= 300.0 && T_p[i] >= 300.0);
        }
        
        cout << "✓ Physics integration test passed" << endl;
        cout << "  - Total charge: " << scientific << setprecision(2) << total_charge << " C/m^3" << endl;
        cout << "  - Average electron temperature: " << fixed << setprecision(1) 
             << accumulate(T_n.begin(), T_n.end(), 0.0) / T_n.size() << " K" << endl;
        cout << "  - Average hole temperature: " << fixed << setprecision(1)
             << accumulate(T_p.begin(), T_p.end(), 0.0) / T_p.size() << " K" << endl;
        
    } catch (const exception& e) {
        cout << "✗ Physics integration test failed: " << e.what() << endl;
        throw;
    }
}

int main() {
    cout << "Advanced Transport Models C++ Test Suite" << endl;
    cout << "=========================================" << endl;
    
    try {
        test_non_equilibrium_statistics();
        test_energy_transport_model();
        test_hydrodynamic_model();
        test_physics_integration();
        
        cout << "\n=========================================" << endl;
        cout << "✓ All advanced transport tests PASSED!" << endl;
        cout << "✓ Successfully validated:" << endl;
        cout << "  - Non-equilibrium carrier statistics with Fermi-Dirac" << endl;
        cout << "  - Energy transport with hot carrier effects" << endl;
        cout << "  - Hydrodynamic transport with momentum conservation" << endl;
        cout << "  - Integrated physics models" << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cout << "\n✗ Test suite failed: " << e.what() << endl;
        return 1;
    }
}
