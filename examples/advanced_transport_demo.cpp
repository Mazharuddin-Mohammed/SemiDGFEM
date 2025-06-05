/**
 * Advanced Transport Models Demonstration
 * Shows practical examples of non-equilibrium statistics, energy transport, and hydrodynamics
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include "../src/physics/advanced_physics.hpp"

using namespace SemiDGFEM::Physics;
using namespace std;

void demonstrate_pn_junction_with_fermi_dirac() {
    cout << "\n=== PN Junction with Fermi-Dirac Statistics ===" << endl;
    
    // Create non-equilibrium statistics model
    SiliconProperties props;
    NonEquilibriumStatistics non_eq_stats(props);
    
    // Device parameters
    const size_t n_points = 200;
    const double device_length = 2e-6; // 2 μm
    const double dx = device_length / (n_points - 1);
    
    // Initialize arrays
    vector<double> x(n_points);
    vector<double> potential(n_points);
    vector<double> quasi_fermi_n(n_points);
    vector<double> quasi_fermi_p(n_points);
    vector<double> Nd(n_points);
    vector<double> Na(n_points);
    vector<double> n, p;
    
    // Set up PN junction with heavy doping
    for (size_t i = 0; i < n_points; ++i) {
        x[i] = i * dx;
        
        if (i < n_points / 2) {
            // N-type region (heavily doped for degeneracy effects)
            Nd[i] = 1e25; // 1e19 cm^-3
            Na[i] = 1e20; // Background
        } else {
            // P-type region (heavily doped)
            Nd[i] = 1e20; // Background
            Na[i] = 1e25; // 1e19 cm^-3
        }
        
        // Built-in potential profile (simplified)
        double Vbi = 0.8; // Built-in voltage
        double W = device_length / 4; // Depletion width
        double x_center = device_length / 2;
        
        if (abs(x[i] - x_center) < W) {
            potential[i] = Vbi * (x[i] - x_center) / W;
        } else {
            potential[i] = (x[i] < x_center) ? -Vbi/2 : Vbi/2;
        }
        
        // Quasi-Fermi levels (forward bias)
        double bias = 0.6; // 0.6V forward bias
        quasi_fermi_n[i] = potential[i] + bias/2;
        quasi_fermi_p[i] = potential[i] - bias/2;
    }
    
    // Calculate carrier densities with Fermi-Dirac statistics
    non_eq_stats.calculate_fermi_dirac_densities(
        potential, quasi_fermi_n, quasi_fermi_p, Nd, Na, n, p, 300.0);
    
    // Save results to file
    ofstream file("pn_junction_fermi_dirac.dat");
    file << "# x(μm) V(V) n(m^-3) p(m^-3) Nd(m^-3) Na(m^-3)" << endl;
    
    for (size_t i = 0; i < n_points; ++i) {
        file << scientific << setprecision(6)
             << x[i] * 1e6 << " "
             << potential[i] << " "
             << n[i] << " "
             << p[i] << " "
             << Nd[i] << " "
             << Na[i] << endl;
    }
    file.close();
    
    // Calculate bandgap narrowing effects
    double max_bgn = 0.0;
    for (size_t i = 0; i < n_points; ++i) {
        double N_total = Nd[i] + Na[i];
        double bgn = non_eq_stats.calculate_bandgap_narrowing(N_total);
        max_bgn = max(max_bgn, bgn);
    }
    
    cout << "✓ PN junction simulation completed" << endl;
    cout << "  - Device length: " << device_length * 1e6 << " μm" << endl;
    cout << "  - Max electron density: " << scientific << setprecision(2) 
         << *max_element(n.begin(), n.end()) << " m^-3" << endl;
    cout << "  - Max hole density: " << scientific << setprecision(2)
         << *max_element(p.begin(), p.end()) << " m^-3" << endl;
    cout << "  - Max bandgap narrowing: " << fixed << setprecision(3) << max_bgn << " eV" << endl;
    cout << "  - Results saved to: pn_junction_fermi_dirac.dat" << endl;
}

void demonstrate_hot_carrier_effects() {
    cout << "\n=== Hot Carrier Effects in High-Field Transport ===" << endl;
    
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
    
    // Device parameters
    const size_t n_points = 100;
    const double device_length = 1e-6; // 1 μm
    const double dx = device_length / (n_points - 1);
    
    // Initialize arrays
    vector<double> x(n_points);
    vector<double> electric_field(n_points);
    vector<double> n(n_points, 1e22); // Uniform electron density
    vector<double> p(n_points, 1e21); // Uniform hole density
    vector<double> energy_density_n(n_points);
    vector<double> energy_density_p(n_points);
    vector<double> T_n, T_p;
    
    // Set up high electric field profile
    for (size_t i = 0; i < n_points; ++i) {
        x[i] = i * dx;
        
        // High field in the center region
        double x_center = device_length / 2;
        double field_width = device_length / 4;
        
        if (abs(x[i] - x_center) < field_width) {
            electric_field[i] = 1e6; // 1 MV/m
        } else {
            electric_field[i] = 1e4; // 10 kV/m
        }
        
        // Calculate energy density from field heating
        double field_energy = 0.5 * PhysicalConstants::m0 * 0.26 * pow(electric_field[i] * 1e-12, 2);
        double thermal_energy = (3.0/2.0) * PhysicalConstants::k * 300.0;
        
        energy_density_n[i] = n[i] * (thermal_energy + field_energy);
        energy_density_p[i] = p[i] * (thermal_energy + field_energy * 0.8); // Holes heat less
    }
    
    // Calculate carrier temperatures
    energy_model.calculate_carrier_temperatures(
        energy_density_n, energy_density_p, n, p, T_n, T_p, 300.0);
    
    // Calculate velocity overshoot
    vector<double> overshoot_n(n_points);
    vector<double> overshoot_p(n_points);
    
    for (size_t i = 0; i < n_points; ++i) {
        overshoot_n[i] = energy_model.calculate_velocity_overshoot(electric_field[i], T_n[i], true);
        overshoot_p[i] = energy_model.calculate_velocity_overshoot(electric_field[i], T_p[i], false);
    }
    
    // Save results to file
    ofstream file("hot_carrier_effects.dat");
    file << "# x(μm) E(V/m) T_n(K) T_p(K) overshoot_n overshoot_p" << endl;
    
    for (size_t i = 0; i < n_points; ++i) {
        file << scientific << setprecision(6)
             << x[i] * 1e6 << " "
             << electric_field[i] << " "
             << T_n[i] << " "
             << T_p[i] << " "
             << overshoot_n[i] << " "
             << overshoot_p[i] << endl;
    }
    file.close();
    
    cout << "✓ Hot carrier simulation completed" << endl;
    cout << "  - Max electric field: " << scientific << setprecision(2) 
         << *max_element(electric_field.begin(), electric_field.end()) << " V/m" << endl;
    cout << "  - Max electron temperature: " << fixed << setprecision(1)
         << *max_element(T_n.begin(), T_n.end()) << " K" << endl;
    cout << "  - Max hole temperature: " << fixed << setprecision(1)
         << *max_element(T_p.begin(), T_p.end()) << " K" << endl;
    cout << "  - Max velocity overshoot (e): " << fixed << setprecision(2)
         << *max_element(overshoot_n.begin(), overshoot_n.end()) << "x" << endl;
    cout << "  - Results saved to: hot_carrier_effects.dat" << endl;
}

void demonstrate_hydrodynamic_transport() {
    cout << "\n=== Hydrodynamic Transport with Momentum Conservation ===" << endl;
    
    // Create hydrodynamic model
    HydrodynamicConfig config;
    config.enable_momentum_relaxation = true;
    config.enable_pressure_gradient = true;
    config.enable_heat_flow = true;
    config.momentum_relaxation_time_n = 0.1e-12;
    config.momentum_relaxation_time_p = 0.1e-12;
    config.thermal_conductivity = 150.0;
    
    SiliconProperties props;
    HydrodynamicModel hydro_model(config, props);
    
    // Device parameters
    const size_t n_points = 150;
    const double device_length = 3e-6; // 3 μm
    const double dx = device_length / (n_points - 1);
    
    // Initialize arrays
    vector<double> x(n_points);
    vector<double> n(n_points);
    vector<double> p(n_points);
    vector<double> T_n(n_points);
    vector<double> T_p(n_points);
    vector<double> velocity_n(n_points);
    vector<double> velocity_p(n_points);
    
    // Set up device profile with varying density and temperature
    for (size_t i = 0; i < n_points; ++i) {
        x[i] = i * dx;
        double x_norm = x[i] / device_length;
        
        // Density profile (channel with source/drain)
        if (x_norm < 0.2 || x_norm > 0.8) {
            // Source/drain regions
            n[i] = 1e24;
            p[i] = 1e20;
            T_n[i] = 320.0;
            T_p[i] = 315.0;
        } else {
            // Channel region
            n[i] = 1e22;
            p[i] = 1e22;
            T_n[i] = 350.0 + 100.0 * sin(M_PI * (x_norm - 0.2) / 0.6); // Hot spot
            T_p[i] = 340.0 + 80.0 * sin(M_PI * (x_norm - 0.2) / 0.6);
        }
        
        // Initial velocity profile
        velocity_n[i] = 2e4 * sin(2.0 * M_PI * x_norm);
        velocity_p[i] = -1.5e4 * sin(2.0 * M_PI * x_norm);
    }
    
    // Calculate momentum relaxation
    vector<double> momentum_relaxation_n, momentum_relaxation_p;
    hydro_model.calculate_momentum_relaxation(
        velocity_n, velocity_p, n, p, momentum_relaxation_n, momentum_relaxation_p);
    
    // Calculate pressure gradients
    vector<double> pressure_grad_n, pressure_grad_p;
    hydro_model.calculate_pressure_gradients(n, p, T_n, T_p, pressure_grad_n, pressure_grad_p);
    
    // Calculate heat flow
    vector<double> lattice_temp(n_points, 300.0);
    vector<double> heat_flow_n, heat_flow_p;
    hydro_model.calculate_heat_flow(T_n, T_p, lattice_temp, heat_flow_n, heat_flow_p);
    
    // Save results to file
    ofstream file("hydrodynamic_transport.dat");
    file << "# x(μm) n(m^-3) p(m^-3) T_n(K) T_p(K) v_n(m/s) v_p(m/s) P_grad_n P_grad_p heat_n heat_p" << endl;
    
    for (size_t i = 0; i < n_points; ++i) {
        file << scientific << setprecision(6)
             << x[i] * 1e6 << " "
             << n[i] << " "
             << p[i] << " "
             << T_n[i] << " "
             << T_p[i] << " "
             << velocity_n[i] << " "
             << velocity_p[i] << " "
             << pressure_grad_n[i] << " "
             << pressure_grad_p[i] << " "
             << heat_flow_n[i] << " "
             << heat_flow_p[i] << endl;
    }
    file.close();
    
    cout << "✓ Hydrodynamic simulation completed" << endl;
    cout << "  - Device length: " << device_length * 1e6 << " μm" << endl;
    cout << "  - Max carrier velocity: " << scientific << setprecision(2)
         << max(*max_element(velocity_n.begin(), velocity_n.end()),
                abs(*min_element(velocity_p.begin(), velocity_p.end()))) << " m/s" << endl;
    cout << "  - Max temperature: " << fixed << setprecision(1)
         << max(*max_element(T_n.begin(), T_n.end()),
                *max_element(T_p.begin(), T_p.end())) << " K" << endl;
    cout << "  - Results saved to: hydrodynamic_transport.dat" << endl;
}

int main() {
    cout << "Advanced Transport Models Demonstration" << endl;
    cout << "=======================================" << endl;
    cout << "This demo showcases the implementation of:" << endl;
    cout << "1. Non-equilibrium carrier statistics with Fermi-Dirac" << endl;
    cout << "2. Energy transport with hot carrier effects" << endl;
    cout << "3. Hydrodynamic transport with momentum conservation" << endl;
    
    try {
        demonstrate_pn_junction_with_fermi_dirac();
        demonstrate_hot_carrier_effects();
        demonstrate_hydrodynamic_transport();
        
        cout << "\n=======================================" << endl;
        cout << "✓ All demonstrations completed successfully!" << endl;
        cout << "✓ Data files generated for visualization:" << endl;
        cout << "  - pn_junction_fermi_dirac.dat" << endl;
        cout << "  - hot_carrier_effects.dat" << endl;
        cout << "  - hydrodynamic_transport.dat" << endl;
        cout << "\nUse plotting tools (gnuplot, matplotlib) to visualize results." << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cout << "\n✗ Demonstration failed: " << e.what() << endl;
        return 1;
    }
}
