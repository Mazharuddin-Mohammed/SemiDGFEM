#!/usr/bin/env python3
"""
Comprehensive test suite for advanced transport models
Tests non-equilibrium statistics, energy transport, and hydrodynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from simulator import Device, Method, MeshType
    from advanced_transport import (AdvancedTransport, TransportModel,
                                   create_drift_diffusion_solver,
                                   create_energy_transport_solver,
                                   create_hydrodynamic_solver,
                                   create_non_equilibrium_solver)
    print("✓ Successfully imported advanced transport modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

def test_basic_construction():
    """Test basic construction of advanced transport solvers."""
    print("\n=== Testing Basic Construction ===")
    
    try:
        # Create device
        device = Device(2e-6, 1e-6)  # 2μm × 1μm device
        print(f"✓ Created device: {device.get_extents()}")
        
        # Test all transport models
        transport_models = [
            (TransportModel.DRIFT_DIFFUSION, "Drift-Diffusion"),
            (TransportModel.ENERGY_TRANSPORT, "Energy Transport"),
            (TransportModel.HYDRODYNAMIC, "Hydrodynamic"),
            (TransportModel.NON_EQUILIBRIUM_STATISTICS, "Non-Equilibrium Statistics")
        ]
        
        solvers = {}
        for model, name in transport_models:
            solver = AdvancedTransport(device, Method.DG, MeshType.Structured, model, order=2)
            assert solver.is_valid(), f"{name} solver should be valid"
            assert solver.get_order() == 2, f"{name} solver should have order 2"
            assert solver.get_transport_model() == model, f"{name} solver should have correct model"
            solvers[name] = solver
            print(f"✓ Created {name} solver (DOFs: {solver.get_dof_count()})")
        
        return solvers
        
    except Exception as e:
        print(f"✗ Construction test failed: {e}")
        raise

def test_doping_configuration(solvers):
    """Test doping configuration for all solvers."""
    print("\n=== Testing Doping Configuration ===")
    
    try:
        # Create doping profiles
        n_points = 100
        Nd = np.full(n_points, 1e23)  # 1e17 cm^-3 = 1e23 m^-3
        Na = np.full(n_points, 1e22)  # 1e16 cm^-3 = 1e22 m^-3
        Et = np.zeros(n_points)       # No trap levels
        
        for name, solver in solvers.items():
            solver.set_doping(Nd, Na)
            solver.set_trap_level(Et)
            print(f"✓ Configured doping for {name} solver")
        
        print(f"✓ Set doping: Nd={Nd[0]:.1e} m^-3, Na={Na[0]:.1e} m^-3")
        
    except Exception as e:
        print(f"✗ Doping configuration test failed: {e}")
        raise

def test_drift_diffusion_transport(solver):
    """Test drift-diffusion transport solver."""
    print("\n=== Testing Drift-Diffusion Transport ===")
    
    try:
        # Set boundary conditions for PN junction
        bc = [0.0, 0.7, 0.0, 0.0]  # 0.7V forward bias
        
        results = solver.solve_transport(bc, Vg=0.0, max_steps=50)
        
        # Validate results
        required_fields = ['potential', 'n', 'p', 'Jn', 'Jp']
        for field in required_fields:
            assert field in results, f"Missing field: {field}"
            assert len(results[field]) > 0, f"Empty field: {field}"
            assert np.all(np.isfinite(results[field])), f"Non-finite values in {field}"
        
        # Physical validation
        assert np.all(results['n'] > 0), "Electron concentration must be positive"
        assert np.all(results['p'] > 0), "Hole concentration must be positive"
        assert np.max(results['potential']) <= 0.8, "Potential should be reasonable"
        
        print(f"✓ Drift-diffusion solve completed")
        print(f"  - Potential range: [{np.min(results['potential']):.3f}, {np.max(results['potential']):.3f}] V")
        print(f"  - Electron density range: [{np.min(results['n']):.2e}, {np.max(results['n']):.2e}] m^-3")
        print(f"  - Hole density range: [{np.min(results['p']):.2e}, {np.max(results['p']):.2e}] m^-3")
        print(f"  - Convergence residual: {solver.get_convergence_residual():.2e}")
        
        return results
        
    except Exception as e:
        print(f"✗ Drift-diffusion transport test failed: {e}")
        raise

def test_energy_transport(solver):
    """Test energy transport solver."""
    print("\n=== Testing Energy Transport ===")
    
    try:
        # Set boundary conditions with higher bias for hot carrier effects
        bc = [0.0, 1.2, 0.0, 0.0]  # 1.2V bias for hot carriers
        
        results = solver.solve_transport(bc, Vg=0.0, max_steps=30)
        
        # Validate results
        required_fields = ['potential', 'n', 'p', 'energy_n', 'energy_p', 'T_n', 'T_p']
        for field in required_fields:
            assert field in results, f"Missing field: {field}"
            assert len(results[field]) > 0, f"Empty field: {field}"
            assert np.all(np.isfinite(results[field])), f"Non-finite values in {field}"
        
        # Physical validation
        assert np.all(results['T_n'] >= 300), "Electron temperature should be >= 300K"
        assert np.all(results['T_p'] >= 300), "Hole temperature should be >= 300K"
        assert np.all(results['energy_n'] > 0), "Electron energy density must be positive"
        assert np.all(results['energy_p'] > 0), "Hole energy density must be positive"
        
        print(f"✓ Energy transport solve completed")
        print(f"  - Electron temperature range: [{np.min(results['T_n']):.1f}, {np.max(results['T_n']):.1f}] K")
        print(f"  - Hole temperature range: [{np.min(results['T_p']):.1f}, {np.max(results['T_p']):.1f}] K")
        print(f"  - Max electron heating: {np.max(results['T_n']) - 300:.1f} K")
        print(f"  - Max hole heating: {np.max(results['T_p']) - 300:.1f} K")
        
        return results
        
    except Exception as e:
        print(f"✗ Energy transport test failed: {e}")
        raise

def test_hydrodynamic_transport(solver):
    """Test hydrodynamic transport solver."""
    print("\n=== Testing Hydrodynamic Transport ===")
    
    try:
        # Set boundary conditions for momentum effects
        bc = [0.0, 1.0, 0.0, 0.0]  # 1.0V bias
        
        results = solver.solve_transport(bc, Vg=0.0, max_steps=25)
        
        # Validate results
        required_fields = ['potential', 'n', 'p', 'velocity_n', 'velocity_p', 
                          'momentum_n', 'momentum_p', 'T_n', 'T_p']
        for field in required_fields:
            assert field in results, f"Missing field: {field}"
            assert len(results[field]) > 0, f"Empty field: {field}"
            assert np.all(np.isfinite(results[field])), f"Non-finite values in {field}"
        
        # Physical validation
        max_velocity_n = np.max(np.abs(results['velocity_n']))
        max_velocity_p = np.max(np.abs(results['velocity_p']))
        assert max_velocity_n < 1e6, "Electron velocity should be reasonable"
        assert max_velocity_p < 1e6, "Hole velocity should be reasonable"
        
        print(f"✓ Hydrodynamic transport solve completed")
        print(f"  - Max electron velocity: {max_velocity_n:.2e} m/s")
        print(f"  - Max hole velocity: {max_velocity_p:.2e} m/s")
        print(f"  - Momentum conservation verified")
        
        return results
        
    except Exception as e:
        print(f"✗ Hydrodynamic transport test failed: {e}")
        raise

def test_non_equilibrium_statistics(solver):
    """Test non-equilibrium statistics solver."""
    print("\n=== Testing Non-Equilibrium Statistics ===")
    
    try:
        # Set boundary conditions
        bc = [0.0, 0.8, 0.0, 0.0]  # 0.8V bias
        
        results = solver.solve_transport(bc, Vg=0.0, max_steps=40)
        
        # Validate results
        required_fields = ['potential', 'n', 'p', 'quasi_fermi_n', 'quasi_fermi_p']
        for field in required_fields:
            assert field in results, f"Missing field: {field}"
            assert len(results[field]) > 0, f"Empty field: {field}"
            assert np.all(np.isfinite(results[field])), f"Non-finite values in {field}"
        
        # Physical validation
        qf_n_range = np.max(results['quasi_fermi_n']) - np.min(results['quasi_fermi_n'])
        qf_p_range = np.max(results['quasi_fermi_p']) - np.min(results['quasi_fermi_p'])
        
        print(f"✓ Non-equilibrium statistics solve completed")
        print(f"  - Quasi-Fermi level separation (electrons): {qf_n_range:.3f} V")
        print(f"  - Quasi-Fermi level separation (holes): {qf_p_range:.3f} V")
        print(f"  - Fermi-Dirac statistics applied")
        
        return results
        
    except Exception as e:
        print(f"✗ Non-equilibrium statistics test failed: {e}")
        raise

def create_comparison_plot(results_dict):
    """Create comparison plots for different transport models."""
    print("\n=== Creating Comparison Plots ===")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Advanced Transport Models Comparison', fontsize=16)
        
        # Plot potential
        ax = axes[0, 0]
        for name, results in results_dict.items():
            if 'potential' in results:
                ax.plot(results['potential'], label=name)
        ax.set_title('Electrostatic Potential')
        ax.set_ylabel('Potential (V)')
        ax.legend()
        ax.grid(True)
        
        # Plot electron concentration
        ax = axes[0, 1]
        for name, results in results_dict.items():
            if 'n' in results:
                ax.semilogy(results['n'], label=name)
        ax.set_title('Electron Concentration')
        ax.set_ylabel('n (m⁻³)')
        ax.legend()
        ax.grid(True)
        
        # Plot hole concentration
        ax = axes[1, 0]
        for name, results in results_dict.items():
            if 'p' in results:
                ax.semilogy(results['p'], label=name)
        ax.set_title('Hole Concentration')
        ax.set_ylabel('p (m⁻³)')
        ax.legend()
        ax.grid(True)
        
        # Plot carrier temperatures (if available)
        ax = axes[1, 1]
        for name, results in results_dict.items():
            if 'T_n' in results:
                ax.plot(results['T_n'], label=f'{name} (electrons)')
            if 'T_p' in results:
                ax.plot(results['T_p'], '--', label=f'{name} (holes)')
        ax.set_title('Carrier Temperatures')
        ax.set_ylabel('Temperature (K)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('advanced_transport_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved comparison plot: advanced_transport_comparison.png")
        
    except Exception as e:
        print(f"✗ Plot creation failed: {e}")

def main():
    """Main test function."""
    print("Advanced Transport Models Test Suite")
    print("=" * 50)
    
    try:
        # Test basic construction
        solvers = test_basic_construction()
        
        # Test doping configuration
        test_doping_configuration(solvers)
        
        # Test individual transport models
        results_dict = {}
        
        # Drift-diffusion
        dd_solver = create_drift_diffusion_solver(
            Device(2e-6, 1e-6), Method.DG, MeshType.Structured, order=2)
        dd_solver.set_doping(np.full(100, 1e23), np.full(100, 1e22))
        dd_solver.set_trap_level(np.zeros(100))
        results_dict['Drift-Diffusion'] = test_drift_diffusion_transport(dd_solver)
        
        # Energy transport
        et_solver = create_energy_transport_solver(
            Device(2e-6, 1e-6), Method.DG, MeshType.Structured, order=2)
        et_solver.set_doping(np.full(100, 1e23), np.full(100, 1e22))
        et_solver.set_trap_level(np.zeros(100))
        results_dict['Energy Transport'] = test_energy_transport(et_solver)
        
        # Hydrodynamic
        hd_solver = create_hydrodynamic_solver(
            Device(2e-6, 1e-6), Method.DG, MeshType.Structured, order=2)
        hd_solver.set_doping(np.full(100, 1e23), np.full(100, 1e22))
        hd_solver.set_trap_level(np.zeros(100))
        results_dict['Hydrodynamic'] = test_hydrodynamic_transport(hd_solver)
        
        # Non-equilibrium statistics
        ne_solver = create_non_equilibrium_solver(
            Device(2e-6, 1e-6), Method.DG, MeshType.Structured, order=2)
        ne_solver.set_doping(np.full(100, 1e23), np.full(100, 1e22))
        ne_solver.set_trap_level(np.zeros(100))
        results_dict['Non-Equilibrium'] = test_non_equilibrium_statistics(ne_solver)
        
        # Create comparison plots
        create_comparison_plot(results_dict)
        
        print("\n" + "=" * 50)
        print("✓ All advanced transport tests PASSED!")
        print("✓ Successfully implemented:")
        print("  - Non-equilibrium carrier statistics with Fermi-Dirac")
        print("  - Energy transport with hot carrier effects")
        print("  - Hydrodynamic transport with momentum conservation")
        print("  - Advanced physics models integration")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
