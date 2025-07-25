"""
Quantum Transport Tutorial for SemiDGFEM

This tutorial demonstrates comprehensive quantum transport modeling capabilities:
1. Quantum Well Simulation
2. Quantum Wire Analysis
3. Quantum Dot Calculations
4. Tunneling Transport
5. Scattering Mechanisms

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import matplotlib.pyplot as plt
from quantum_transport import (
    QuantumTransportSolver, QuantumConfinementCalculator, TunnelingCalculator,
    QuantumTransportParameters, QuantumTransportType, ConfinementType, ScatteringType,
    create_test_quantum_mesh, create_quantum_well_potential, create_tunneling_barrier
)

def tutorial_1_quantum_well_simulation():
    """Tutorial 1: Quantum Well Simulation"""
    print("=" * 60)
    print("Tutorial 1: Quantum Well Simulation")
    print("=" * 60)
    
    # Create quantum transport solver
    solver = QuantumTransportSolver()
    
    # Set up parameters for quantum well
    params = QuantumTransportParameters()
    params.transport_type = QuantumTransportType.BALLISTIC
    params.confinement_type = ConfinementType.QUANTUM_WELL
    params.effective_mass = 0.067  # GaAs effective mass
    params.temperature = 300.0     # Room temperature
    params.fermi_level = 0.5       # eV
    params.max_states = 20
    params.energy_cutoff = 2.0
    
    solver.set_parameters(params)
    
    # Create mesh for quantum well
    vertices, elements = create_test_quantum_mesh(50, 30, 
                                                domain_x=(0.0, 100e-9), 
                                                domain_y=(0.0, 50e-9))
    
    # Create quantum well potential
    potential = create_quantum_well_potential(vertices, 
                                            well_width=20e-9, 
                                            well_depth=0.0, 
                                            barrier_height=1.0)
    
    # Set mesh and potential
    solver.set_mesh(vertices, elements)
    solver.set_potential(potential)
    
    # Solve quantum states
    print("Solving Schrödinger equation...")
    success = solver.calculate_quantum_states()
    
    if success:
        print(f"✓ Successfully calculated {len(solver.results.states)} quantum states")
        
        # Get bound states
        bound_states = solver.get_bound_states()
        print(f"✓ Found {len(bound_states)} bound states")
        
        # Print first few energy levels
        print("\nEnergy levels (eV):")
        for i, state in enumerate(bound_states[:5]):
            print(f"  E_{i+1} = {state.energy:.4f} eV")
            
        # Calculate transport properties
        print("\nCalculating transport properties...")
        solver.calculate_transmission_coefficients()
        solver.calculate_current_density()
        solver.calculate_charge_density()
        
        print(f"✓ Transmission coefficients calculated")
        print(f"✓ Current density calculated")
        print(f"✓ Charge density calculated")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Plot potential and energy levels
        plt.subplot(2, 2, 1)
        x_coords = vertices[:, 0] * 1e9  # Convert to nm
        plt.plot(x_coords, potential, 'b-', linewidth=2, label='Potential')
        for i, state in enumerate(bound_states[:3]):
            plt.axhline(y=state.energy, color='r', linestyle='--', alpha=0.7, 
                       label=f'E_{i+1}' if i < 3 else '')
        plt.xlabel('Position (nm)')
        plt.ylabel('Energy (eV)')
        plt.title('Quantum Well Potential and Energy Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot first wavefunction
        plt.subplot(2, 2, 2)
        if len(bound_states) > 0:
            wavefunction = np.real(bound_states[0].wavefunction)
            plt.plot(x_coords, wavefunction**2, 'g-', linewidth=2, label='|ψ₁|²')
            plt.xlabel('Position (nm)')
            plt.ylabel('Probability Density')
            plt.title('Ground State Wavefunction')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot transmission function
        plt.subplot(2, 2, 3)
        if solver.results.energy_grid.size > 0:
            plt.plot(solver.results.energy_grid, solver.results.transmission_function, 
                    'purple', linewidth=2)
            plt.xlabel('Energy (eV)')
            plt.ylabel('Transmission')
            plt.title('Transmission Function')
            plt.grid(True, alpha=0.3)
        
        # Plot charge density
        plt.subplot(2, 2, 4)
        if solver.results.charge_density.size > 0:
            charge_1d = solver.results.charge_density[:, 0]
            plt.plot(x_coords, charge_1d, 'orange', linewidth=2)
            plt.xlabel('Position (nm)')
            plt.ylabel('Charge Density')
            plt.title('Quantum Charge Density')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantum_well_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Results plotted and saved as 'quantum_well_simulation.png'")
        
    else:
        print("✗ Failed to solve quantum states")
    
    print("\nTutorial 1 completed!\n")


def tutorial_2_quantum_confinement_analysis():
    """Tutorial 2: Quantum Confinement Analysis"""
    print("=" * 60)
    print("Tutorial 2: Quantum Confinement Analysis")
    print("=" * 60)
    
    # Create confinement calculator
    calculator = QuantumConfinementCalculator()
    
    # Set material parameters
    calculator.set_material_parameters(effective_mass=0.067, band_offset=0.0)
    
    # Analyze different confinement types
    confinement_types = [
        (ConfinementType.QUANTUM_WELL, "Quantum Well (1D confinement)"),
        (ConfinementType.QUANTUM_WIRE, "Quantum Wire (2D confinement)"),
        (ConfinementType.QUANTUM_DOT, "Quantum Dot (3D confinement)")
    ]
    
    plt.figure(figsize=(15, 10))
    
    for idx, (conf_type, title) in enumerate(confinement_types):
        print(f"\nAnalyzing {title}...")
        
        calculator.set_confinement_type(conf_type)
        calculator.set_dimensions(np.array([10e-9, 10e-9, 10e-9]))
        
        # Calculate energy levels
        energy_levels = calculator.calculate_energy_levels(10)
        print(f"✓ Calculated {len(energy_levels)} energy levels")
        
        # Calculate wavefunctions
        wavefunctions = calculator.calculate_wavefunctions(5)
        print(f"✓ Calculated {len(wavefunctions)} wavefunctions")
        
        # Calculate density of states
        energy_range = np.linspace(0.0, 1.0, 100)
        dos = np.array([calculator.calculate_density_of_states(E) for E in energy_range])
        
        # Get properties
        ground_energy = calculator.get_ground_state_energy()
        level_spacing = calculator.get_level_spacing()
        is_confined = calculator.is_quantum_confined()
        
        print(f"  Ground state energy: {ground_energy:.4f} eV")
        print(f"  Level spacing: {level_spacing:.4f} eV")
        print(f"  Quantum confined: {is_confined}")
        
        # Plot energy levels
        plt.subplot(3, 3, idx*3 + 1)
        plt.bar(range(len(energy_levels[:8])), energy_levels[:8], 
                color=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'][:len(energy_levels[:8])])
        plt.xlabel('Level Index')
        plt.ylabel('Energy (eV)')
        plt.title(f'{title}\nEnergy Levels')
        plt.grid(True, alpha=0.3)
        
        # Plot first few wavefunctions
        plt.subplot(3, 3, idx*3 + 2)
        x = np.linspace(0, 1, len(wavefunctions[0]))
        for i, wf in enumerate(wavefunctions[:3]):
            plt.plot(x, wf + i*0.5, label=f'ψ_{i+1}')
        plt.xlabel('Normalized Position')
        plt.ylabel('Wavefunction + offset')
        plt.title(f'{title}\nWavefunctions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot density of states
        plt.subplot(3, 3, idx*3 + 3)
        plt.plot(energy_range, dos, linewidth=2)
        plt.xlabel('Energy (eV)')
        plt.ylabel('DOS')
        plt.title(f'{title}\nDensity of States')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_confinement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Results plotted and saved as 'quantum_confinement_analysis.png'")
    print("\nTutorial 2 completed!\n")


def tutorial_3_tunneling_transport():
    """Tutorial 3: Tunneling Transport"""
    print("=" * 60)
    print("Tutorial 3: Tunneling Transport")
    print("=" * 60)
    
    # Create tunneling calculator
    calculator = TunnelingCalculator()
    
    # Create barrier profile
    position = np.linspace(0, 20e-9, 200)  # 20 nm device
    potential = create_tunneling_barrier(position, 
                                       barrier_start=7e-9, 
                                       barrier_end=13e-9, 
                                       barrier_height=1.5)
    
    calculator.set_barrier_profile(position, potential)
    calculator.set_energy_range(0.0, 2.5, 150)
    
    print("Calculating tunneling properties...")
    
    # Calculate transmission and reflection
    transmission = calculator.calculate_transmission_coefficients()
    reflection = calculator.calculate_reflection_coefficients()
    
    # Get barrier properties
    barrier_height = calculator.get_barrier_height()
    barrier_width = calculator.get_barrier_width()
    
    print(f"✓ Barrier height: {barrier_height:.2f} eV")
    print(f"✓ Barrier width: {barrier_width*1e9:.2f} nm")
    print(f"✓ Transmission coefficients calculated")
    
    # Calculate I-V characteristics
    voltages = np.linspace(0.0, 1.0, 21)
    currents = []
    
    print("Calculating I-V characteristics...")
    for V in voltages:
        current = calculator.calculate_tunneling_current(V, 300.0)
        currents.append(current)
    
    currents = np.array(currents)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot barrier profile
    plt.subplot(2, 2, 1)
    plt.plot(position*1e9, potential, 'b-', linewidth=3, label='Barrier')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='E = 0.5 eV')
    plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='E = 1.0 eV')
    plt.xlabel('Position (nm)')
    plt.ylabel('Potential (eV)')
    plt.title('Tunneling Barrier Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot transmission function
    plt.subplot(2, 2, 2)
    energy_grid = np.linspace(calculator.min_energy, calculator.max_energy, len(transmission))
    plt.semilogy(energy_grid, transmission, 'purple', linewidth=2, label='Transmission')
    plt.semilogy(energy_grid, reflection, 'orange', linewidth=2, label='Reflection')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Coefficient')
    plt.title('Transmission and Reflection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot I-V characteristics
    plt.subplot(2, 2, 3)
    plt.plot(voltages, currents*1e12, 'red', linewidth=2, marker='o')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (pA)')
    plt.title('Tunneling I-V Characteristics')
    plt.grid(True, alpha=0.3)
    
    # Plot WKB transmission for different energies
    plt.subplot(2, 2, 4)
    test_energies = np.linspace(0.1, 2.0, 50)
    wkb_transmission = [calculator.get_wkb_transmission(E) for E in test_energies]
    plt.semilogy(test_energies, wkb_transmission, 'green', linewidth=2)
    plt.xlabel('Energy (eV)')
    plt.ylabel('WKB Transmission')
    plt.title('WKB Approximation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tunneling_transport.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Results plotted and saved as 'tunneling_transport.png'")
    print("\nTutorial 3 completed!\n")


def tutorial_4_scattering_mechanisms():
    """Tutorial 4: Scattering Mechanisms"""
    print("=" * 60)
    print("Tutorial 4: Scattering Mechanisms")
    print("=" * 60)
    
    # Create quantum transport solver
    solver = QuantumTransportSolver()
    
    # Set up mesh and potential
    vertices, elements = create_test_quantum_mesh(40, 40)
    potential = create_quantum_well_potential(vertices, well_width=30e-9)
    
    solver.set_mesh(vertices, elements)
    solver.set_potential(potential)
    
    # Calculate initial quantum states
    print("Calculating quantum states...")
    solver.calculate_quantum_states()
    solver.calculate_transmission_coefficients()
    
    # Test different scattering mechanisms
    scattering_types = [
        (ScatteringType.ACOUSTIC_PHONON, "Acoustic Phonon"),
        (ScatteringType.OPTICAL_PHONON, "Optical Phonon"),
        (ScatteringType.IONIZED_IMPURITY, "Ionized Impurity")
    ]
    
    results = {}
    
    for scattering_type, name in scattering_types:
        print(f"\nAnalyzing {name} scattering...")
        
        # Reset solver
        solver_copy = QuantumTransportSolver()
        solver_copy.set_mesh(vertices, elements)
        solver_copy.set_potential(potential)
        solver_copy.calculate_quantum_states()
        solver_copy.calculate_transmission_coefficients()
        
        # Add scattering mechanism
        solver_copy.add_scattering_mechanism(scattering_type)
        
        # Solve Boltzmann equation
        success = solver_copy.solve_boltzmann_equation()
        
        if success:
            results[name] = {
                'mobility': solver_copy.results.mobility,
                'conductance': solver_copy.results.conductance,
                'resistance': solver_copy.results.resistance
            }
            print(f"  ✓ Mobility: {solver_copy.results.mobility:.4e} m²/V·s")
            print(f"  ✓ Conductance: {solver_copy.results.conductance:.4e} S")
            print(f"  ✓ Resistance: {solver_copy.results.resistance:.4e} Ω")
        else:
            print(f"  ✗ Failed to solve with {name} scattering")
    
    # Plot comparison
    if results:
        plt.figure(figsize=(12, 6))
        
        names = list(results.keys())
        mobilities = [results[name]['mobility'] for name in names]
        conductances = [results[name]['conductance'] for name in names]
        
        plt.subplot(1, 2, 1)
        plt.bar(names, mobilities, color=['blue', 'green', 'red'])
        plt.ylabel('Mobility (m²/V·s)')
        plt.title('Mobility vs Scattering Mechanism')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(names, conductances, color=['blue', 'green', 'red'])
        plt.ylabel('Conductance (S)')
        plt.title('Conductance vs Scattering Mechanism')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scattering_mechanisms.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Results plotted and saved as 'scattering_mechanisms.png'")
    
    print("\nTutorial 4 completed!\n")


def tutorial_5_advanced_quantum_effects():
    """Tutorial 5: Advanced Quantum Effects"""
    print("=" * 60)
    print("Tutorial 5: Advanced Quantum Effects")
    print("=" * 60)
    
    # Create quantum transport solver
    solver = QuantumTransportSolver()
    
    # Set up complex potential landscape
    vertices, elements = create_test_quantum_mesh(60, 40, 
                                                domain_x=(0.0, 150e-9), 
                                                domain_y=(0.0, 100e-9))
    
    # Create double quantum well
    x_coords = vertices[:, 0]
    potential = np.ones_like(x_coords) * 1.0  # Barrier regions
    
    # First well
    well1_mask = (x_coords >= 30e-9) & (x_coords <= 50e-9)
    potential[well1_mask] = 0.0
    
    # Second well
    well2_mask = (x_coords >= 100e-9) & (x_coords <= 120e-9)
    potential[well2_mask] = 0.0
    
    # Tunneling barrier between wells
    barrier_mask = (x_coords >= 50e-9) & (x_coords <= 100e-9)
    potential[barrier_mask] = 0.8
    
    solver.set_mesh(vertices, elements)
    solver.set_potential(potential)
    
    print("Solving double quantum well system...")
    
    # Set parameters for coherent transport
    params = QuantumTransportParameters()
    params.transport_type = QuantumTransportType.COHERENT
    params.max_states = 30
    params.energy_cutoff = 1.5
    solver.set_parameters(params)
    
    # Calculate quantum states
    success = solver.calculate_quantum_states()
    
    if success:
        print(f"✓ Calculated {len(solver.results.states)} quantum states")
        
        # Calculate advanced properties
        solver.calculate_transmission_coefficients()
        solver.calculate_coherent_transport()
        
        # Calculate quantum properties
        quantum_capacitance = solver.calculate_quantum_capacitance()
        shot_noise = solver.calculate_shot_noise()
        ldos = solver.calculate_local_density_of_states()
        
        print(f"✓ Quantum capacitance: {quantum_capacitance:.4e} F")
        print(f"✓ Shot noise: {shot_noise:.4e} A²/Hz")
        print(f"✓ Local density of states calculated")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot potential landscape
        plt.subplot(2, 3, 1)
        plt.plot(x_coords*1e9, potential, 'b-', linewidth=2)
        plt.xlabel('Position (nm)')
        plt.ylabel('Potential (eV)')
        plt.title('Double Quantum Well')
        plt.grid(True, alpha=0.3)
        
        # Plot energy levels
        plt.subplot(2, 3, 2)
        bound_states = solver.get_bound_states()
        energies = [state.energy for state in bound_states[:10]]
        plt.bar(range(len(energies)), energies, color='red', alpha=0.7)
        plt.xlabel('State Index')
        plt.ylabel('Energy (eV)')
        plt.title('Energy Levels')
        plt.grid(True, alpha=0.3)
        
        # Plot transmission function
        plt.subplot(2, 3, 3)
        if solver.results.energy_grid.size > 0:
            plt.plot(solver.results.energy_grid, solver.results.transmission_function, 
                    'purple', linewidth=2)
            plt.xlabel('Energy (eV)')
            plt.ylabel('Transmission')
            plt.title('Coherent Transmission')
            plt.grid(True, alpha=0.3)
        
        # Plot selected wavefunctions
        plt.subplot(2, 3, 4)
        for i, state in enumerate(bound_states[:3]):
            wf = np.real(state.wavefunction)
            plt.plot(x_coords*1e9, wf + i*0.5, label=f'ψ_{i+1}')
        plt.xlabel('Position (nm)')
        plt.ylabel('Wavefunction + offset')
        plt.title('Quantum Wavefunctions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot local density of states
        plt.subplot(2, 3, 5)
        if ldos.size > 0:
            plt.plot(x_coords*1e9, ldos, 'orange', linewidth=2)
            plt.xlabel('Position (nm)')
            plt.ylabel('LDOS')
            plt.title('Local Density of States')
            plt.grid(True, alpha=0.3)
        
        # Plot charge density
        plt.subplot(2, 3, 6)
        solver.calculate_charge_density()
        if solver.results.charge_density.size > 0:
            charge_1d = solver.results.charge_density[:, 0]
            plt.plot(x_coords*1e9, charge_1d, 'green', linewidth=2)
            plt.xlabel('Position (nm)')
            plt.ylabel('Charge Density')
            plt.title('Quantum Charge Distribution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_quantum_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Results plotted and saved as 'advanced_quantum_effects.png'")
        
    else:
        print("✗ Failed to solve double quantum well system")
    
    print("\nTutorial 5 completed!\n")


def main():
    """Run all quantum transport tutorials"""
    print("Quantum Transport Tutorial Suite")
    print("=" * 60)
    print("This tutorial demonstrates comprehensive quantum transport modeling")
    print("capabilities in the SemiDGFEM simulator.\n")
    
    try:
        # Run all tutorials
        tutorial_1_quantum_well_simulation()
        tutorial_2_quantum_confinement_analysis()
        tutorial_3_tunneling_transport()
        tutorial_4_scattering_mechanisms()
        tutorial_5_advanced_quantum_effects()
        
        print("=" * 60)
        print("All quantum transport tutorials completed successfully!")
        print("Generated plots:")
        print("  - quantum_well_simulation.png")
        print("  - quantum_confinement_analysis.png")
        print("  - tunneling_transport.png")
        print("  - scattering_mechanisms.png")
        print("  - advanced_quantum_effects.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running tutorials: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
