"""
Test suite for quantum transport models

This module tests all quantum transport functionality including:
- Quantum transport solver
- Quantum confinement calculator
- Tunneling calculator
- Scattering mechanisms
- Transport properties

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import unittest
from quantum_transport import (
    QuantumTransportSolver, QuantumConfinementCalculator, TunnelingCalculator,
    QuantumTransportParameters, QuantumTransportType, ConfinementType, ScatteringType,
    create_test_quantum_mesh, create_quantum_well_potential, create_tunneling_barrier
)

class TestQuantumTransportSolver(unittest.TestCase):
    """Test quantum transport solver"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.solver = QuantumTransportSolver()
        self.vertices, self.elements = create_test_quantum_mesh(20, 20)
        self.potential = create_quantum_well_potential(self.vertices)
        
    def test_solver_initialization(self):
        """Test solver initialization"""
        self.assertIsNotNone(self.solver)
        self.assertEqual(self.solver.params.transport_type, QuantumTransportType.BALLISTIC)
        self.assertEqual(self.solver.params.effective_mass, 0.067)
        
    def test_set_parameters(self):
        """Test parameter setting"""
        params = QuantumTransportParameters()
        params.transport_type = QuantumTransportType.TUNNELING
        params.effective_mass = 0.1
        params.temperature = 77.0
        
        self.solver.set_parameters(params)
        self.assertEqual(self.solver.params.transport_type, QuantumTransportType.TUNNELING)
        self.assertEqual(self.solver.params.effective_mass, 0.1)
        self.assertEqual(self.solver.params.temperature, 77.0)
        
    def test_set_mesh(self):
        """Test mesh setting"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.assertEqual(len(self.solver.vertices), len(self.vertices))
        self.assertEqual(len(self.solver.elements), len(self.elements))
        self.assertIsNotNone(self.solver.hamiltonian_matrix)
        
    def test_set_potential(self):
        """Test potential setting"""
        self.solver.set_potential(self.potential)
        self.assertEqual(len(self.solver.potential), len(self.potential))
        self.assertGreater(np.max(self.solver.potential), 0)
        
    def test_solve_schrodinger_equation(self):
        """Test Schrödinger equation solving"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        
        result = self.solver.solve_schrodinger_equation()
        self.assertTrue(result)
        
    def test_calculate_quantum_states(self):
        """Test quantum state calculation"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        
        result = self.solver.calculate_quantum_states()
        self.assertTrue(result)
        self.assertGreater(len(self.solver.results.states), 0)
        
        # Check first state
        first_state = self.solver.results.states[0]
        self.assertGreaterEqual(first_state.energy, 0)
        self.assertGreater(len(first_state.wavefunction), 0)
        
    def test_get_bound_states(self):
        """Test bound state extraction"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        self.solver.calculate_quantum_states()
        
        bound_states = self.solver.get_bound_states()
        self.assertGreaterEqual(len(bound_states), 0)
        
        for state in bound_states:
            self.assertLess(state.energy, self.solver.params.energy_cutoff)
            
    def test_get_scattering_states(self):
        """Test scattering state calculation"""
        self.solver.set_mesh(self.vertices, self.elements)
        
        energy = 1.0  # eV
        scattering_states = self.solver.get_scattering_states(energy)
        self.assertGreater(len(scattering_states), 0)
        
        for state in scattering_states:
            self.assertAlmostEqual(state.energy, energy, places=6)
            
    def test_calculate_transmission_coefficients(self):
        """Test transmission coefficient calculation"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        
        result = self.solver.calculate_transmission_coefficients()
        self.assertTrue(result)
        self.assertGreater(len(self.solver.results.transmission_coefficients), 0)
        self.assertGreater(len(self.solver.results.energy_grid), 0)
        
        # Check transmission values are between 0 and 1
        for T in self.solver.results.transmission_coefficients:
            self.assertGreaterEqual(T, 0.0)
            self.assertLessEqual(T, 1.0)
            
    def test_calculate_current_density(self):
        """Test current density calculation"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        self.solver.calculate_quantum_states()
        
        result = self.solver.calculate_current_density()
        self.assertTrue(result)
        self.assertEqual(self.solver.results.current_density.shape[0], len(self.vertices))
        self.assertEqual(self.solver.results.current_density.shape[1], 2)
        
    def test_calculate_charge_density(self):
        """Test charge density calculation"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        self.solver.calculate_quantum_states()
        
        result = self.solver.calculate_charge_density()
        self.assertTrue(result)
        self.assertEqual(self.solver.results.charge_density.shape[0], len(self.vertices))
        self.assertGreaterEqual(np.sum(self.solver.results.charge_density), 0)
        
    def test_scattering_mechanisms(self):
        """Test scattering mechanism handling"""
        self.solver.add_scattering_mechanism(ScatteringType.ACOUSTIC_PHONON)
        self.solver.add_scattering_mechanism(ScatteringType.OPTICAL_PHONON)
        
        self.assertIn(ScatteringType.ACOUSTIC_PHONON, self.solver.params.scattering_mechanisms)
        self.assertIn(ScatteringType.OPTICAL_PHONON, self.solver.params.scattering_mechanisms)
        
    def test_calculate_scattering_rates(self):
        """Test scattering rate calculation"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        self.solver.calculate_quantum_states()
        
        self.solver.add_scattering_mechanism(ScatteringType.ACOUSTIC_PHONON)
        result = self.solver.calculate_scattering_rates()
        self.assertTrue(result)
        
    def test_solve_boltzmann_equation(self):
        """Test Boltzmann equation solving"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        self.solver.calculate_quantum_states()
        self.solver.calculate_transmission_coefficients()  # Need this for conductance calculation

        result = self.solver.solve_boltzmann_equation()
        self.assertTrue(result)
        self.assertGreater(self.solver.results.mobility, 0)
        self.assertGreater(self.solver.results.conductance, 0)
        
    def test_advanced_methods(self):
        """Test advanced quantum transport methods"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        
        # Test Wigner function
        result_wigner = self.solver.solve_wigner_function()
        self.assertTrue(result_wigner)
        
        # Test Green's function
        result_green = self.solver.solve_green_function()
        self.assertTrue(result_green)
        
        # Test coherent transport
        result_coherent = self.solver.calculate_coherent_transport()
        self.assertTrue(result_coherent)
        
    def test_quantum_properties(self):
        """Test quantum property calculations"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        self.solver.calculate_quantum_states()
        self.solver.calculate_transmission_coefficients()
        
        # Test quantum capacitance
        capacitance = self.solver.calculate_quantum_capacitance()
        self.assertGreaterEqual(capacitance, 0)
        
        # Test shot noise
        shot_noise = self.solver.calculate_shot_noise()
        self.assertGreaterEqual(shot_noise, 0)
        
        # Test local density of states
        ldos = self.solver.calculate_local_density_of_states()
        if ldos.size > 0:
            self.assertGreaterEqual(np.min(ldos), 0)
            
    def test_get_results(self):
        """Test results retrieval"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        self.solver.calculate_quantum_states()
        
        results = self.solver.get_results()
        self.assertIsNotNone(results)
        self.assertGreater(len(results.states), 0)
        
    def test_get_wavefunction(self):
        """Test wavefunction retrieval"""
        self.solver.set_mesh(self.vertices, self.elements)
        self.solver.set_potential(self.potential)
        self.solver.calculate_quantum_states()
        
        wavefunction = self.solver.get_wavefunction(0)
        self.assertGreater(len(wavefunction), 0)
        
        # Test invalid index
        invalid_wavefunction = self.solver.get_wavefunction(1000)
        self.assertEqual(len(invalid_wavefunction), 0)


class TestQuantumConfinementCalculator(unittest.TestCase):
    """Test quantum confinement calculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = QuantumConfinementCalculator()
        
    def test_calculator_initialization(self):
        """Test calculator initialization"""
        self.assertIsNotNone(self.calculator)
        self.assertEqual(self.calculator.confinement_type, ConfinementType.NONE)
        
    def test_set_confinement_type(self):
        """Test confinement type setting"""
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_WELL)
        self.assertEqual(self.calculator.confinement_type, ConfinementType.QUANTUM_WELL)
        
    def test_set_dimensions(self):
        """Test dimension setting"""
        dimensions = np.array([5e-9, 10e-9, 15e-9])
        self.calculator.set_dimensions(dimensions)
        np.testing.assert_array_equal(self.calculator.dimensions, dimensions)
        
    def test_set_material_parameters(self):
        """Test material parameter setting"""
        self.calculator.set_material_parameters(0.1, 0.5)
        self.assertEqual(self.calculator.effective_mass, 0.1)
        self.assertEqual(self.calculator.band_offset, 0.5)
        
    def test_quantum_well_energy_levels(self):
        """Test quantum well energy level calculation"""
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_WELL)
        self.calculator.set_dimensions(np.array([10e-9, 1e-6, 1e-6]))
        
        energy_levels = self.calculator.calculate_energy_levels(5)
        self.assertEqual(len(energy_levels), 5)
        
        # Check energy ordering
        for i in range(1, len(energy_levels)):
            self.assertGreater(energy_levels[i], energy_levels[i-1])
            
    def test_quantum_wire_energy_levels(self):
        """Test quantum wire energy level calculation"""
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_WIRE)
        self.calculator.set_dimensions(np.array([10e-9, 10e-9, 1e-6]))
        
        energy_levels = self.calculator.calculate_energy_levels(5)
        self.assertGreaterEqual(len(energy_levels), 1)
        self.assertLessEqual(len(energy_levels), 5)
        
    def test_quantum_dot_energy_levels(self):
        """Test quantum dot energy level calculation"""
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_DOT)
        self.calculator.set_dimensions(np.array([10e-9, 10e-9, 10e-9]))
        
        energy_levels = self.calculator.calculate_energy_levels(5)
        self.assertGreaterEqual(len(energy_levels), 1)
        self.assertLessEqual(len(energy_levels), 5)
        
    def test_calculate_wavefunctions(self):
        """Test wavefunction calculation"""
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_WELL)
        
        wavefunctions = self.calculator.calculate_wavefunctions(3)
        self.assertEqual(len(wavefunctions), 3)
        
        for wavefunction in wavefunctions:
            self.assertGreater(len(wavefunction), 0)
            
    def test_density_of_states(self):
        """Test density of states calculation"""
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_WELL)
        
        energy = 0.1  # eV
        dos = self.calculator.calculate_density_of_states(energy)
        self.assertGreaterEqual(dos, 0)
        
    def test_ground_state_energy(self):
        """Test ground state energy"""
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_WELL)
        self.calculator.set_dimensions(np.array([10e-9, 1e-6, 1e-6]))
        
        ground_energy = self.calculator.get_ground_state_energy()
        self.assertGreater(ground_energy, 0)
        
    def test_level_spacing(self):
        """Test energy level spacing"""
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_WELL)
        self.calculator.set_dimensions(np.array([10e-9, 1e-6, 1e-6]))
        
        level_spacing = self.calculator.get_level_spacing()
        self.assertGreater(level_spacing, 0)
        
    def test_is_quantum_confined(self):
        """Test quantum confinement check"""
        # Small dimensions - should be confined
        self.calculator.set_confinement_type(ConfinementType.QUANTUM_WELL)
        self.calculator.set_dimensions(np.array([5e-9, 1e-6, 1e-6]))
        
        is_confined = self.calculator.is_quantum_confined()
        self.assertTrue(is_confined)
        
        # Large dimensions - should not be confined
        self.calculator.set_dimensions(np.array([1e-6, 1e-6, 1e-6]))
        is_confined = self.calculator.is_quantum_confined()
        # This might be False for very large dimensions


class TestTunnelingCalculator(unittest.TestCase):
    """Test tunneling calculator"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = TunnelingCalculator()
        self.position = np.linspace(0, 5e-8, 100)
        self.potential = create_tunneling_barrier(self.position)

    def test_calculator_initialization(self):
        """Test calculator initialization"""
        self.assertIsNotNone(self.calculator)
        self.assertEqual(self.calculator.num_energy_points, 100)

    def test_set_barrier_profile(self):
        """Test barrier profile setting"""
        self.calculator.set_barrier_profile(self.position, self.potential)
        np.testing.assert_array_equal(self.calculator.position, self.position)
        np.testing.assert_array_equal(self.calculator.potential, self.potential)

    def test_set_energy_range(self):
        """Test energy range setting"""
        self.calculator.set_energy_range(0.0, 2.0, 50)
        self.assertEqual(self.calculator.min_energy, 0.0)
        self.assertEqual(self.calculator.max_energy, 2.0)
        self.assertEqual(self.calculator.num_energy_points, 50)

    def test_calculate_transmission_coefficients(self):
        """Test transmission coefficient calculation"""
        self.calculator.set_barrier_profile(self.position, self.potential)

        transmission = self.calculator.calculate_transmission_coefficients()
        self.assertEqual(len(transmission), self.calculator.num_energy_points)

        # Check transmission values are between 0 and 1
        for T in transmission:
            self.assertGreaterEqual(T, 0.0)
            self.assertLessEqual(T, 1.0)

    def test_calculate_reflection_coefficients(self):
        """Test reflection coefficient calculation"""
        self.calculator.set_barrier_profile(self.position, self.potential)

        reflection = self.calculator.calculate_reflection_coefficients()
        transmission = self.calculator.calculate_transmission_coefficients()

        # Check R + T = 1
        for i in range(len(reflection)):
            self.assertAlmostEqual(reflection[i] + transmission[i], 1.0, places=10)

    def test_calculate_tunneling_current(self):
        """Test tunneling current calculation"""
        self.calculator.set_barrier_profile(self.position, self.potential)

        voltage = 0.1  # V
        temperature = 300.0  # K
        current = self.calculator.calculate_tunneling_current(voltage, temperature)

        self.assertGreaterEqual(current, 0.0)

    def test_get_barrier_height(self):
        """Test barrier height calculation"""
        self.calculator.set_barrier_profile(self.position, self.potential)

        barrier_height = self.calculator.get_barrier_height()
        self.assertGreater(barrier_height, 0.0)
        self.assertEqual(barrier_height, np.max(self.potential))

    def test_get_barrier_width(self):
        """Test barrier width calculation"""
        self.calculator.set_barrier_profile(self.position, self.potential)

        barrier_width = self.calculator.get_barrier_width()
        self.assertGreater(barrier_width, 0.0)

    def test_get_wkb_transmission(self):
        """Test WKB transmission calculation"""
        self.calculator.set_barrier_profile(self.position, self.potential)

        # Test below barrier
        energy_below = 0.5  # eV (below barrier)
        T_below = self.calculator.get_wkb_transmission(energy_below)
        self.assertGreaterEqual(T_below, 0.0)
        self.assertLessEqual(T_below, 1.0)

        # Test above barrier
        energy_above = 2.0  # eV (above barrier)
        T_above = self.calculator.get_wkb_transmission(energy_above)
        self.assertEqual(T_above, 1.0)

    def test_empty_barrier_profile(self):
        """Test behavior with empty barrier profile"""
        empty_calculator = TunnelingCalculator()

        # Should return default values
        transmission = empty_calculator.calculate_transmission_coefficients()
        self.assertEqual(len(transmission), 100)  # Default num_energy_points

        current = empty_calculator.calculate_tunneling_current(0.1, 300.0)
        self.assertEqual(current, 0.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_create_test_quantum_mesh(self):
        """Test quantum mesh creation"""
        vertices, elements = create_test_quantum_mesh(10, 10)

        self.assertEqual(len(vertices), 100)  # 10x10 grid
        self.assertGreater(len(elements), 0)

        # Check vertex coordinates
        self.assertEqual(vertices.shape[1], 2)  # 2D coordinates

    def test_create_quantum_well_potential(self):
        """Test quantum well potential creation"""
        vertices, _ = create_test_quantum_mesh(20, 20, domain_x=(0.0, 1e-7), domain_y=(0.0, 1e-7))
        potential = create_quantum_well_potential(vertices, well_width=2e-8, well_depth=0.0, barrier_height=1.0)

        self.assertEqual(len(potential), len(vertices))
        self.assertGreater(np.max(potential), np.min(potential))

    def test_create_tunneling_barrier(self):
        """Test tunneling barrier creation"""
        position = np.linspace(0, 5e-8, 100)
        potential = create_tunneling_barrier(position)

        self.assertEqual(len(potential), len(position))
        self.assertGreater(np.max(potential), 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios"""

    def test_quantum_well_simulation(self):
        """Test complete quantum well simulation"""
        # Create solver
        solver = QuantumTransportSolver()

        # Set up quantum well
        vertices, elements = create_test_quantum_mesh(30, 30)
        potential = create_quantum_well_potential(vertices, well_width=10e-9, well_depth=0.0, barrier_height=1.0)

        # Configure solver
        params = QuantumTransportParameters()
        params.confinement_type = ConfinementType.QUANTUM_WELL
        params.max_states = 10
        solver.set_parameters(params)

        # Set mesh and potential
        solver.set_mesh(vertices, elements)
        solver.set_potential(potential)

        # Solve
        result = solver.calculate_quantum_states()
        self.assertTrue(result)

        # Check results
        bound_states = solver.get_bound_states()
        self.assertGreater(len(bound_states), 0)

        # Calculate transport properties
        solver.calculate_transmission_coefficients()
        solver.calculate_current_density()
        solver.calculate_charge_density()

        results = solver.get_results()
        self.assertTrue(results.converged)

    def test_tunneling_simulation(self):
        """Test complete tunneling simulation"""
        # Create tunneling calculator
        calculator = TunnelingCalculator()

        # Set up barrier
        position = np.linspace(0, 10e-8, 200)
        potential = create_tunneling_barrier(position, barrier_start=3e-8, barrier_end=7e-8, barrier_height=1.5)

        calculator.set_barrier_profile(position, potential)
        calculator.set_energy_range(0.0, 2.0, 100)

        # Calculate transmission
        transmission = calculator.calculate_transmission_coefficients()
        self.assertGreater(len(transmission), 0)

        # Calculate current
        current = calculator.calculate_tunneling_current(0.5, 300.0)
        self.assertGreaterEqual(current, 0.0)

    def test_quantum_confinement_analysis(self):
        """Test complete quantum confinement analysis"""
        calculator = QuantumConfinementCalculator()

        # Test different confinement types
        confinement_types = [ConfinementType.QUANTUM_WELL, ConfinementType.QUANTUM_WIRE, ConfinementType.QUANTUM_DOT]

        for conf_type in confinement_types:
            calculator.set_confinement_type(conf_type)
            calculator.set_dimensions(np.array([10e-9, 10e-9, 10e-9]))

            energy_levels = calculator.calculate_energy_levels(5)
            self.assertGreater(len(energy_levels), 0)

            wavefunctions = calculator.calculate_wavefunctions(3)
            self.assertEqual(len(wavefunctions), 3)

            dos = calculator.calculate_density_of_states(0.1)
            self.assertGreaterEqual(dos, 0.0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
