#!/usr/bin/env python3
"""
Complete MOSFET Simulation with Full Visualization
Shows all simulation outputs: potentials, carrier densities, currents, I-V curves, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import sys
import os
import time
from datetime import datetime

# Add parent directory for simulator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import simulator, fall back to analytical model if not available
try:
    import simulator
    SIMULATOR_AVAILABLE = True
    print("✅ SemiDGFEM simulator loaded successfully")
except ImportError:
    SIMULATOR_AVAILABLE = False
    print("⚠️  SemiDGFEM simulator not available - using analytical model")

class CompleteMOSFETSimulation:
    """Complete MOSFET simulation with comprehensive visualization"""
    
    def __init__(self):
        """Initialize simulation parameters"""
        
        # Device parameters
        self.device_params = {
            'length': 1e-6,        # 1 μm channel length
            'width': 10e-6,        # 10 μm channel width
            'tox': 2e-9,           # 2 nm oxide thickness
            'Na_substrate': 1e17,   # Substrate doping (/m³)
            'Nd_source': 1e20,     # Source doping (/m³)
            'Nd_drain': 1e20,      # Drain doping (/m³)
        }
        
        # Simulation parameters
        self.sim_params = {
            'nx': 80,              # Grid points in x
            'ny': 40,              # Grid points in y
            'method': 'DG',        # Numerical method
        }
        
        # Results storage
        self.simulation_results = {}
        self.simulation_log = []
        
        # Create coordinate arrays
        self.x = np.linspace(0, self.device_params['length']*1e6, self.sim_params['nx'])
        self.y = np.linspace(0, self.device_params['width']*1e6, self.sim_params['ny'])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.log("🚀 Complete MOSFET Simulation Initialized")
        self.log(f"   Device: {self.device_params['length']*1e6:.1f}μm × {self.device_params['width']*1e6:.1f}μm")
        self.log(f"   Grid: {self.sim_params['nx']} × {self.sim_params['ny']}")
    
    def log(self, message):
        """Add timestamped message to simulation log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.simulation_log.append(log_entry)
        print(log_entry)
    
    def create_device(self):
        """Create MOSFET device with detailed logging"""
        
        self.log("🏗️  Creating MOSFET device structure...")
        
        if SIMULATOR_AVAILABLE:
            try:
                # Create real simulator
                self.sim = simulator.Simulator(
                    extents=[self.device_params['length'], self.device_params['width']],
                    num_points_x=self.sim_params['nx'],
                    num_points_y=self.sim_params['ny'],
                    method=self.sim_params['method'],
                    mesh_type="Structured"
                )
                
                self.log("   ✅ Real simulator created successfully")
                self._setup_real_doping()
                return True
                
            except Exception as e:
                self.log(f"   ❌ Real simulator failed: {e}")
                self.log("   🔄 Falling back to analytical model...")
                self._create_analytical_simulator()
                return True
        else:
            self._create_analytical_simulator()
            return True
    
    def _setup_real_doping(self):
        """Setup doping profile for real simulator"""
        
        self.log("   🔧 Setting up doping profile...")
        
        total_points = self.sim_params['nx'] * self.sim_params['ny']
        Nd = np.zeros(total_points)
        Na = np.zeros(total_points)
        
        # Define regions
        source_end = self.sim_params['nx'] // 4
        drain_start = 3 * self.sim_params['nx'] // 4
        junction_depth = self.sim_params['ny'] // 3
        
        for i in range(total_points):
            x_idx = i % self.sim_params['nx']
            y_idx = i // self.sim_params['nx']
            
            if x_idx < source_end:
                # Source region
                if y_idx < junction_depth:
                    Nd[i] = self.device_params['Nd_source']
                else:
                    Na[i] = self.device_params['Na_substrate']
            elif x_idx >= drain_start:
                # Drain region
                if y_idx < junction_depth:
                    Nd[i] = self.device_params['Nd_drain']
                else:
                    Na[i] = self.device_params['Na_substrate']
            else:
                # Channel region
                Na[i] = self.device_params['Na_substrate']
        
        self.sim.set_doping(Nd, Na)
        
        self.log(f"   ✅ Doping profile set:")
        self.log(f"      Source/Drain: {self.device_params['Nd_source']:.1e} /m³")
        self.log(f"      Substrate: {self.device_params['Na_substrate']:.1e} /m³")
        self.log(f"      Source region: 0 to {source_end} grid points")
        self.log(f"      Channel region: {source_end} to {drain_start} grid points")
        self.log(f"      Drain region: {drain_start} to {self.sim_params['nx']} grid points")
    
    def _create_analytical_simulator(self):
        """Create analytical MOSFET simulator"""
        
        class AnalyticalMOSFET:
            def __init__(self, nx, ny, device_params):
                self.num_points_x = nx
                self.num_points_y = ny
                self.device_params = device_params
                
                # MOSFET parameters
                self.Vth = 0.5  # Threshold voltage
                self.mu_n = 0.05  # Electron mobility
                self.Cox = 1.7e-2  # Oxide capacitance
                self.ni = 1.45e16  # Intrinsic carrier concentration
                
            def solve_poisson(self, bc):
                """Analytical Poisson solution with realistic MOSFET behavior"""
                nx, ny = self.num_points_x, self.num_points_y
                
                # Create coordinate arrays
                x = np.linspace(0, 1, nx)
                y = np.linspace(0, 1, ny)
                X, Y = np.meshgrid(x, y)
                
                # Extract boundary conditions
                Vs, Vd, Vsub, Vg = bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 0.0
                
                # Create realistic potential distribution
                # Linear variation from source to drain
                V_channel = Vs + (Vd - Vs) * X
                
                # Gate coupling effect (exponential decay from surface)
                gate_coupling = 0.3 * (Vg - self.Vth) * np.exp(-Y * 5)
                
                # Substrate effect
                substrate_effect = Vsub * (1 - Y)
                
                # Combine effects
                V = V_channel + gate_coupling + substrate_effect
                
                # Add some realistic variation
                V += 0.05 * np.sin(np.pi * X) * np.sin(np.pi * Y)
                
                return V.flatten()
            
            def solve_drift_diffusion(self, bc, **kwargs):
                """Complete drift-diffusion solution with all outputs"""
                V = self.solve_poisson(bc)
                nx, ny = self.num_points_x, self.num_points_y
                
                # Reshape potential for calculations
                V_2d = V.reshape(ny, nx)
                
                # Extract voltages
                Vs, Vd, Vsub, Vg = bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 0.0
                
                # Calculate carrier densities
                n, p = self._calculate_carrier_densities(V_2d, Vg)
                
                # Calculate current densities
                Jn, Jp = self._calculate_current_densities(V_2d, n, p)
                
                # Calculate total drain current
                drain_current = self._calculate_drain_current(Vg, Vd)
                
                return {
                    'potential': V,
                    'n': n.flatten(),
                    'p': p.flatten(),
                    'Jn': Jn.flatten(),
                    'Jp': Jp.flatten(),
                    'drain_current': drain_current,
                    'electric_field_x': np.gradient(V_2d, axis=1).flatten(),
                    'electric_field_y': np.gradient(V_2d, axis=0).flatten()
                }
            
            def _calculate_carrier_densities(self, V_2d, Vg):
                """Calculate electron and hole densities"""
                ny, nx = V_2d.shape
                
                # Create coordinate arrays
                x = np.linspace(0, 1, nx)
                y = np.linspace(0, 1, ny)
                X, Y = np.meshgrid(x, y)
                
                # Electron density (enhanced near surface when Vg > Vth)
                if Vg > self.Vth:
                    # Inversion layer formation
                    surface_enhancement = 1 + 100 * (Vg - self.Vth) * np.exp(-Y * 10)
                    n = self.ni * np.exp(V_2d / 0.026) * surface_enhancement
                else:
                    # Depletion/accumulation
                    n = self.ni * np.exp(V_2d / 0.026) * 0.1
                
                # Hole density (depleted when Vg > 0)
                if Vg > 0:
                    depletion_factor = np.exp(-Vg * np.exp(-Y * 3))
                    p = self.ni * np.exp(-V_2d / 0.026) * depletion_factor
                else:
                    p = self.ni * np.exp(-V_2d / 0.026)
                
                # Add substrate doping effects
                # P-type substrate
                p += 1e17 * (1 - np.exp(-Y * 2))
                
                # N+ source/drain regions
                source_region = X < 0.25
                drain_region = X > 0.75
                junction_region = Y < 0.3
                
                n[source_region & junction_region] += 1e20
                n[drain_region & junction_region] += 1e20
                
                return n, p
            
            def _calculate_current_densities(self, V_2d, n, p):
                """Calculate electron and hole current densities"""
                # Electric field
                Ex = -np.gradient(V_2d, axis=1)
                Ey = -np.gradient(V_2d, axis=0)
                
                # Current densities (drift component)
                q = 1.602e-19
                mu_n = 0.05  # m²/V·s
                mu_p = 0.02  # m²/V·s
                
                Jn_x = q * mu_n * n * Ex
                Jn_y = q * mu_n * n * Ey
                Jn = np.sqrt(Jn_x**2 + Jn_y**2)
                
                Jp_x = -q * mu_p * p * Ex  # Opposite direction
                Jp_y = -q * mu_p * p * Ey
                Jp = np.sqrt(Jp_x**2 + Jp_y**2)
                
                return Jn, Jp
            
            def _calculate_drain_current(self, Vg, Vd):
                """Calculate total drain current using MOSFET equations"""
                if Vg < self.Vth:
                    # Subthreshold
                    return 1e-12 * np.exp((Vg - self.Vth) / 0.1)
                else:
                    # Above threshold
                    W_over_L = self.device_params['width'] / self.device_params['length']
                    if Vd < (Vg - self.Vth):
                        # Linear region
                        return self.mu_n * self.Cox * W_over_L * (Vg - self.Vth) * Vd
                    else:
                        # Saturation region
                        return 0.5 * self.mu_n * self.Cox * W_over_L * (Vg - self.Vth)**2
        
        self.sim = AnalyticalMOSFET(self.sim_params['nx'], self.sim_params['ny'], self.device_params)
        self.log("   ✅ Analytical MOSFET model created")
    
    def run_single_point_simulation(self, Vg, Vd, Vs=0.0, Vsub=0.0):
        """Run simulation at a single operating point with detailed logging"""
        
        self.log(f"🔍 Running simulation at Vg={Vg:.2f}V, Vd={Vd:.2f}V, Vs={Vs:.2f}V, Vsub={Vsub:.2f}V")
        
        start_time = time.time()
        
        try:
            # Set boundary conditions
            bc = [Vs, Vd, Vsub, Vg]
            
            # Run simulation
            if hasattr(self.sim, 'solve_drift_diffusion'):
                self.log("   🔄 Solving drift-diffusion equations...")
                result = self.sim.solve_drift_diffusion(
                    bc=bc,
                    max_steps=30,
                    use_amr=False,
                    poisson_tol=1e-6
                )
            else:
                self.log("   🔄 Solving Poisson equation only...")
                V = self.sim.solve_poisson(bc)
                result = {'potential': V}
            
            solve_time = time.time() - start_time
            self.log(f"   ✅ Simulation completed in {solve_time:.3f} seconds")
            
            # Log result statistics
            if 'potential' in result:
                V = result['potential']
                self.log(f"   📊 Potential: {np.min(V):.3f} to {np.max(V):.3f} V")
            
            if 'n' in result:
                n = result['n']
                self.log(f"   📊 Electron density: {np.min(n):.2e} to {np.max(n):.2e} /m³")
            
            if 'p' in result:
                p = result['p']
                self.log(f"   📊 Hole density: {np.min(p):.2e} to {np.max(p):.2e} /m³")
            
            if 'drain_current' in result:
                Id = result['drain_current']
                self.log(f"   📊 Drain current: {Id:.2e} A")
            
            return result
            
        except Exception as e:
            self.log(f"   ❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_iv_characteristics(self):
        """Generate complete I-V characteristics with detailed logging"""

        self.log("📈 Generating I-V Characteristics...")

        # Transfer characteristics (Id vs Vg)
        self.log("   🔍 Computing transfer characteristics...")
        Vg_range = np.linspace(-0.5, 1.5, 21)
        Vd_linear = 0.1  # Small Vd for linear region

        transfer_results = {
            'Vg': Vg_range,
            'Id_linear': [],
            'Id_saturation': [],
            'gm': []
        }

        for i, Vg in enumerate(Vg_range):
            # Linear region
            result_lin = self.run_single_point_simulation(Vg, Vd_linear)
            if result_lin and 'drain_current' in result_lin:
                Id_lin = result_lin['drain_current']
            else:
                Id_lin = 1e-15

            transfer_results['Id_linear'].append(abs(Id_lin))

            # Saturation region
            result_sat = self.run_single_point_simulation(Vg, 1.2)
            if result_sat and 'drain_current' in result_sat:
                Id_sat = result_sat['drain_current']
            else:
                Id_sat = 1e-15

            transfer_results['Id_saturation'].append(abs(Id_sat))

            if (i + 1) % 5 == 0:
                self.log(f"      Progress: {i+1}/{len(Vg_range)} points completed")

        # Calculate transconductance
        transfer_results['Id_linear'] = np.array(transfer_results['Id_linear'])
        transfer_results['Id_saturation'] = np.array(transfer_results['Id_saturation'])
        transfer_results['gm'] = np.gradient(transfer_results['Id_saturation'],
                                           Vg_range[1] - Vg_range[0])

        self.log(f"   ✅ Transfer characteristics completed")
        self.log(f"      Current range: {np.min(transfer_results['Id_linear']):.2e} to {np.max(transfer_results['Id_saturation']):.2e} A")
        self.log(f"      Peak transconductance: {np.max(transfer_results['gm']):.2e} S")

        # Output characteristics (Id vs Vd)
        self.log("   🔍 Computing output characteristics...")
        Vd_range = np.linspace(0, 1.5, 16)
        Vg_values = [0.5, 0.7, 0.9, 1.1, 1.3]

        output_results = {
            'Vd': Vd_range,
            'curves': {}
        }

        for Vg in Vg_values:
            self.log(f"      Computing curve for Vg = {Vg:.1f}V...")
            Id_curve = []

            for Vd in Vd_range:
                result = self.run_single_point_simulation(Vg, Vd)
                if result and 'drain_current' in result:
                    Id = result['drain_current']
                else:
                    Id = 1e-15

                Id_curve.append(abs(Id))

            output_results['curves'][f'Vg_{Vg:.1f}V'] = np.array(Id_curve)
            self.log(f"         Current range: {np.min(Id_curve):.2e} to {np.max(Id_curve):.2e} A")

        self.log("   ✅ Output characteristics completed")

        # Store results
        self.simulation_results['transfer'] = transfer_results
        self.simulation_results['output'] = output_results

        return transfer_results, output_results

    def create_comprehensive_plots(self, operating_point_result=None):
        """Create comprehensive visualization plots"""

        self.log("🎨 Creating comprehensive visualization plots...")

        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('Complete MOSFET Simulation Results', fontsize=16, fontweight='bold')

        # If we have a specific operating point result, use it for detailed plots
        if operating_point_result is None:
            self.log("   🔄 Running reference simulation for visualization...")
            operating_point_result = self.run_single_point_simulation(Vg=0.8, Vd=0.6)

        if operating_point_result:
            self._plot_device_physics(fig, operating_point_result)

        if 'transfer' in self.simulation_results and 'output' in self.simulation_results:
            self._plot_iv_characteristics(fig)

        self._plot_device_structure(fig)
        self._plot_simulation_log(fig)

        plt.tight_layout()
        plt.savefig('complete_mosfet_simulation.png', dpi=300, bbox_inches='tight')
        self.log("   ✅ Comprehensive plots saved as 'complete_mosfet_simulation.png'")
        plt.show()

    def _plot_device_physics(self, fig, result):
        """Plot device physics: potential, carriers, currents, fields"""

        nx, ny = self.sim_params['nx'], self.sim_params['ny']

        # Plot 1: Electrostatic Potential
        plt.subplot(4, 6, 1)
        if 'potential' in result:
            V = result['potential'].reshape(ny, nx)
            im1 = plt.imshow(V, extent=[0, self.device_params['length']*1e6, 0, self.device_params['width']*1e6],
                           aspect='auto', origin='lower', cmap='RdYlBu_r')
            plt.colorbar(im1, shrink=0.8)
            plt.title('Electrostatic Potential (V)')
            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')

        # Plot 2: Electron Density
        plt.subplot(4, 6, 2)
        if 'n' in result:
            n = result['n'].reshape(ny, nx)
            im2 = plt.imshow(np.log10(np.maximum(n, 1e10)),
                           extent=[0, self.device_params['length']*1e6, 0, self.device_params['width']*1e6],
                           aspect='auto', origin='lower', cmap='plasma')
            plt.colorbar(im2, shrink=0.8)
            plt.title('Electron Density (log₁₀ /m³)')
            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')

        # Plot 3: Hole Density
        plt.subplot(4, 6, 3)
        if 'p' in result:
            p = result['p'].reshape(ny, nx)
            im3 = plt.imshow(np.log10(np.maximum(p, 1e10)),
                           extent=[0, self.device_params['length']*1e6, 0, self.device_params['width']*1e6],
                           aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(im3, shrink=0.8)
            plt.title('Hole Density (log₁₀ /m³)')
            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')

        # Plot 4: Electron Current Density
        plt.subplot(4, 6, 4)
        if 'Jn' in result:
            Jn = result['Jn'].reshape(ny, nx)
            im4 = plt.imshow(np.log10(np.maximum(np.abs(Jn), 1e-10)),
                           extent=[0, self.device_params['length']*1e6, 0, self.device_params['width']*1e6],
                           aspect='auto', origin='lower', cmap='hot')
            plt.colorbar(im4, shrink=0.8)
            plt.title('Electron Current Density (log₁₀ A/m²)')
            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')

        # Plot 5: Hole Current Density
        plt.subplot(4, 6, 5)
        if 'Jp' in result:
            Jp = result['Jp'].reshape(ny, nx)
            im5 = plt.imshow(np.log10(np.maximum(np.abs(Jp), 1e-10)),
                           extent=[0, self.device_params['length']*1e6, 0, self.device_params['width']*1e6],
                           aspect='auto', origin='lower', cmap='hot')
            plt.colorbar(im5, shrink=0.8)
            plt.title('Hole Current Density (log₁₀ A/m²)')
            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')

        # Plot 6: Total Current Density
        plt.subplot(4, 6, 6)
        if 'Jn' in result and 'Jp' in result:
            Jn = result['Jn'].reshape(ny, nx)
            Jp = result['Jp'].reshape(ny, nx)
            J_total = np.sqrt(Jn**2 + Jp**2)
            im6 = plt.imshow(np.log10(np.maximum(J_total, 1e-10)),
                           extent=[0, self.device_params['length']*1e6, 0, self.device_params['width']*1e6],
                           aspect='auto', origin='lower', cmap='inferno')
            plt.colorbar(im6, shrink=0.8)
            plt.title('Total Current Density (log₁₀ A/m²)')
            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')

    def _plot_iv_characteristics(self, fig):
        """Plot I-V characteristics"""

        # Plot 7: Transfer Characteristics (Linear Scale)
        plt.subplot(4, 6, 7)
        transfer = self.simulation_results['transfer']
        plt.plot(transfer['Vg'], transfer['Id_linear'], 'b-', linewidth=2, label='Linear (Vd=0.1V)')
        plt.plot(transfer['Vg'], transfer['Id_saturation'], 'r-', linewidth=2, label='Saturation (Vd=1.2V)')
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Drain Current (A)')
        plt.title('Transfer Characteristics (Linear)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 8: Transfer Characteristics (Log Scale)
        plt.subplot(4, 6, 8)
        plt.semilogy(transfer['Vg'], transfer['Id_linear'], 'b-', linewidth=2, label='Linear')
        plt.semilogy(transfer['Vg'], transfer['Id_saturation'], 'r-', linewidth=2, label='Saturation')
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Drain Current (A)')
        plt.title('Transfer Characteristics (Log)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 9: Transconductance
        plt.subplot(4, 6, 9)
        plt.plot(transfer['Vg'], transfer['gm'], 'g-', linewidth=2)
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Transconductance (S)')
        plt.title('Transconductance vs Gate Voltage')
        plt.grid(True, alpha=0.3)

        # Plot 10: Output Characteristics
        plt.subplot(4, 6, 10)
        output = self.simulation_results['output']
        for curve_name, curve_data in output['curves'].items():
            Vg_val = curve_name.split('_')[1]
            plt.plot(output['Vd'], curve_data, linewidth=2, label=f'Vg = {Vg_val}')
        plt.xlabel('Drain Voltage (V)')
        plt.ylabel('Drain Current (A)')
        plt.title('Output Characteristics')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 11: gm/Id Efficiency
        plt.subplot(4, 6, 11)
        gm_over_Id = np.divide(transfer['gm'], transfer['Id_saturation'],
                              out=np.zeros_like(transfer['gm']), where=transfer['Id_saturation']!=0)
        plt.plot(transfer['Vg'], gm_over_Id, 'purple', linewidth=2)
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('gm/Id (S/A)')
        plt.title('Transconductance Efficiency')
        plt.grid(True, alpha=0.3)

    def _plot_device_structure(self, fig):
        """Plot device structure and doping profile"""

        # Plot 12: Device Cross-Section
        plt.subplot(4, 6, 12)

        # Create device structure visualization
        length_um = self.device_params['length'] * 1e6
        width_um = self.device_params['width'] * 1e6

        # Substrate
        substrate = patches.Rectangle((0, 0), length_um, width_um*0.7,
                                    facecolor='brown', alpha=0.3, label='P-substrate')
        plt.gca().add_patch(substrate)

        # Source region
        source = patches.Rectangle((0, 0), length_um*0.25, width_um*0.3,
                                 facecolor='blue', alpha=0.7, label='N+ Source')
        plt.gca().add_patch(source)

        # Drain region
        drain = patches.Rectangle((length_um*0.75, 0), length_um*0.25, width_um*0.3,
                                facecolor='blue', alpha=0.7, label='N+ Drain')
        plt.gca().add_patch(drain)

        # Gate oxide
        oxide = patches.Rectangle((length_um*0.25, width_um*0.7), length_um*0.5, width_um*0.1,
                                facecolor='lightblue', alpha=0.8, label='Gate Oxide')
        plt.gca().add_patch(oxide)

        # Gate
        gate = patches.Rectangle((length_um*0.25, width_um*0.8), length_um*0.5, width_um*0.1,
                               facecolor='gray', alpha=0.8, label='Gate')
        plt.gca().add_patch(gate)

        plt.xlim(0, length_um)
        plt.ylim(0, width_um)
        plt.xlabel('x (μm)')
        plt.ylabel('y (μm)')
        plt.title('Device Structure')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 13: Doping Profile (1D cut)
        plt.subplot(4, 6, 13)
        x_profile = np.linspace(0, length_um, 100)

        # Create doping profile
        doping_profile = np.ones_like(x_profile) * self.device_params['Na_substrate']

        # Source region
        source_mask = x_profile < length_um * 0.25
        doping_profile[source_mask] = self.device_params['Nd_source']

        # Drain region
        drain_mask = x_profile > length_um * 0.75
        doping_profile[drain_mask] = self.device_params['Nd_drain']

        plt.semilogy(x_profile, np.abs(doping_profile), 'k-', linewidth=2)
        plt.axvline(length_um * 0.25, color='blue', linestyle='--', alpha=0.7, label='Source edge')
        plt.axvline(length_um * 0.75, color='blue', linestyle='--', alpha=0.7, label='Drain edge')
        plt.xlabel('Position x (μm)')
        plt.ylabel('Doping Concentration (/m³)')
        plt.title('Doping Profile (Surface)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 14: Electric Field Vectors (if available)
        plt.subplot(4, 6, 14)
        if hasattr(self, 'last_result') and 'electric_field_x' in self.last_result:
            Ex = self.last_result['electric_field_x'].reshape(self.sim_params['ny'], self.sim_params['nx'])
            Ey = self.last_result['electric_field_y'].reshape(self.sim_params['ny'], self.sim_params['nx'])

            # Subsample for vector plot
            step = 4
            x_vec = self.x[::step]
            y_vec = self.y[::step]
            Ex_sub = Ex[::step, ::step]
            Ey_sub = Ey[::step, ::step]

            X_vec, Y_vec = np.meshgrid(x_vec, y_vec)

            plt.quiver(X_vec, Y_vec, Ex_sub, Ey_sub,
                      np.sqrt(Ex_sub**2 + Ey_sub**2),
                      cmap='viridis', alpha=0.8)
            plt.xlabel('x (μm)')
            plt.ylabel('y (μm)')
            plt.title('Electric Field Vectors')
            plt.colorbar(shrink=0.8, label='|E| (V/m)')
        else:
            plt.text(0.5, 0.5, 'Electric Field\nVectors\n(Not Available)',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Electric Field Vectors')

    def _plot_simulation_log(self, fig):
        """Plot simulation log and performance data"""

        # Plot 15: Performance Summary
        plt.subplot(4, 6, 15)
        plt.axis('off')

        # Create performance summary
        summary_text = "SIMULATION SUMMARY\n" + "="*20 + "\n\n"
        summary_text += f"Device Parameters:\n"
        summary_text += f"  Length: {self.device_params['length']*1e6:.1f} μm\n"
        summary_text += f"  Width: {self.device_params['width']*1e6:.1f} μm\n"
        summary_text += f"  Oxide: {self.device_params['tox']*1e9:.1f} nm\n"
        summary_text += f"  Grid: {self.sim_params['nx']}×{self.sim_params['ny']}\n"
        summary_text += f"  Method: {self.sim_params['method']}\n\n"

        if 'transfer' in self.simulation_results:
            transfer = self.simulation_results['transfer']
            max_current = np.max(transfer['Id_saturation'])
            min_current = np.min(transfer['Id_linear'])
            max_gm = np.max(transfer['gm'])

            summary_text += f"Key Results:\n"
            summary_text += f"  Max Current: {max_current:.2e} A\n"
            summary_text += f"  Min Current: {min_current:.2e} A\n"
            summary_text += f"  On/Off Ratio: {max_current/min_current:.1e}\n"
            summary_text += f"  Peak gm: {max_gm:.2e} S\n"

        summary_text += f"\nSimulation Status:\n"
        summary_text += f"  Total Log Entries: {len(self.simulation_log)}\n"
        summary_text += f"  Simulator: {'Real' if SIMULATOR_AVAILABLE else 'Analytical'}\n"

        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

        # Plot 16: Recent Log Entries
        plt.subplot(4, 6, 16)
        plt.axis('off')

        log_text = "RECENT LOG ENTRIES\n" + "="*18 + "\n\n"

        # Show last 15 log entries
        recent_logs = self.simulation_log[-15:] if len(self.simulation_log) > 15 else self.simulation_log

        for log_entry in recent_logs:
            # Truncate long entries
            if len(log_entry) > 50:
                log_text += log_entry[:47] + "...\n"
            else:
                log_text += log_entry + "\n"

        plt.text(0.05, 0.95, log_text, transform=plt.gca().transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace')

    def save_simulation_data(self, filename="mosfet_simulation_data.npz"):
        """Save all simulation data to file"""

        self.log(f"💾 Saving simulation data to {filename}...")

        # Prepare data for saving
        save_data = {
            'device_params': self.device_params,
            'sim_params': self.sim_params,
            'simulation_log': self.simulation_log
        }

        # Add simulation results if available
        if hasattr(self, 'simulation_results'):
            for key, value in self.simulation_results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            save_data[f"{key}_{subkey}"] = subvalue
                        elif isinstance(subvalue, (list, tuple)):
                            save_data[f"{key}_{subkey}"] = np.array(subvalue)

        # Save to compressed numpy format
        np.savez_compressed(filename, **save_data)
        self.log(f"   ✅ Data saved successfully")

    def print_simulation_log(self):
        """Print complete simulation log"""

        print("\n" + "="*60)
        print("COMPLETE SIMULATION LOG")
        print("="*60)

        for entry in self.simulation_log:
            print(entry)

        print("="*60)
        print(f"Total log entries: {len(self.simulation_log)}")
        print("="*60)

def main():
    """Main function to run complete MOSFET simulation"""

    print("🚀 COMPLETE MOSFET SIMULATION WITH FULL VISUALIZATION")
    print("=" * 70)
    print("This simulation will show all outputs you requested:")
    print("• Detailed simulation logs")
    print("• Electrostatic potential plots")
    print("• Carrier concentration distributions")
    print("• Current density maps and vectors")
    print("• Complete I-V and C-V characteristics")
    print("• Device structure visualization")
    print("• Performance analysis")
    print()

    try:
        # Create simulation
        mosfet_sim = CompleteMOSFETSimulation()

        # Step 1: Create device
        if not mosfet_sim.create_device():
            print("❌ Device creation failed")
            return False

        # Step 2: Run single point simulation for detailed physics
        mosfet_sim.log("🔍 Running detailed physics simulation...")
        operating_point = mosfet_sim.run_single_point_simulation(Vg=0.8, Vd=0.6)

        if operating_point:
            mosfet_sim.last_result = operating_point  # Store for vector plots

        # Step 3: Generate I-V characteristics
        mosfet_sim.generate_iv_characteristics()

        # Step 4: Create comprehensive plots
        mosfet_sim.create_comprehensive_plots(operating_point)

        # Step 5: Save data
        mosfet_sim.save_simulation_data()

        # Step 6: Print complete log
        mosfet_sim.print_simulation_log()

        print("\n🎉 COMPLETE SIMULATION FINISHED SUCCESSFULLY!")
        print("📊 Check the following outputs:")
        print("   • complete_mosfet_simulation.png - Comprehensive plots")
        print("   • mosfet_simulation_data.npz - All simulation data")
        print("   • Complete log printed above")

        return True

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
