#!/usr/bin/env python3
"""
Advanced MOSFET Device Simulation
Demonstrates complex device physics including:
- Gate oxide effects
- Channel formation
- Threshold voltage extraction
- I-V characteristics
- Small-signal analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for simulator import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import simulator
    SIMULATOR_AVAILABLE = True
except ImportError:
    print("⚠️  Simulator module not available. Running in demo mode.")
    SIMULATOR_AVAILABLE = False

class MOSFETSimulator:
    """Advanced MOSFET device simulator with comprehensive analysis"""
    
    def __init__(self, device_params=None):
        """Initialize MOSFET simulator with device parameters"""
        
        # Default device parameters
        self.params = {
            'length': 1e-6,        # Channel length (m)
            'width': 10e-6,        # Channel width (m)
            'tox': 2e-9,           # Oxide thickness (m)
            'Na': 1e17,            # Substrate doping (1/m³)
            'Nd_source': 1e20,     # Source doping (1/m³)
            'Nd_drain': 1e20,      # Drain doping (1/m³)
            'junction_depth': 0.1e-6,  # Junction depth (m)
            'gate_work_function': 4.1,  # Gate work function (eV)
            'substrate_work_function': 4.05,  # Substrate work function (eV)
        }
        
        if device_params:
            self.params.update(device_params)
        
        # Simulation parameters
        self.nx = 100  # Grid points in x (channel length)
        self.ny = 50   # Grid points in y (channel width)
        
        # Results storage
        self.results = {}
        
    def create_device_structure(self):
        """Create MOSFET device structure with proper doping profiles"""
        
        print("🔧 Creating MOSFET device structure...")
        
        if not SIMULATOR_AVAILABLE:
            print("   Using analytical model (simulator not available)")
            return self._create_analytical_structure()
        
        # Create simulator instance
        total_length = self.params['length'] + 2 * self.params['junction_depth']
        total_width = self.params['width']
        
        self.sim = simulator.Simulator(
            extents=[total_length, total_width],
            num_points_x=self.nx,
            num_points_y=self.ny,
            method="DG",
            mesh_type="Structured"
        )
        
        # Create doping profile
        total_points = self.nx * self.ny
        Nd = np.zeros(total_points)
        Na = np.zeros(total_points)
        
        # Device geometry
        source_end = int(self.nx * self.params['junction_depth'] / total_length)
        channel_start = source_end
        channel_end = int(self.nx * (self.params['junction_depth'] + self.params['length']) / total_length)
        drain_start = channel_end
        
        for i in range(total_points):
            x_idx = i % self.nx
            y_idx = i // self.nx
            
            # Vertical position for junction depth consideration
            y_pos = y_idx / self.ny
            
            if x_idx < source_end:
                # Source region
                if y_pos < 0.2:  # Junction depth
                    Nd[i] = self.params['Nd_source']
                else:
                    Na[i] = self.params['Na']
            elif x_idx >= drain_start:
                # Drain region
                if y_pos < 0.2:  # Junction depth
                    Nd[i] = self.params['Nd_drain']
                else:
                    Na[i] = self.params['Na']
            else:
                # Channel region
                Na[i] = self.params['Na']
        
        # Set doping in simulator
        self.sim.set_doping(Nd, Na)
        
        print(f"   ✓ Device created: {self.params['length']*1e6:.1f}μm × {self.params['width']*1e6:.1f}μm")
        print(f"   ✓ Grid: {self.nx} × {self.ny}")
        print(f"   ✓ Source/Drain doping: {self.params['Nd_source']:.1e} /m³")
        print(f"   ✓ Substrate doping: {self.params['Na']:.1e} /m³")
        
        return True
    
    def _create_analytical_structure(self):
        """Create analytical device structure for demo mode"""
        
        # Create mock simulator-like interface
        class MockSimulator:
            def __init__(self, nx, ny):
                self.nx = nx
                self.ny = ny
                self.num_points_x = nx
                self.num_points_y = ny
                
            def solve_poisson(self, bc):
                # Return analytical solution
                n = self.nx * self.ny
                x = np.linspace(0, 1, self.nx)
                y = np.linspace(0, 1, self.ny)
                X, Y = np.meshgrid(x, y)
                
                # Simple parabolic potential
                V = bc[0] + (bc[1] - bc[0]) * X.flatten() + 0.1 * np.sin(np.pi * X.flatten())
                return V
                
            def solve_drift_diffusion(self, bc, **kwargs):
                n = self.nx * self.ny
                V = self.solve_poisson(bc)
                
                # Mock carrier densities
                n_carriers = 1e16 * np.ones(n) * np.exp(-V / 0.026)
                p_carriers = 1e16 * np.ones(n) * np.exp(V / 0.026)
                
                return {
                    'potential': V,
                    'n': n_carriers,
                    'p': p_carriers,
                    'Jn': np.gradient(n_carriers) * 1e-3,
                    'Jp': np.gradient(p_carriers) * 1e-3
                }
        
        self.sim = MockSimulator(self.nx, self.ny)
        return True
    
    def calculate_threshold_voltage(self):
        """Calculate threshold voltage using various methods"""
        
        print("📊 Calculating threshold voltage...")
        
        # Gate voltage sweep
        Vg_range = np.linspace(-1.0, 2.0, 31)
        Vd = 0.1  # Small drain voltage for linear region
        
        Id_values = []
        
        for Vg in Vg_range:
            # Set boundary conditions: [source, drain, substrate, gate]
            bc = [0.0, Vd, 0.0, Vg]
            
            try:
                if hasattr(self.sim, 'solve_drift_diffusion'):
                    result = self.sim.solve_drift_diffusion(bc, max_steps=20)
                    
                    # Calculate drain current (simplified)
                    Jn = result.get('Jn', np.zeros(self.nx * self.ny))
                    Jp = result.get('Jp', np.zeros(self.nx * self.ny))
                    
                    # Current at drain contact (last column)
                    drain_indices = np.arange(self.nx-1, self.nx * self.ny, self.nx)
                    Id = np.mean(Jn[drain_indices] + Jp[drain_indices]) * self.params['width']
                    
                else:
                    # Analytical approximation
                    if Vg < 0.5:
                        Id = 1e-12 * np.exp(Vg / 0.1)  # Subthreshold
                    else:
                        Id = 1e-6 * (Vg - 0.5)**2 * Vd  # Above threshold
                
                Id_values.append(abs(Id))
                
            except Exception as e:
                print(f"   Warning: Simulation failed at Vg={Vg:.2f}V: {e}")
                Id_values.append(1e-15)
        
        Id_values = np.array(Id_values)
        
        # Extract threshold voltage using different methods
        methods = {}
        
        # Method 1: Linear extrapolation
        try:
            # Find linear region (above threshold)
            log_Id = np.log10(np.maximum(Id_values, 1e-15))
            linear_region = np.where(Vg_range > 0.5)[0]
            
            if len(linear_region) > 5:
                p = np.polyfit(Vg_range[linear_region], log_Id[linear_region], 1)
                # Extrapolate to find Vth where Id = 1e-12 A
                Vth_linear = (np.log10(1e-12) - p[1]) / p[0]
                methods['linear_extrapolation'] = Vth_linear
        except:
            methods['linear_extrapolation'] = 0.5
        
        # Method 2: Constant current method
        try:
            target_current = 1e-9  # 1 nA
            idx = np.argmin(np.abs(Id_values - target_current))
            methods['constant_current'] = Vg_range[idx]
        except:
            methods['constant_current'] = 0.5
        
        # Method 3: Transconductance maximum
        try:
            gm = np.gradient(Id_values, Vg_range[1] - Vg_range[0])
            idx_max_gm = np.argmax(gm)
            methods['transconductance_max'] = Vg_range[idx_max_gm]
        except:
            methods['transconductance_max'] = 0.5
        
        self.results['threshold_voltage'] = methods
        self.results['Id_vs_Vg'] = {'Vg': Vg_range, 'Id': Id_values}
        
        print(f"   ✓ Threshold voltage (linear extrapolation): {methods.get('linear_extrapolation', 'N/A'):.3f} V")
        print(f"   ✓ Threshold voltage (constant current): {methods.get('constant_current', 'N/A'):.3f} V")
        print(f"   ✓ Threshold voltage (transconductance max): {methods.get('transconductance_max', 'N/A'):.3f} V")
        
        return methods
    
    def generate_iv_characteristics(self):
        """Generate comprehensive I-V characteristics"""
        
        print("📈 Generating I-V characteristics...")
        
        # Output characteristics (Id vs Vd for different Vg)
        Vd_range = np.linspace(0, 2.0, 21)
        Vg_values = [0.5, 0.8, 1.0, 1.2, 1.5]
        
        output_curves = {}
        
        for Vg in Vg_values:
            Id_output = []
            
            for Vd in Vd_range:
                bc = [0.0, Vd, 0.0, Vg]
                
                try:
                    if hasattr(self.sim, 'solve_drift_diffusion'):
                        result = self.sim.solve_drift_diffusion(bc, max_steps=15)
                        Jn = result.get('Jn', np.zeros(self.nx * self.ny))
                        Jp = result.get('Jp', np.zeros(self.nx * self.ny))
                        
                        drain_indices = np.arange(self.nx-1, self.nx * self.ny, self.nx)
                        Id = np.mean(Jn[drain_indices] + Jp[drain_indices]) * self.params['width']
                    else:
                        # Analytical MOSFET model
                        Vth = 0.5
                        if Vg < Vth:
                            Id = 1e-12 * np.exp((Vg - Vth) / 0.1)
                        else:
                            if Vd < (Vg - Vth):  # Linear region
                                Id = 1e-6 * (Vg - Vth) * Vd
                            else:  # Saturation region
                                Id = 0.5e-6 * (Vg - Vth)**2
                    
                    Id_output.append(abs(Id))
                    
                except Exception as e:
                    print(f"   Warning: Failed at Vg={Vg}V, Vd={Vd}V: {e}")
                    Id_output.append(1e-15)
            
            output_curves[f'Vg_{Vg}V'] = {'Vd': Vd_range, 'Id': np.array(Id_output)}
        
        self.results['output_characteristics'] = output_curves
        
        print(f"   ✓ Generated output curves for {len(Vg_values)} gate voltages")
        print(f"   ✓ Drain voltage range: 0 to {max(Vd_range):.1f} V")
        
        return output_curves
    
    def analyze_small_signal_parameters(self):
        """Analyze small-signal parameters (gm, gds, etc.)"""
        
        print("🔬 Analyzing small-signal parameters...")
        
        if 'Id_vs_Vg' not in self.results:
            self.calculate_threshold_voltage()
        
        Vg_range = self.results['Id_vs_Vg']['Vg']
        Id_values = self.results['Id_vs_Vg']['Id']
        
        # Calculate transconductance (gm = dId/dVg)
        gm = np.gradient(Id_values, Vg_range[1] - Vg_range[0])
        
        # Find peak transconductance
        max_gm_idx = np.argmax(gm)
        max_gm = gm[max_gm_idx]
        Vg_max_gm = Vg_range[max_gm_idx]
        
        # Calculate transconductance efficiency (gm/Id)
        gm_over_Id = np.divide(gm, Id_values, out=np.zeros_like(gm), where=Id_values!=0)
        
        small_signal_params = {
            'transconductance': gm,
            'max_transconductance': max_gm,
            'Vg_at_max_gm': Vg_max_gm,
            'transconductance_efficiency': gm_over_Id,
            'Vg_range': Vg_range
        }
        
        self.results['small_signal'] = small_signal_params
        
        print(f"   ✓ Peak transconductance: {max_gm:.2e} S")
        print(f"   ✓ At gate voltage: {Vg_max_gm:.3f} V")
        print(f"   ✓ Max gm/Id: {np.max(gm_over_Id):.1f} S/A")
        
        return small_signal_params
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive analysis plots"""
        
        print("📊 Creating comprehensive analysis plots...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Transfer characteristics (Id vs Vg)
        plt.subplot(2, 3, 1)
        if 'Id_vs_Vg' in self.results:
            data = self.results['Id_vs_Vg']
            plt.semilogy(data['Vg'], data['Id'], 'b-', linewidth=2, label='Id')
            plt.xlabel('Gate Voltage (V)')
            plt.ylabel('Drain Current (A)')
            plt.title('Transfer Characteristics')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Mark threshold voltages
            if 'threshold_voltage' in self.results:
                for method, Vth in self.results['threshold_voltage'].items():
                    if isinstance(Vth, (int, float)):
                        plt.axvline(Vth, linestyle='--', alpha=0.7, 
                                   label=f'Vth ({method[:8]}): {Vth:.3f}V')
                plt.legend()
        
        # Plot 2: Output characteristics (Id vs Vd)
        plt.subplot(2, 3, 2)
        if 'output_characteristics' in self.results:
            for label, data in self.results['output_characteristics'].items():
                Vg_val = label.split('_')[1]
                plt.plot(data['Vd'], data['Id'], linewidth=2, label=f'Vg = {Vg_val}')
            plt.xlabel('Drain Voltage (V)')
            plt.ylabel('Drain Current (A)')
            plt.title('Output Characteristics')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 3: Transconductance
        plt.subplot(2, 3, 3)
        if 'small_signal' in self.results:
            data = self.results['small_signal']
            plt.plot(data['Vg_range'], data['transconductance'], 'r-', linewidth=2)
            plt.xlabel('Gate Voltage (V)')
            plt.ylabel('Transconductance (S)')
            plt.title('Transconductance vs Gate Voltage')
            plt.grid(True, alpha=0.3)
            
            # Mark maximum
            max_idx = np.argmax(data['transconductance'])
            plt.plot(data['Vg_range'][max_idx], data['transconductance'][max_idx], 
                    'ro', markersize=8, label=f"Max: {data['max_transconductance']:.2e} S")
            plt.legend()
        
        # Plot 4: Transconductance efficiency
        plt.subplot(2, 3, 4)
        if 'small_signal' in self.results:
            data = self.results['small_signal']
            plt.plot(data['Vg_range'], data['transconductance_efficiency'], 'g-', linewidth=2)
            plt.xlabel('Gate Voltage (V)')
            plt.ylabel('gm/Id (S/A)')
            plt.title('Transconductance Efficiency')
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Device structure visualization
        plt.subplot(2, 3, 5)
        # Create a simple device cross-section visualization
        x = np.linspace(0, self.params['length']*1e6, 100)
        y_substrate = np.ones_like(x) * 0.5
        y_oxide = np.ones_like(x) * 0.7
        y_gate = np.ones_like(x) * 0.8
        
        plt.fill_between(x, 0, y_substrate, alpha=0.3, color='brown', label='Substrate (p-type)')
        plt.fill_between(x, y_substrate, y_oxide, alpha=0.3, color='blue', label='Oxide')
        plt.fill_between(x, y_oxide, y_gate, alpha=0.3, color='gray', label='Gate')
        
        # Mark source and drain
        source_x = 0.1 * self.params['length'] * 1e6
        drain_x = 0.9 * self.params['length'] * 1e6
        plt.axvline(source_x, color='red', linestyle='--', label='Source')
        plt.axvline(drain_x, color='red', linestyle='--', label='Drain')
        
        plt.xlabel('Position (μm)')
        plt.ylabel('Height (a.u.)')
        plt.title('Device Structure')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Performance summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Create performance summary text
        summary_text = "MOSFET Performance Summary\n" + "="*30 + "\n\n"
        
        if 'threshold_voltage' in self.results:
            Vth = self.results['threshold_voltage'].get('linear_extrapolation', 'N/A')
            summary_text += f"Threshold Voltage: {Vth:.3f} V\n"
        
        if 'small_signal' in self.results:
            max_gm = self.results['small_signal']['max_transconductance']
            summary_text += f"Peak gm: {max_gm:.2e} S\n"
        
        summary_text += f"\nDevice Parameters:\n"
        summary_text += f"Length: {self.params['length']*1e6:.1f} μm\n"
        summary_text += f"Width: {self.params['width']*1e6:.1f} μm\n"
        summary_text += f"Oxide thickness: {self.params['tox']*1e9:.1f} nm\n"
        summary_text += f"Substrate doping: {self.params['Na']:.1e} /m³\n"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('mosfet_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✓ Comprehensive analysis plot saved as 'mosfet_comprehensive_analysis.png'")

def main():
    """Main function to run advanced MOSFET simulation"""
    
    print("🚀 Advanced MOSFET Device Simulation")
    print("=" * 50)
    
    # Create MOSFET simulator with custom parameters
    device_params = {
        'length': 0.5e-6,      # 500 nm channel
        'width': 10e-6,        # 10 μm width
        'tox': 2e-9,           # 2 nm oxide
        'Na': 5e17,            # Higher substrate doping
    }
    
    mosfet = MOSFETSimulator(device_params)
    
    # Run comprehensive simulation
    try:
        # Step 1: Create device structure
        mosfet.create_device_structure()
        
        # Step 2: Calculate threshold voltage
        mosfet.calculate_threshold_voltage()
        
        # Step 3: Generate I-V characteristics
        mosfet.generate_iv_characteristics()
        
        # Step 4: Analyze small-signal parameters
        mosfet.analyze_small_signal_parameters()
        
        # Step 5: Create comprehensive plots
        mosfet.plot_comprehensive_analysis()
        
        print("\n🎉 Advanced MOSFET simulation completed successfully!")
        print("📊 Check 'mosfet_comprehensive_analysis.png' for detailed results")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
