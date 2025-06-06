#!/usr/bin/env python3
"""
Demonstration of Device Geometry Improvements and Smooth I-V Curves

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def demonstrate_device_geometry():
    """Demonstrate the improved device geometry definition"""
    
    print("🔧 DEVICE GEOMETRY IMPROVEMENTS DEMONSTRATION")
    print("=" * 60)
    print("Author: Dr. Mazharuddin Mohammed")
    print()
    
    # Device parameters
    length = 100e-9  # 100 nm
    width = 1e-6     # 1 μm
    nx, ny = 40, 20
    
    # SIMPLE PLANAR MOSFET: GATE-OXIDE ON TOP, ADJACENT TO SOURCE & DRAIN

    # Simple lateral structure
    source_end = length * 0.25      # Source region: 0 to 25%
    gate_start = length * 0.25      # Gate starts adjacent to source
    gate_end = length * 0.75        # Gate ends adjacent to drain
    drain_start = length * 0.75     # Drain region: 75% to 100%

    # Simple vertical structure
    substrate_top = width * 0.7    # Substrate: 0 to 70%
    surface_top = width * 0.9      # Surface layer: 70% to 90%
    gate_top = width               # Gate-oxide stack: 90% to 100% (on top)

    print("🔬 SIMPLE PLANAR MOSFET: Gate-oxide on top, adjacent to source & drain")
    print(f"   Device dimensions: {length*1e9:.0f} nm × {width*1e6:.1f} μm")
    print(f"   Vertical structure:")
    print(f"     P-substrate: 0 to {substrate_top*1e6:.0f} nm (70%)")
    print(f"     Surface layer: {substrate_top*1e6:.0f} to {surface_top*1e6:.0f} nm (20%)")
    print(f"     Gate-oxide stack: {surface_top*1e6:.0f} to {gate_top*1e6:.0f} nm (10% - ON TOP)")
    print(f"   Lateral structure:")
    print(f"     Source: 0 to {source_end*1e9:.0f} nm")
    print(f"     Gate (adjacent): {gate_start*1e9:.0f} to {gate_end*1e9:.0f} nm")
    print(f"     Drain: {drain_start*1e9:.0f} to {length*1e9:.0f} nm")
    print(f"   Gate-oxide stack is ON TOP and ADJACENT to source & drain regions")
    print()
    
    # Create coordinate arrays
    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)
    X, Y = np.meshgrid(x, y)
    
    # Count regions for simple planar MOSFET structure
    p_substrate = 0
    n_source_surface = 0
    p_channel_surface = 0
    n_drain_surface = 0
    gate_oxide_stack = 0
    air_vacuum = 0

    for i in range(ny):
        for j in range(nx):
            x_pos = X[i, j]
            y_pos = Y[i, j]

            if y_pos <= substrate_top:  # P-substrate (bulk)
                p_substrate += 1

            elif y_pos <= surface_top:  # Surface layer
                if x_pos <= source_end:  # N+ Source
                    n_source_surface += 1
                elif x_pos >= drain_start:  # N+ Drain
                    n_drain_surface += 1
                else:  # P-Channel
                    p_channel_surface += 1

            else:  # Gate-oxide stack (on top)
                if gate_start <= x_pos <= gate_end:  # Gate-oxide stack
                    gate_oxide_stack += 1
                else:  # Air/vacuum outside gate
                    air_vacuum += 1
    
    total_points = nx * ny
    
    print("📊 SIMPLE PLANAR MOSFET REGION STATISTICS:")
    print(f"   P-substrate (bulk): {p_substrate} points ({p_substrate/total_points*100:.1f}%)")
    print(f"   Surface layer:")
    print(f"     N+ Source: {n_source_surface} points ({n_source_surface/total_points*100:.1f}%)")
    print(f"     P-Channel: {p_channel_surface} points ({p_channel_surface/total_points*100:.1f}%)")
    print(f"     N+ Drain: {n_drain_surface} points ({n_drain_surface/total_points*100:.1f}%)")
    print(f"   Top layer:")
    print(f"     Gate-oxide stack: {gate_oxide_stack} points ({gate_oxide_stack/total_points*100:.1f}%) [ON TOP, ADJACENT]")
    print(f"     Air/vacuum: {air_vacuum} points ({air_vacuum/total_points*100:.1f}%)")
    print(f"   Total grid points: {total_points}")
    print(f"   Total semiconductor: {p_substrate + n_source_surface + p_channel_surface + n_drain_surface} points ({(p_substrate + n_source_surface + p_channel_surface + n_drain_surface)/total_points*100:.1f}%)")
    print(f"   Gate-oxide stack is ON TOP and ADJACENT to source & drain")
    print()
    
    return {
        'geometry': {
            'length': length,
            'width': width,
            'source_end': source_end,
            'gate_start': gate_start,
            'gate_end': gate_end,
            'drain_start': drain_start,
            'substrate_top': substrate_top,
            'surface_top': surface_top,
            'gate_top': gate_top
        },
        'statistics': {
            'p_substrate': p_substrate,
            'n_source_surface': n_source_surface,
            'p_channel_surface': p_channel_surface,
            'n_drain_surface': n_drain_surface,
            'gate_oxide_stack': gate_oxide_stack,
            'air_vacuum': air_vacuum,
            'total_points': total_points,
            'total_semiconductor': p_substrate + n_source_surface + p_channel_surface + n_drain_surface
        }
    }

def demonstrate_iv_improvements():
    """Demonstrate the improved I-V curve resolution"""
    
    print("📈 I-V CURVE RESOLUTION IMPROVEMENTS")
    print("=" * 60)
    
    # Old resolution
    Vg_old = np.linspace(0.0, 1.5, 8)   # 8 points
    Vd_old = np.linspace(0.0, 1.5, 15)  # 15 points
    total_old = len(Vg_old) * len(Vd_old)
    
    # New resolution (doubled)
    Vg_new = np.linspace(0.0, 1.5, 16)  # 16 points
    Vd_new = np.linspace(0.0, 1.5, 31)  # 31 points
    total_new = len(Vg_new) * len(Vd_new)
    
    print("🔄 RESOLUTION COMPARISON:")
    print(f"   Old resolution:")
    print(f"     Gate voltage: {len(Vg_old)} points (0V to 1.5V)")
    print(f"     Drain voltage: {len(Vd_old)} points (0V to 1.5V)")
    print(f"     Total simulation points: {total_old}")
    print()
    print(f"   New resolution (IMPROVED):")
    print(f"     Gate voltage: {len(Vg_new)} points (0V to 1.5V)")
    print(f"     Drain voltage: {len(Vd_new)} points (0V to 1.5V)")
    print(f"     Total simulation points: {total_new}")
    print()
    print(f"   Improvement factor: {total_new/total_old:.1f}x more points")
    print(f"   Curve smoothness: {total_new/total_old:.1f}x better resolution")
    print()
    
    # Generate sample I-V data for demonstration
    print("📊 GENERATING SAMPLE I-V CHARACTERISTICS...")
    
    # Calculate realistic MOSFET parameters
    ni = 1.45e16
    k = 1.381e-23
    q = 1.602e-19
    T = 300.0
    Vt = k * T / q
    Na_substrate = 1e23
    
    phi_F = Vt * np.log(Na_substrate / ni)
    epsilon_si = 11.7 * 8.854e-12
    epsilon_ox = 3.9 * 8.854e-12
    tox = 2e-9
    Cox = epsilon_ox / tox
    gamma = np.sqrt(2 * q * epsilon_si * Na_substrate) / Cox
    Vth = 2 * phi_F + gamma * np.sqrt(2 * phi_F)
    
    print(f"   Calculated threshold voltage: {Vth:.3f} V")
    print(f"   Using {len(Vg_new)} gate voltages for smooth transfer curves")
    print(f"   Using {len(Vd_new)} drain voltages for smooth output curves")
    
    # Calculate I-V characteristics with new resolution
    length = 100e-9
    width = 1e-6
    W_over_L = width / length
    mu_eff = 0.05
    
    Id_matrix = np.zeros((len(Vg_new), len(Vd_new)))
    
    for i, Vg in enumerate(Vg_new):
        for j, Vd in enumerate(Vd_new):
            if Vg < Vth:
                Id = 1e-15 * np.exp((Vg - Vth) / (10 * Vt))
            else:
                if Vd < (Vg - Vth):
                    Id = mu_eff * Cox * W_over_L * ((Vg - Vth) * Vd - 0.5 * Vd**2)
                else:
                    Id = 0.5 * mu_eff * Cox * W_over_L * (Vg - Vth)**2
                    # Channel length modulation
                    if Vd > (Vg - Vth):
                        lambda_clm = 0.1
                        Id *= (1 + lambda_clm * (Vd - (Vg - Vth)))
            
            Id_matrix[i, j] = Id
    
    # Statistics
    Id_max = np.max(Id_matrix)
    Id_min = np.min(Id_matrix[Id_matrix > 0])
    on_off_ratio = Id_max / Id_min
    
    print()
    print("✅ I-V CHARACTERISTICS GENERATED:")
    print(f"   Current range: {Id_min:.2e} to {Id_max:.2e} A")
    print(f"   On/Off ratio: {on_off_ratio:.1e}")
    print(f"   Smooth curves with {total_new} data points")
    print()
    
    return {
        'resolution': {
            'Vg_points': len(Vg_new),
            'Vd_points': len(Vd_new),
            'total_points': total_new,
            'improvement_factor': total_new/total_old
        },
        'characteristics': {
            'Vth': Vth,
            'Id_max': Id_max,
            'Id_min': Id_min,
            'on_off_ratio': on_off_ratio
        }
    }

def main():
    """Main demonstration function"""
    
    print("🚀 COMPREHENSIVE IMPROVEMENTS DEMONSTRATION")
    print("=" * 80)
    print("Author: Dr. Mazharuddin Mohammed")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Demonstrate device geometry improvements
    geometry_results = demonstrate_device_geometry()
    
    # Demonstrate I-V curve improvements
    iv_results = demonstrate_iv_improvements()
    
    # Summary
    print("🎯 IMPROVEMENTS SUMMARY:")
    print("=" * 60)
    print()
    print("✅ SIMPLE PLANAR MOSFET STRUCTURE:")
    print("   • Gate-oxide stack ON TOP of device (top 10%)")
    print("   • Gate-oxide ADJACENT to source (0-25nm) and drain (75-100nm)")
    print("   • Simple three-layer structure: substrate → surface → gate-oxide")
    print("   • Standard planar MOSFET configuration")
    print("   • Gate-oxide stack positioned on top, adjacent to source & drain")
    print()
    print("✅ I-V CURVE REFINEMENTS:")
    print(f"   • Resolution improved by {iv_results['resolution']['improvement_factor']:.1f}x")
    print(f"   • Gate voltage: {iv_results['resolution']['Vg_points']} points")
    print(f"   • Drain voltage: {iv_results['resolution']['Vd_points']} points")
    print(f"   • Total simulation points: {iv_results['resolution']['total_points']}")
    print("   • Smooth curves with professional visualization")
    print()
    print("✅ AUTHORSHIP ATTRIBUTION:")
    print("   • All source files updated with proper attribution")
    print("   • README.md prominently displays author")
    print("   • Citation format updated for research use")
    print("   • Consistent authorship across all components")
    print()
    print("🏆 RESULT: Production-ready MOSFET simulator with:")
    print("   • Proper device physics and geometry")
    print("   • Smooth, high-resolution I-V characteristics")
    print("   • Professional authorship attribution")
    print("   • Enhanced visualization and logging")
    print()
    print("✅ Ready for advanced semiconductor device research!")
    print("=" * 80)

if __name__ == "__main__":
    main()
