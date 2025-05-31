#!/usr/bin/env python3
"""
Real-time Comprehensive MOSFET GUI Launcher
==========================================

Launch the comprehensive real-time MOSFET simulation with:
• Live GUI logging and progress tracking
• Steady-state and transient analysis
• Potential profiles, carrier concentrations, current densities
• Professional white theme visualization

Author: Dr. Mazharuddin Mohammed
Institution: Advanced Semiconductor Research Lab
"""

import sys
import os

# Add examples directory to path
examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
sys.path.append(examples_dir)

try:
    print("🚀 Launching Real-time Comprehensive MOSFET Simulator...")
    print("=" * 70)
    print("Features:")
    print("• Real-time GUI with live simulation logging")
    print("• Planar MOSFET with gate-oxide on top, adjacent to source/drain")
    print("• Comprehensive I-V characteristics (multiple gate voltages)")
    print("• Full transient analysis with gate step response")
    print("• Detailed device analysis:")
    print("  - Potential profile visualization")
    print("  - Carrier concentration maps (electrons & holes)")
    print("  - Current density distributions")
    print("  - Electric field visualization")
    print("• Professional white theme with multi-tab interface")
    print("• Real-time plot updates during simulation")
    print()
    print("Author: Dr. Mazharuddin Mohammed")
    print("Institution: Advanced Semiconductor Research Lab")
    print()
    
    # Import and run the GUI
    from realtime_comprehensive_mosfet_gui import main
    
    sys.exit(main())
    
except ImportError as e:
    print("❌ Failed to import GUI components:")
    print(f"   {e}")
    print()
    print("Please install required dependencies:")
    print("   pip install PySide6 matplotlib numpy")
    print()
    sys.exit(1)
except Exception as e:
    print(f"❌ Failed to start GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
