#!/usr/bin/env python3
"""
Real-time Heterostructure PN Diode GUI Launcher
==============================================

Launch the comprehensive real-time heterostructure PN diode simulation
with live GUI logging and results visualization.

Author: Dr. Mazharuddin Mohammed
Institution: Advanced Semiconductor Research Lab
"""

import sys
import os

# Add examples directory to path
examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
sys.path.append(examples_dir)

try:
    print("🚀 Launching Real-time Heterostructure PN Diode Simulator...")
    print("=" * 60)
    print("Features:")
    print("• Real-time GUI with live simulation logging")
    print("• GaAs/AlGaAs heterostructure device modeling")
    print("• Live I-V and C-V characteristics plotting")
    print("• Professional white theme visualization")
    print("• Interactive parameter controls")
    print("• Multi-tab results display")
    print()
    print("Author: Dr. Mazharuddin Mohammed")
    print("Institution: Advanced Semiconductor Research Lab")
    print()
    
    # Import and run the GUI
    from realtime_heterostructure_pn_gui import main
    
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
