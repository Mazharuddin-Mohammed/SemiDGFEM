#!/usr/bin/env python3
"""
Launcher for Modern PySide6 MOSFET Simulator GUI

Author: Dr. Mazharuddin Mohammed
"""

import sys
import os

# Add gui directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))

try:
    from modern_mosfet_gui import main
    
    if __name__ == "__main__":
        print("🚀 Starting Modern MOSFET Simulator GUI...")
        print("Features:")
        print("• Modern PySide6 interface with dark theme")
        print("• Advanced physics models in C++ backend")
        print("• Real-time simulation logging")
        print("• Interactive parameter controls")
        print("• Comprehensive visualization")
        print("• Configuration save/load")
        print()
        
        sys.exit(main())
        
except ImportError as e:
    print("❌ Failed to import PySide6 GUI:")
    print(f"   {e}")
    print()
    print("Please install PySide6:")
    print("   pip install PySide6 matplotlib numpy")
    print()
    sys.exit(1)
except Exception as e:
    print(f"❌ Failed to start GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
