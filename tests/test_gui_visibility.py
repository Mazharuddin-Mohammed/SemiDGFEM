#!/usr/bin/env python3
"""
Test GUI visibility and log functionality
"""

import sys
import os

# Add gui directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))

try:
    from PySide6.QtWidgets import QApplication
    from modern_mosfet_gui import ModernMOSFETGUI
    
    def test_gui():
        app = QApplication(sys.argv)
        
        # Create GUI
        gui = ModernMOSFETGUI()
        
        # Test log functionality
        gui.log_message("🧪 Testing log functionality")
        gui.log_message("📋 Log should be visible in the right panel")
        gui.log_message("🔍 Checking if results panel is displayed")
        
        # Show window
        gui.show()
        gui.raise_()
        gui.activateWindow()
        
        print("✅ GUI created and shown")
        print("✅ Log messages sent")
        print("✅ Window should be visible with log panel on the right")
        print()
        print("Check if you can see:")
        print("  • Left panel: Controls and parameters")
        print("  • Right panel: Tabs with 'Live Log', 'Plots', 'Summary'")
        print("  • Log messages in the 'Live Log' tab")
        
        return app.exec()
        
    if __name__ == "__main__":
        sys.exit(test_gui())
        
except ImportError as e:
    print("❌ Failed to import GUI components:")
    print(f"   {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Failed to test GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
