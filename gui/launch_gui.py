#!/usr/bin/env python3
"""
SemiDGFEM GUI Launcher
Simple launcher script for the SemiDGFEM graphical interface
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if required dependencies are available"""
    
    missing_deps = []
    
    # Check for required packages
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")
    
    return missing_deps

def show_startup_info():
    """Show startup information"""
    
    print("🚀 SemiDGFEM GUI Launcher")
    print("=" * 40)
    print("Starting graphical interface...")
    print()
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print()
        print("Please install missing packages:")
        print("   pip install numpy matplotlib")
        return False
    
    print("✅ All dependencies available")
    
    # Check simulator availability
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import simulator
        print("✅ SemiDGFEM simulator module loaded")
    except ImportError:
        print("⚠️  SemiDGFEM simulator not available - running in demo mode")
    
    print()
    print("🎯 Features available:")
    print("   • Device parameter configuration")
    print("   • Real-time simulation")
    print("   • Interactive plotting")
    print("   • Performance profiling")
    print("   • Device builder tools")
    print()
    
    return True

def main():
    """Main launcher function"""
    
    # Show startup info
    if not show_startup_info():
        input("Press Enter to exit...")
        return
    
    try:
        # Import and launch GUI
        from main_window import SemiDGFEMGUI
        
        print("🖥️  Launching GUI...")
        
        # Create root window
        root = tk.Tk()
        
        # Set window icon (if available)
        try:
            # You can add an icon file here
            # root.iconbitmap("icon.ico")
            pass
        except:
            pass
        
        # Create and run GUI
        app = SemiDGFEMGUI(root)
        
        print("✅ GUI launched successfully!")
        print("   Close the GUI window to exit.")
        print()
        
        # Start GUI main loop
        root.mainloop()
        
        print("👋 GUI closed. Thank you for using SemiDGFEM!")
        
    except ImportError as e:
        error_msg = f"Failed to import GUI components: {e}"
        print(f"❌ {error_msg}")
        
        # Show error dialog if tkinter is available
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            messagebox.showerror("Import Error", error_msg)
        except:
            pass
        
        input("Press Enter to exit...")
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ {error_msg}")
        
        # Show error dialog
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", error_msg)
        except:
            pass
        
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
