#!/usr/bin/env python3
"""
Launcher script for Real-Time MOSFET Simulator GUI
"""

import sys
import os

# Add gui directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))

# Import and run the GUI
from mosfet_simulator_gui import main

if __name__ == "__main__":
    main()
