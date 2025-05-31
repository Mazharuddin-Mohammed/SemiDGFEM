import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton, QTabWidget
from PySide6.QtCore import Qt
import numpy as np
from simulator import Simulator
from visualization.viz_2d import plot_2d_potential, plot_2d_quantity, plot_current_vectors, plot_mesh
import threading

class SimulatorGUI2D(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Semiconductor Simulator")
        self.sim = None
        self.results = None
        self.running = False
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Control Panel
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        self.tabs = QTabWidget()
        control_layout.addWidget(self.tabs)

        # Device Config Tab
        device_widget = QWidget()
        device_layout = QVBoxLayout(device_widget)
        self.lx_edit = QLineEdit("1.0")
        self.ly_edit = QLineEdit("0.5")
        self.nx_edit = QLineEdit("50")
        self.ny_edit = QLineEdit("25")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["DG"])
        self.order_combo = QComboBox()
        self.order_combo.addItems(["P2", "P3"])
        self.mesh_combo = QComboBox()
        self.mesh_combo.addItems(["Structured", "Unstructured"])
        self.amr_check = QCheckBox("Enable AMR")
        self.amr_check.setChecked(True)

        for label, widget in [
            ("Lx (μm):", self.lx_edit), ("Ly (μm):", self.ly_edit),
            ("Nx:", self.nx_edit), ("Ny:", self.ny_edit),
            ("Method:", self.method_combo), ("Order:", self.order_combo),
            ("Mesh Type:", self.mesh_combo), ("AMR:", self.amr_check)
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            device_layout.addLayout(row)
        self.tabs.addTab(device_widget, "Device Config")

        # Simulation Params Tab
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        self.nd_peak_edit = QLineEdit("1e17")
        self.na_peak_edit = QLineEdit("1e12")
        self.nd_x_edit = QLineEdit("100")
        self.na_x_edit = QLineEdit("900")
        self.sigma_edit = QLineEdit("50")
        self.et_edit = QLineEdit("0.3")
        self.v_left_edit = QLineEdit("0.0")
        self.v_right_edit = QLineEdit("1.0")
        self.v_bottom_edit = QLineEdit("0.0")
        self.v_top_edit = QLineEdit("0.0")

        for label, widget in [
            ("Nd Peak (cm⁻³):", self.nd_peak_edit), ("Na Peak (cm⁻³):", self.na_peak_edit),
            ("Nd x-position (nm):", self.nd_x_edit), ("Na x-position (nm):", self.na_x_edit),
            ("Sigma (nm):", self.sigma_edit), ("Trap Level (eV):", self.et_edit),
            ("V_left (V):", self.v_left_edit), ("V_right (V):", self.v_right_edit),
            ("V_bottom (V):", self.v_bottom_edit), ("V_top (V):", self.v_top_edit)
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            params_layout.addLayout(row)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.start_simulation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        params_layout.addLayout(button_layout)
        params_layout.addWidget(self.save_button)
        self.tabs.addTab(params_widget, "Simulation Params")

        main_layout.addWidget(control_widget)

        # Plot Tabs
        self.plot_tabs = QTabWidget()
        self.potential_window = QWidget()  # Placeholder for Vulkan window
        self.density_window = QWidget()
        self.current_window = QWidget()
        self.mesh_window = QWidget()
        self.plot_tabs.addTab(self.potential_window, "Potential")
        self.plot_tabs.addTab(self.density_window, "Density")
        self.plot_tabs.addTab(self.current_window, "Current")
        self.plot_tabs.addTab(self.mesh_window, "Mesh")
        main_layout.addWidget(self.plot_tabs)

    def start_simulation(self):
        if self.running:
            print("Simulation already running")
            return
        self.running = True
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        threading.Thread(target=self.run_simulation, daemon=True).start()

    def run_simulation(self):
        try:
            Lx = float(self.lx_edit.text()) * 1e-6
            Ly = float(self.ly_edit.text()) * 1e-6
            nx = int(self.nx_edit.text())
            ny = int(self.ny_edit.text())
            self.sim = Simulator(dimension="2D", extents=[Lx, Ly], num_points_x=nx, num_points_y=ny,
                                 method=self.method_combo.currentText(), 
                                 mesh_type=self.mesh_combo.currentText(), 
                                 order=self.order_combo.currentText())
            if self.mesh_combo.currentText() == "Unstructured":
                self.sim.generate_mesh(f"device_2d_{self.mesh_combo.currentText().lower()}.msh")

            grid = self.sim.get_grid_points()
            Nd_peak = float(self.nd_peak_edit.text()) * 1e6
            Na_peak = float(self.na_peak_edit.text()) * 1e6
            Nd_x = float(self.nd_x_edit.text()) * 1e-6
            Na_x = float(self.na_x_edit.text()) * 1e-6
            sigma = float(self.sigma_edit.text()) * 1e-6
            Et_val = float(self.et_edit.text())
            Nd = np.array([self.gaussian_doping(x, Nd_x, Nd_peak, sigma) for x in grid["x"]])
            Na = np.array([self.gaussian_doping(x, Na_x, Na_peak, sigma) for x in grid["x"]])
            Et = np.array([Et_val if x > Lx/2 else -Et_val for x in grid["x"]])
            self.sim.set_doping(Nd, Na)
            self.sim.set_trap_level(Et)

            self.results = self.sim.solve_drift_diffusion(
                [float(self.v_left_edit.text()), float(self.v_right_edit.text()),
                 float(self.v_bottom_edit.text()), float(self.v_top_edit.text())],
                max_steps=100, use_amr=self.amr_check.isChecked(),
                poisson_max_iter=50, poisson_tol=1e-6
            )
            self.results["x"] = grid["x"]
            self.results["y"] = grid["y"]

            self.update_plots()
        except Exception as e:
            print(f"Simulation failed: {str(e)}")
        finally:
            self.running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_button.setEnabled(True)

    def gaussian_doping(self, x, x0, Npeak, sigma):
        return Npeak * np.exp(-((x - x0)**2) / (2 * sigma**2))

    def update_plots(self):
        if not self.results:
            return
        plot_2d_potential(self.results, self.potential_window)
        plot_2d_quantity(self.results, "n", self.density_window)
        plot_current_vectors(self.results, self.current_window)
        plot_mesh(self.results, self.mesh_window)

    def stop_simulation(self):
        self.running = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def save_results(self):
        if not self.results:
            return
        np.savez("simulation_results.npz", **self.results)
        print("Results saved to simulation_results.npz")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulatorGUI2D()
    window.show()
    sys.exit(app.exec())