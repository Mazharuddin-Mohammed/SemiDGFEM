import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from python.simulator import Simulator
from visualization.viz_2d import plot_2d_potential, plot_2d_quantity, plot_current_vectors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os

class SimulatorGUI2D:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Semiconductor Simulator")
        self.sim = None
        self.results = None
        self.running = False
        self.create_widgets()
        self.setup_plots()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Device Configuration Tab
        device_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(device_frame, text="Device Config")

        ttk.Label(device_frame, text="Lx (μm):").grid(row=1, column=0, sticky="e")
        self.lx_entry = ttk.Entry(device_frame)
        self.lx_entry.grid(row=1, column=1)
        self.lx_entry.insert(0, "1.0")

        ttk.Label(device_frame, text="Ly (μm):").grid(row=2, column=0, sticky="e")
        self.ly_entry = ttk.Entry(device_frame)
        self.ly_entry.grid(row=2, column=1)
        self.ly_entry.insert(0, "0.5")

        ttk.Label(device_frame, text="Nx:").grid(row=3, column=0, sticky="e")
        self.nx_entry = ttk.Entry(device_frame)
        self.nx_entry.grid(row=3, column=1)
        self.nx_entry.insert(0, "50")

        ttk.Label(device_frame, text="Ny:").grid(row=4, column=0, sticky="e")
        self.ny_entry = ttk.Entry(device_frame)
        self.ny_entry.grid(row=4, column=1)
        self.ny_entry.insert(0, "25")

        ttk.Label(device_frame, text="Method:").grid(row=5, column=0, sticky="e")
        self.method_var = tk.StringVar(value="DG")
        ttk.OptionMenu(device_frame, self.method_var, "DG", "DG").grid(row=5, column=1)

        # [MODIFICATION]: Added order selection
        ttk.Label(device_frame, text="Order:").grid(row=6, column=0, sticky="e")
        self.order_var = tk.StringVar(value="P3")
        ttk.OptionMenu(device_frame, self.order_var, "P3", "P2", "P3").grid(row=6, column=1)

        ttk.Label(device_frame, text="Mesh Type:").grid(row=7, column=0, sticky="e")
        self.mesh_var = tk.StringVar(value="Structured")
        ttk.OptionMenu(device_frame, self.mesh_var, "Structured", "Structured", "Unstructured").grid(row=7, column=1)

        ttk.Label(device_frame, text="AMR:").grid(row=8, column=0, sticky="e")
        self.amr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(device_frame, variable=self.amr_var).grid(row=8, column=1)

        # Simulation Parameters Tab
        params_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(params_frame, text="Simulation Params")

        ttk.Label(params_frame, text="Nd Peak (cm⁻³):").grid(row=0, column=0, sticky="e")
        self.nd_peak = ttk.Entry(params_frame)
        self.nd_peak.grid(row=0, column=1)
        self.nd_peak.insert(0, "1e17")

        ttk.Label(params_frame, text="Na Peak (cm⁻³):").grid(row=1, column=0, sticky="e")
        self.na_peak = ttk.Entry(params_frame)
        self.na_peak.grid(row=1, column=1)
        self.na_peak.insert(0, "1e12")

        ttk.Label(params_frame, text="Nd x-position (nm):").grid(row=2, column=0, sticky="e")
        self.nd_x = ttk.Entry(params_frame)
        self.nd_x.grid(row=2, column=1)
        self.nd_x.insert(0, "100")

        ttk.Label(params_frame, text="Na x-position (nm):").grid(row=3, column=0, e")
        self.na_x = ttk.Entry(params_frame)
        self.na_x.grid(row=3, column=1)
        self.na_x.insert(0, "900")

        ttk.Label(params_frame, text="Sigma (nm):").grid(row=4, column=0, sticky="e")
        self.sigma = ttk.Entry(params_frame)
        self.sigma.grid(row=4, column=1)
        self.sigma.insert(0, "50")

        ttk.Label(params_frame, text="Trap Level (eV):").grid(row=5, column=0, sticky="e")
        self.et_entry = ttk.Entry(params_frame)
        self.et_entry.grid(row=5, column=1)
        self.et_entry.insert(0, "0.3")

        ttk.Label(params_frame, text="V_left (V):").grid(row=6, column=0, sticky="e")
        self.v_left = ttk.Entry(params_frame)
        self.v_left.grid(row=6, column=1)
        self.v_left.insert(0, "0.0")

        ttk.Label(params_frame, text="V_right (V):").grid(row=7, column=0, sticky="e")
        self.v_right = ttk.Entry(params_frame)
        self.v_right.grid(row=7, column=1)
        self.v_right.insert(0, "1.0")

        ttk.Label(params_frame, text="V_bottom (V):").grid(row=8, column=0, sticky="e")
        self.v_bottom = ttk.Entry(params_frame)
        self.v_bottom.grid(row=8, column=1)
        self.v_bottom.insert(0, "0.0")

        ttk.Label(params_frame, text="V_top (V):").grid(row=9, column=0, e="e")
        self.v_top = ttk.Entry(params_frame)
        self.v_top.grid(row=9, column=1)
        self.v_top.insert(0, "0.0")

        self.run_button = ttk.Button(params_frame, text="Run Simulation", command=self.start_simulation)
        self.run_button.grid(row=10, column=0)

        self.stop_button = ttk.Button(params_frame, text="Stop", command=self.stop_simulation, state="disabled")
        self.stop_button.grid(row=10, column=1)

        self.save_button = ttk.Button(params_frame, text="Save Results", command=self.save_results, state="disabled")
        self.save_button.grid(row=11, column=0, columnspan=2)

    def setup_plots(self):
        self.plot_notebook = ttk.Notebook(self.root)
        self.plot_notebook.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.fig_potential, self.ax_potential = plt.subplots(figsize=(6, 4))
        self.canvas_potential = FigureCanvasTkAgg(self.fig_potential, master=self.plot_notebook)
        self.canvas_potential.get_tk_widget().pack()
        self.plot_notebook.add(self.canvas_potential.get_tk_widget(), text="Potential")

        self.fig_density, self.ax_density = plt.subplots(figsize=(6, 4))
        self.canvas_density = FigureCanvasTkAgg(self.fig_density, master=self.plot_notebook)
        self.canvas_density.get_tk_widget().pack()
        self.plot_notebook.add(self.canvas_density.get_tk_widget(), text="Density")

        self.fig_current, self.ax_current = plt.subplots(figsize=(6, 4))
        self.canvas_current = FigureCanvasTkAgg(self.fig_current, master=self.plot_notebook)
        self.canvas_current.get_tk_widget().pack()
        self.plot_notebook.add(self.canvas_current.get_tk_widget(), text="Current")

        self.fig_mesh, self.ax_mesh = plt.subplots(figsize=(6, 4))
        self.canvas_mesh = FigureCanvasTkAgg(self.fig_mesh, master=self.plot_notebook)
        self.canvas_mesh.get_tk_widget().pack()
        self.plot_notebook.add(self.canvas_mesh.get_tk_widget(), text="Mesh")

    def start_simulation(self):
        if self.running:
            messagebox.showinfo("Error", "Simulation already running")
            return
        self.running = True
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.save_button.config(state="disabled")
        threading.Thread(target=self.run_simulation, daemon=True).start()

    def run_simulation(self):
        try:
            Lx = float(self.lx_entry.get()) * 1e-6
            Ly = float(self.ly_entry.get()) * 1e-6
            nx = int(self.nx_entry.get())
            ny = int(self.ny_entry.get())
            # [MODIFICATION]: Pass order to Simulator
            self.sim = Simulator(dimension="2D", extents=[Lx, Ly], num_points_x=nx, num_points_y=ny,
                                 method=self.method_var.get(), mesh_type=self.mesh_var.get(), 
                                 order=self.order_var.get())
            if self.mesh_var.get() == "Unstructured":
                self.sim.generate_mesh(f"device_2d_{self.mesh_var.get().lower()}.msh")

            grid = self.sim.get_grid_points()
            Nd_peak = float(self.nd_peak.get()) * 1e6
            Na_peak = float(self.na_peak.get()) * 1e6
            Nd_x = float(self.nd_x.get()) * 1e-6
            Na_x = float(self.na_x.get()) * 1e-6
            sigma = float(self.sigma.get()) * 1e-6
            Et_val = float(self.et_entry.get())
            Nd = [self.gaussian_doping(x, Nd_x, Nd_peak, sigma) for x in grid["x"]]
            Na = [self.gaussian_doping(x, Na_x, Na_peak, sigma) for x in grid["x"]]
            Et = [Et_val if x > Lx/2 else -Et_val for x in grid["x"]]
            self.sim.set_doping(Nd, Na)
            self.sim.set_trap_level(Et)

            self.results = self.sim.solve_drift_diffusion(
                [float(self.v_left.get()), float(self.v_right.get()),
                 float(self.v_bottom.get()), float(self.v_top.get())],
                max_steps=100, use_amr=self.amr_var.get()
            )

            self.root.after(0, self.update_plots)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Simulation failed: {str(e)}"))
        finally:
            self.running = False
            self.root.after(0, self.reset_buttons)

    def gaussian_doping(self, x, x0, Npeak, sigma):
        return Npeak * np.exp(-((x - x0)**2) / (2 * sigma**2))

    def update_plots(self):
        if not self.results:
            return
        grid = self.sim.get_grid_points()
        Jx = [self.results["Jn"][i * 2] for i in range(len(self.results["Jn"]) // 2)]
        Jy = [self.results["Jn"][i * 2 + 1] for i in range(len(self.results["Jn"]) // 2)]

        self.ax_potential.clear()
        plot_2d_potential(self.ax_potential, grid["x"], grid["y"], self.results["potential"], "Potential (V)")
        self.canvas_potential.draw()

        self.ax_density.clear()
        plot_2d_quantity(self.ax_density, grid["x"], grid["y"], self.results["n"], "Electron Density (m⁻³)")
        self.canvas_density.draw()

        self.ax_current.clear()
        plot_current_vectors(self.ax_current, grid["x"], grid["y"], Jx, Jy, "Electron Current Density")
        self.canvas_current.draw()

        self.ax_mesh.clear()
        for e in self.sim.mesh.get_elements():
            x = [grid["x"][e[0]], grid["x"][e[1]], grid["x"][e[2]], grid["x"][e[0]]]
            y = [grid["y"][e[0]], grid["y"][e[1]], grid["y"][e[2]], grid["y"][e[0]]]
            self.ax_mesh.plot(x, y, 'b-')
        self.ax_mesh.set_title("Refined Mesh")
        self.ax_mesh.set_xlabel("X (m)")
        self.ax_mesh.set_ylabel("Y (m)")
        self.ax_mesh.set_aspect("equal")
        self.canvas_mesh.draw()

        self.save_button.config(state="normal")

    def stop_simulation(self):
        self.running = False
        self.reset_buttons()

    def reset_buttons(self):
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.save_button.config(state="normal" if self.results else "disabled")

    def save_results(self):
        if not self.results:
            return
        np.savez("simulation_results.npz", **self.results)
        messagebox.showinfo("Info", "Results saved to simulation_results.npz")

    def on_closing(self):
        plt.close("all")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorGUI2D(root)
    root.mainloop()