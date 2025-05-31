import numpy as np
from ctypes import cdll, c_int, c_double, c_char_p, POINTER, Structure, c_bool

class Mesh(Structure):
    _fields_ = []

class Device(Structure):
    _fields_ = []

class Simulator:
    def __init__(self, dimension="TwoD", extents=[1e-6, 0.5e-6], num_points_x=50, num_points_y=25,
                 method="DG", mesh_type="Structured", regions=None, order="P3"): # [MODIFICATION]: Added order
        self.lib = cdll.LoadLibrary("./simulator/libsimulator.so")
        self.dimension = dimension
        self.extents = extents
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.method = {"FDM": 0, "FEM": 1, "FVM": 2, "SEM": 3, "MC": 4, "DG": 5}[method]
        self.mesh_type = {"Structured": 0, "Unstructured": 1}[mesh_type]
        self.order = {"P2": 2, "P3": 3}[order] # [MODIFICATION]: Map order to integer
        self.regions = regions or []

        self.lib.create_device.argtypes = [c_double, c_double]
        self.lib.create_device.restype = POINTER(Device)
        self.device = self.lib.create_device(extents[0], extents[1])

        self.lib.create_mesh.argtypes = [POINTER(Device), c_int]
        self.lib.create_mesh.restype = POINTER(Mesh)
        self.mesh = self.lib.create_mesh(self.device, self.mesh_type)

        self.lib.create_drift_diffusion.argtypes = [POINTER(Device), c_int, c_int, c_int] # [MODIFICATION]: Added order
        self.lib.drift_diffusion_solve_drift_diffusion.argtypes = [
            POINTER(Device), POINTER(c_double), c_int, c_double, c_int, c_bool, 
            POINTER(c_double), c_int, c_int
        ]
        self.lib.drift_diffusion_solve_drift_diffusion.restype = None

        # [MODIFICATION]: Initialize with order
        self.drift_diffusion = self.lib.create_drift_diffusion(self.device, self.method, self.mesh_type, self.order)

    def generate_mesh(self, filename):
        self.lib.mesh_generate_gmsh_mesh(self.mesh, c_char_p(filename.encode('utf-8')))

    def get_grid_points(self):
        grid_x = (c_double * self.num_points_x)()
        grid_y = (c_double * self.num_points_y)()
        self.lib.get_grid_points_x(self.mesh, grid_x, self.num_points_x)
        self.lib.get_grid_points_y(self.mesh, grid_y, self.num_points_y)
        return {"x": list(grid_x), "y": list(grid_y)}

    def set_doping(self, Nd, Na):
        Nd_array = (c_double * len(Nd))(*Nd)
        Na_array = (c_double * len(Na))(*Na)
        self.lib.set_doping(self.device, Nd_array, Na_array, len(Nd))

    def set_trap_level(self, Et):
        Et_array = (c_double * len(Et))(*Et)
        self.lib.set_trap_level(self.device, Et_array, len(Et))

    def solve_drift_diffusion(self, bc, Vg=0.0, max_steps=100, use_amr=False):
        bc_array = (c_double * len(bc))(*bc)
        n_nodes = self.num_points_x * self.num_points_y
        results = (c_double * (5 * n_nodes))()
        self.lib.drift_diffusion_solve_drift_diffusion(
            self.device, bc_array, len(bc), Vg, max_steps, use_amr, results, 5, n_nodes
        )
        results = list(results)
        return {
            "potential": results[:n_nodes],
            "n": results[n_nodes:2*n_nodes],
            "p": results[2*n_nodes:3*n_nodes],
            "Jn": results[3*n_nodes:4*n_nodes],
            "Jp": results[4*n_nodes:5*n_nodes]
        }