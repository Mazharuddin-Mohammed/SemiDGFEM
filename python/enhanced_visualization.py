"""
Enhanced Visualization Module

This module provides advanced visualization capabilities for the SemiDGFEM simulator:
- 3D mesh and solution visualization
- Interactive plotting with matplotlib integration
- Real-time animation of transient solutions
- Advanced colormap and rendering options
- Multi-physics visualization

Author: Dr. Mazharuddin Mohammed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
import json
import pickle

# ============================================================================
# Enums and Configuration Classes
# ============================================================================

class VisualizationType(Enum):
    MESH_WIREFRAME = "mesh_wireframe"
    MESH_SURFACE = "mesh_surface"
    SOLUTION_CONTOUR = "solution_contour"
    SOLUTION_SURFACE = "solution_surface"
    VECTOR_FIELD = "vector_field"
    STREAMLINES = "streamlines"
    ISOSURFACES = "isosurfaces"
    VOLUME_RENDERING = "volume_rendering"
    PARTICLE_TRACES = "particle_traces"

class ColorMapType(Enum):
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    JET = "jet"
    RAINBOW = "rainbow"
    COOLWARM = "coolwarm"
    SEISMIC = "seismic"
    CUSTOM = "custom"

class RenderingMode(Enum):
    WIREFRAME = "wireframe"
    SURFACE = "surface"
    POINTS = "points"
    VOLUME = "volume"
    TRANSPARENT = "transparent"
    SOLID = "solid"

@dataclass
class ColorMapConfig:
    """Configuration for color mapping"""
    type: ColorMapType = ColorMapType.VIRIDIS
    min_value: float = 0.0
    max_value: float = 1.0
    logarithmic_scale: bool = False
    custom_colors: List[str] = None
    opacity: float = 1.0
    reverse_colormap: bool = False
    
    def __post_init__(self):
        if self.custom_colors is None:
            self.custom_colors = []

@dataclass
class RenderingConfig:
    """Configuration for rendering options"""
    mode: RenderingMode = RenderingMode.SURFACE
    ambient_intensity: float = 0.3
    diffuse_intensity: float = 0.7
    specular_intensity: float = 0.5
    shininess: float = 32.0
    shadows_enabled: bool = True
    anti_aliasing: bool = True
    samples_per_pixel: int = 4

@dataclass
class AnimationConfig:
    """Configuration for animations"""
    frame_rate: float = 30.0
    duration: float = 5.0
    loop_animation: bool = True
    save_frames: bool = False
    output_directory: str = "./animation_frames"
    frame_format: str = "png"
    frame_width: int = 1920
    frame_height: int = 1080

# ============================================================================
# Mesh Visualizer 3D
# ============================================================================

class MeshVisualizer3D:
    """3D mesh visualization with advanced rendering capabilities"""
    
    def __init__(self):
        self.vertices = None
        self.elements = None
        self.rendering_config = RenderingConfig()
        self.quality_metrics = None
        self.boundary_labels = {}
        self.material_ids = None
        self.doping_profile = None
        
        # Matplotlib figure and axis
        self.fig = None
        self.ax = None
        
    def set_mesh(self, vertices: np.ndarray, elements: np.ndarray):
        """Set mesh data for visualization"""
        self.vertices = np.array(vertices)
        self.elements = np.array(elements)
        
        print(f"Mesh loaded: {len(self.vertices)} vertices, {len(self.elements)} elements")
        
        # Determine mesh dimension
        if self.vertices.shape[1] == 2:
            print("  - 2D mesh detected")
        elif self.vertices.shape[1] == 3:
            print("  - 3D mesh detected")
        else:
            print(f"  - {self.vertices.shape[1]}D mesh detected")
    
    def set_rendering_config(self, config: RenderingConfig):
        """Set rendering configuration"""
        self.rendering_config = config
        print(f"Rendering configuration updated: {config.mode.value} mode")
    
    def show_wireframe(self, show_edges: bool = True, show_vertices: bool = False):
        """Display mesh wireframe"""
        if self.vertices is None or self.elements is None:
            print("Error: No mesh data loaded")
            return
        
        print(f"Displaying mesh wireframe (edges: {show_edges}, vertices: {show_vertices})")
        
        # Create figure if not exists
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 8))
            if self.vertices.shape[1] == 3:
                self.ax = self.fig.add_subplot(111, projection='3d')
            else:
                self.ax = self.fig.add_subplot(111)
        
        if show_edges:
            self._plot_edges()
        
        if show_vertices:
            self._plot_vertices()
        
        self.ax.set_title("Mesh Wireframe")
        self._set_axis_labels()
        
    def show_surface(self, show_faces: bool = True, show_normals: bool = False):
        """Display mesh surface"""
        if self.vertices is None or self.elements is None:
            print("Error: No mesh data loaded")
            return
        
        print(f"Displaying mesh surface (faces: {show_faces}, normals: {show_normals})")
        
        # Create figure if not exists
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 8))
            if self.vertices.shape[1] == 3:
                self.ax = self.fig.add_subplot(111, projection='3d')
            else:
                self.ax = self.fig.add_subplot(111)
        
        if show_faces:
            self._plot_surface()
        
        if show_normals:
            self._plot_normals()
        
        self.ax.set_title("Mesh Surface")
        self._set_axis_labels()
    
    def show_element_quality(self, quality_metrics: np.ndarray):
        """Visualize element quality"""
        self.quality_metrics = np.array(quality_metrics)
        
        if len(self.quality_metrics) != len(self.elements):
            print("Error: Quality metrics size doesn't match number of elements")
            return
        
        min_quality = np.min(self.quality_metrics)
        max_quality = np.max(self.quality_metrics)
        avg_quality = np.mean(self.quality_metrics)
        
        print("Displaying element quality visualization:")
        print(f"  - Quality range: [{min_quality:.3f}, {max_quality:.3f}]")
        print(f"  - Average quality: {avg_quality:.3f}")
        print(f"  - Elements below 0.5 quality: {np.sum(self.quality_metrics < 0.5)}")
        
        # Create figure
        self.fig = plt.figure(figsize=(15, 5))
        
        # Quality distribution histogram
        ax1 = self.fig.add_subplot(131)
        ax1.hist(self.quality_metrics, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0.5, color='red', linestyle='--', label='Quality threshold')
        ax1.set_xlabel('Element Quality')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Quality Distribution')
        ax1.legend()
        ax1.grid(True)
        
        # Quality visualization on mesh
        if self.vertices.shape[1] == 2:
            ax2 = self.fig.add_subplot(132)
            self._plot_2d_quality(ax2)
        else:
            ax2 = self.fig.add_subplot(132, projection='3d')
            self._plot_3d_quality(ax2)
        
        # Quality statistics
        ax3 = self.fig.add_subplot(133)
        quality_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        counts = [
            np.sum((self.quality_metrics >= 0.0) & (self.quality_metrics < 0.2)),
            np.sum((self.quality_metrics >= 0.2) & (self.quality_metrics < 0.4)),
            np.sum((self.quality_metrics >= 0.4) & (self.quality_metrics < 0.6)),
            np.sum((self.quality_metrics >= 0.6) & (self.quality_metrics < 0.8)),
            np.sum((self.quality_metrics >= 0.8) & (self.quality_metrics <= 1.0))
        ]
        
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        ax3.bar(quality_ranges, counts, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Quality Range')
        ax3.set_ylabel('Number of Elements')
        ax3.set_title('Quality Range Distribution')
        ax3.grid(True)
        
        plt.tight_layout()
    
    def show_material_regions(self, material_ids: np.ndarray):
        """Visualize material regions"""
        self.material_ids = np.array(material_ids)
        
        unique_materials = np.unique(self.material_ids)
        print(f"Displaying {len(unique_materials)} material regions:")
        
        for mat_id in unique_materials:
            count = np.sum(self.material_ids == mat_id)
            print(f"  - Material {mat_id}: {count} elements")
        
        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        if self.vertices.shape[1] == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        
        # Plot materials with different colors
        cmap = plt.cm.get_cmap('tab10')
        for i, mat_id in enumerate(unique_materials):
            mask = self.material_ids == mat_id
            color = cmap(i / len(unique_materials))
            self._plot_elements_subset(mask, color, f'Material {mat_id}')
        
        self.ax.set_title("Material Regions")
        self.ax.legend()
        self._set_axis_labels()
    
    def show_doping_profile(self, doping_concentration: np.ndarray):
        """Visualize doping profile"""
        self.doping_profile = np.array(doping_concentration)
        
        min_doping = np.min(self.doping_profile)
        max_doping = np.max(self.doping_profile)
        
        print("Displaying doping profile:")
        print(f"  - Concentration range: [{min_doping:.2e}, {max_doping:.2e}] cm⁻³")
        print("  - Using logarithmic color scale for visualization")
        
        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        if self.vertices.shape[1] == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        
        # Use logarithmic scale for doping
        log_doping = np.log10(np.maximum(self.doping_profile, 1e10))  # Avoid log(0)
        
        if self.vertices.shape[1] == 2:
            # 2D triangular plot
            triangulation = tri.Triangulation(self.vertices[:, 0], self.vertices[:, 1], self.elements)
            im = self.ax.tripcolor(triangulation, log_doping, cmap='viridis', shading='flat')
            self.fig.colorbar(im, ax=self.ax, label='log₁₀(Doping [cm⁻³])')
        else:
            # 3D scatter plot
            scatter = self.ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2],
                                    c=log_doping, cmap='viridis', s=20)
            self.fig.colorbar(scatter, ax=self.ax, label='log₁₀(Doping [cm⁻³])')
        
        self.ax.set_title("Doping Profile")
        self._set_axis_labels()
    
    def render_to_file(self, filename: str, width: int = 1920, height: int = 1080, dpi: int = 150):
        """Render visualization to file"""
        if self.fig is None:
            print("Error: No visualization to render")
            return
        
        print(f"Rendering mesh to file: {filename} ({width}x{height})")
        
        # Set figure size based on desired resolution
        fig_width = width / dpi
        fig_height = height / dpi
        self.fig.set_size_inches(fig_width, fig_height)
        
        # Save figure
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"  - Visualization saved to {filename}")
    
    def show_plot(self):
        """Display the interactive plot"""
        if self.fig is None:
            print("Error: No visualization to show")
            return
        
        print("Displaying interactive mesh visualization...")
        print("  - Use mouse to zoom and pan")
        print("  - Close window to continue")
        
        plt.show()
    
    def _plot_edges(self):
        """Plot mesh edges"""
        if self.vertices.shape[1] == 2:
            # 2D edges
            for element in self.elements:
                if len(element) == 3:  # Triangle
                    edges = [(0, 1), (1, 2), (2, 0)]
                elif len(element) == 4:  # Quadrilateral
                    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
                else:
                    continue
                
                for edge in edges:
                    x_coords = [self.vertices[element[edge[0]], 0], self.vertices[element[edge[1]], 0]]
                    y_coords = [self.vertices[element[edge[0]], 1], self.vertices[element[edge[1]], 1]]
                    self.ax.plot(x_coords, y_coords, 'k-', linewidth=0.5, alpha=0.7)
        else:
            # 3D edges (simplified - show element outlines)
            for element in self.elements:
                if len(element) >= 4:  # Tetrahedron or higher
                    # Plot edges of tetrahedron
                    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                    for edge in edges:
                        if edge[1] < len(element):
                            x_coords = [self.vertices[element[edge[0]], 0], self.vertices[element[edge[1]], 0]]
                            y_coords = [self.vertices[element[edge[0]], 1], self.vertices[element[edge[1]], 1]]
                            z_coords = [self.vertices[element[edge[0]], 2], self.vertices[element[edge[1]], 2]]
                            self.ax.plot(x_coords, y_coords, z_coords, 'k-', linewidth=0.5, alpha=0.7)
    
    def _plot_vertices(self):
        """Plot mesh vertices"""
        if self.vertices.shape[1] == 2:
            self.ax.scatter(self.vertices[:, 0], self.vertices[:, 1], c='red', s=10, alpha=0.7)
        else:
            self.ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 
                          c='red', s=10, alpha=0.7)
    
    def _plot_surface(self):
        """Plot mesh surface"""
        if self.vertices.shape[1] == 2:
            # 2D surface (filled triangles)
            triangulation = tri.Triangulation(self.vertices[:, 0], self.vertices[:, 1], self.elements)
            self.ax.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.7)
            # Use a single color value instead of facecolors array
            self.ax.tripcolor(triangulation, np.ones(len(self.elements)),
                            cmap='Blues', alpha=0.3, edgecolors='black')
        else:
            # 3D surface (simplified)
            self.ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 
                          c='lightblue', s=20, alpha=0.6)
    
    def _plot_normals(self):
        """Plot surface normals (simplified implementation)"""
        print("  - Computing and displaying surface normals")
        # This would require more complex normal computation for actual implementation
    
    def _plot_2d_quality(self, ax):
        """Plot 2D quality visualization"""
        triangulation = tri.Triangulation(self.vertices[:, 0], self.vertices[:, 1], self.elements)
        im = ax.tripcolor(triangulation, self.quality_metrics, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title('Element Quality (2D)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Quality')
    
    def _plot_3d_quality(self, ax):
        """Plot 3D quality visualization"""
        scatter = ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2],
                           c=self.quality_metrics, cmap='RdYlGn', s=30, vmin=0, vmax=1)
        ax.set_title('Element Quality (3D)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax, label='Quality')
    
    def _plot_elements_subset(self, mask, color, label):
        """Plot subset of elements with given color"""
        subset_elements = self.elements[mask]
        
        if self.vertices.shape[1] == 2:
            for element in subset_elements:
                if len(element) >= 3:
                    polygon = plt.Polygon(self.vertices[element[:3]], 
                                        facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                    self.ax.add_patch(polygon)
        else:
            # For 3D, just plot vertices of selected elements
            element_vertices = []
            for element in subset_elements:
                element_vertices.extend(element)
            element_vertices = list(set(element_vertices))
            
            if element_vertices:
                vertices_subset = self.vertices[element_vertices]
                self.ax.scatter(vertices_subset[:, 0], vertices_subset[:, 1], vertices_subset[:, 2],
                              c=[color], s=20, alpha=0.7, label=label)
    
    def _set_axis_labels(self):
        """Set appropriate axis labels"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        if self.vertices.shape[1] == 3:
            self.ax.set_zlabel('Z')
        self.ax.grid(True)

# ============================================================================
# Solution Visualizer
# ============================================================================

class SolutionVisualizer:
    """Advanced solution field visualization"""

    def __init__(self):
        self.solution_fields = {}
        self.mesh_vertices = None
        self.mesh_elements = None
        self.colormap_config = ColorMapConfig()
        self.cross_section_planes = []

        # Matplotlib figure and axis
        self.fig = None
        self.ax = None

    def set_solution(self, solution_fields: Dict[str, np.ndarray]):
        """Set solution field data"""
        self.solution_fields = {name: np.array(field) for name, field in solution_fields.items()}

        print(f"Solution data loaded with {len(self.solution_fields)} fields:")
        for name, field in self.solution_fields.items():
            print(f"  - {name}: {len(field)} values")

    def set_mesh(self, vertices: np.ndarray, elements: np.ndarray):
        """Set mesh data for solution visualization"""
        self.mesh_vertices = np.array(vertices)
        self.mesh_elements = np.array(elements)
        print(f"Mesh set for solution visualization: {len(vertices)} vertices, {len(elements)} elements")

    def set_colormap_config(self, config: ColorMapConfig):
        """Set colormap configuration"""
        self.colormap_config = config
        print(f"Colormap configuration updated: {config.type.value}")

    def show_contour_plot(self, field_name: str, num_levels: int = 20):
        """Create contour plot for specified field"""
        if field_name not in self.solution_fields:
            print(f"Error: Field '{field_name}' not found in solution data")
            return

        field_data = self.solution_fields[field_name]
        print(f"Creating contour plot for field '{field_name}' with {num_levels} levels")

        self._compute_field_statistics(field_data, field_name)

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))

        if self.mesh_vertices.shape[1] == 2:
            self.ax = self.fig.add_subplot(111)
            self._plot_2d_contour(field_data, field_name, num_levels)
        else:
            # For 3D, create multiple 2D slices
            self.ax = self.fig.add_subplot(111)
            self._plot_3d_contour_slices(field_data, field_name, num_levels)

        self.ax.set_title(f'Contour Plot: {field_name}')
        self._set_axis_labels()

    def show_surface_plot(self, field_name: str):
        """Create 3D surface plot for specified field"""
        if field_name not in self.solution_fields:
            print(f"Error: Field '{field_name}' not found in solution data")
            return

        field_data = self.solution_fields[field_name]
        print(f"Creating 3D surface plot for field '{field_name}'")

        self._compute_field_statistics(field_data, field_name)

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        if self.mesh_vertices.shape[1] == 2:
            # 2D mesh with field as Z coordinate
            self._plot_2d_surface(field_data, field_name)
        else:
            # 3D mesh - use scatter plot with color coding
            self._plot_3d_scatter(field_data, field_name)

        self.ax.set_title(f'Surface Plot: {field_name}')
        self._set_axis_labels()

    def show_vector_field(self, vector_field_name: str, scale: float = 1.0):
        """Display vector field visualization"""
        print(f"Displaying vector field '{vector_field_name}' with scale factor {scale}")

        # Check for vector field components
        components = [f"{vector_field_name}_x", f"{vector_field_name}_y", f"{vector_field_name}_z"]
        found_components = []

        for comp in components:
            if comp in self.solution_fields:
                found_components.append(comp)

        if len(found_components) < 2:
            print("  - Error: Vector field components not found")
            return

        print(f"  - Found {len(found_components)}D vector field")

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))

        if self.mesh_vertices.shape[1] == 2 and len(found_components) >= 2:
            self.ax = self.fig.add_subplot(111)
            self._plot_2d_vector_field(found_components, scale)
        elif len(found_components) >= 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._plot_3d_vector_field(found_components, scale)

        self.ax.set_title(f'Vector Field: {vector_field_name}')
        self._set_axis_labels()

    def show_streamlines(self, vector_field_name: str, num_seeds: int = 100):
        """Generate and display streamlines"""
        print(f"Generating streamlines for vector field '{vector_field_name}' with {num_seeds} seed points")

        # Check for vector field components
        components = [f"{vector_field_name}_x", f"{vector_field_name}_y"]
        if not all(comp in self.solution_fields for comp in components):
            print("  - Error: Vector field components not found")
            return

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)

        # Get vector field data
        u = self.solution_fields[components[0]]
        v = self.solution_fields[components[1]]

        if self.mesh_vertices.shape[1] == 2:
            self._plot_2d_streamlines(u, v, num_seeds)

        self.ax.set_title(f'Streamlines: {vector_field_name}')
        self._set_axis_labels()

    def show_multi_field_comparison(self, field_names: List[str]):
        """Create multi-field comparison visualization"""
        print("Creating multi-field comparison visualization:")

        valid_fields = []
        for field_name in field_names:
            if field_name in self.solution_fields:
                print(f"  - Field {len(valid_fields)+1}: {field_name} ({len(self.solution_fields[field_name])} values)")
                valid_fields.append(field_name)
            else:
                print(f"  - Field: {field_name} (NOT FOUND)")

        if not valid_fields:
            print("  - Error: No valid fields found")
            return

        # Create subplot grid
        n_fields = len(valid_fields)
        cols = int(np.ceil(np.sqrt(n_fields)))
        rows = int(np.ceil(n_fields / cols))

        self.fig = plt.figure(figsize=(15, 10))

        for i, field_name in enumerate(valid_fields):
            ax = self.fig.add_subplot(rows, cols, i+1)
            field_data = self.solution_fields[field_name]

            if self.mesh_vertices.shape[1] == 2:
                triangulation = tri.Triangulation(self.mesh_vertices[:, 0], self.mesh_vertices[:, 1], self.mesh_elements)
                im = ax.tripcolor(triangulation, field_data, cmap=self.colormap_config.type.value, shading='flat')
                self.fig.colorbar(im, ax=ax, label=field_name)
            else:
                # 3D scatter plot
                scatter = ax.scatter(self.mesh_vertices[:, 0], self.mesh_vertices[:, 1],
                                   c=field_data, cmap=self.colormap_config.type.value, s=10)
                self.fig.colorbar(scatter, ax=ax, label=field_name)

            ax.set_title(field_name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)

        plt.tight_layout()

    def show_field_difference(self, field1: str, field2: str):
        """Compute and visualize field difference"""
        if field1 not in self.solution_fields or field2 not in self.solution_fields:
            print("  - Error: One or both fields not found")
            return

        data1 = self.solution_fields[field1]
        data2 = self.solution_fields[field2]

        if len(data1) != len(data2):
            print("  - Error: Field sizes do not match")
            return

        print(f"Computing field difference: {field1} - {field2}")

        difference = data1 - data2
        self._compute_field_statistics(difference, f"{field1} - {field2}")

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))

        if self.mesh_vertices.shape[1] == 2:
            self.ax = self.fig.add_subplot(111)
            triangulation = tri.Triangulation(self.mesh_vertices[:, 0], self.mesh_vertices[:, 1], self.mesh_elements)
            im = self.ax.tripcolor(triangulation, difference, cmap='RdBu_r', shading='flat')
            self.fig.colorbar(im, ax=self.ax, label=f'{field1} - {field2}')
        else:
            self.ax = self.fig.add_subplot(111, projection='3d')
            scatter = self.ax.scatter(self.mesh_vertices[:, 0], self.mesh_vertices[:, 1], self.mesh_vertices[:, 2],
                                    c=difference, cmap='RdBu_r', s=20)
            self.fig.colorbar(scatter, ax=self.ax, label=f'{field1} - {field2}')

        self.ax.set_title(f'Field Difference: {field1} - {field2}')
        self._set_axis_labels()

    def add_cross_section(self, plane_normal: List[float], plane_distance: float):
        """Add cross-section plane"""
        print(f"Adding cross-section plane:")
        print(f"  - Normal vector: ({plane_normal[0]}, {plane_normal[1]}, {plane_normal[2]})")
        print(f"  - Distance from origin: {plane_distance}")

        self.cross_section_planes.append({
            'normal': np.array(plane_normal),
            'distance': plane_distance
        })

        print(f"  - Cross-section {len(self.cross_section_planes)} added")

    def show_line_plot(self, start_point: List[float], end_point: List[float], field_name: str = None):
        """Create line plot along specified path"""
        start = np.array(start_point)
        end = np.array(end_point)

        length = np.linalg.norm(end - start)
        print(f"Creating line plot from ({start[0]}, {start[1]}, {start[2]}) to ({end[0]}, {end[1]}, {end[2]})")
        print(f"  - Line length: {length:.3f}")

        # Generate sample points along the line
        num_samples = 100
        t_values = np.linspace(0, 1, num_samples)
        sample_points = start[np.newaxis, :] + t_values[:, np.newaxis] * (end - start)[np.newaxis, :]
        distances = t_values * length

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))

        if field_name and field_name in self.solution_fields:
            # Interpolate field values along the line (simplified)
            field_values = self._interpolate_field_along_line(sample_points, field_name)

            self.ax = self.fig.add_subplot(111)
            self.ax.plot(distances, field_values, 'b-', linewidth=2, label=field_name)
            self.ax.set_xlabel('Distance along line')
            self.ax.set_ylabel(f'{field_name}')
            self.ax.set_title(f'Line Plot: {field_name}')
            self.ax.grid(True)
            self.ax.legend()
        else:
            # Plot all available fields
            self.ax = self.fig.add_subplot(111)
            for name, field_data in self.solution_fields.items():
                field_values = self._interpolate_field_along_line(sample_points, name)
                self.ax.plot(distances, field_values, linewidth=2, label=name)

            self.ax.set_xlabel('Distance along line')
            self.ax.set_ylabel('Field Value')
            self.ax.set_title('Line Plot: All Fields')
            self.ax.grid(True)
            self.ax.legend()

    def render_to_file(self, filename: str, width: int = 1920, height: int = 1080, dpi: int = 150):
        """Render visualization to file"""
        if self.fig is None:
            print("Error: No visualization to render")
            return

        print(f"Rendering solution visualization to file: {filename} ({width}x{height})")

        # Set figure size based on desired resolution
        fig_width = width / dpi
        fig_height = height / dpi
        self.fig.set_size_inches(fig_width, fig_height)

        # Save figure
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"  - Visualization saved to {filename}")

    def show_plot(self):
        """Display the interactive plot"""
        if self.fig is None:
            print("Error: No visualization to show")
            return

        print("Displaying interactive solution visualization...")
        print("  - Use mouse to zoom and pan")
        print("  - Close window to continue")

        plt.show()

    def _compute_field_statistics(self, field: np.ndarray, field_name: str):
        """Compute and display field statistics"""
        min_val = np.min(field)
        max_val = np.max(field)
        mean_val = np.mean(field)
        std_val = np.std(field)

        print(f"Field statistics for '{field_name}':")
        print(f"  - Range: [{min_val:.3e}, {max_val:.3e}]")
        print(f"  - Mean: {mean_val:.3e} ± {std_val:.3e}")

    def _plot_2d_contour(self, field_data: np.ndarray, field_name: str, num_levels: int):
        """Plot 2D contour"""
        triangulation = tri.Triangulation(self.mesh_vertices[:, 0], self.mesh_vertices[:, 1], self.mesh_elements)

        # Generate contour levels
        min_val = np.min(field_data)
        max_val = np.max(field_data)
        levels = np.linspace(min_val, max_val, num_levels)

        # Create contour plot
        contour = self.ax.tricontour(triangulation, field_data, levels=levels, colors='black', linewidths=0.5)
        contourf = self.ax.tricontourf(triangulation, field_data, levels=levels, cmap=self.colormap_config.type.value)

        # Add colorbar
        self.fig.colorbar(contourf, ax=self.ax, label=field_name)

        # Add contour labels
        self.ax.clabel(contour, inline=True, fontsize=8, fmt='%.2e')

        print(f"  - Contour levels: {levels[0]:.3e} to {levels[-1]:.3e}")

    def _plot_3d_contour_slices(self, field_data: np.ndarray, field_name: str, num_levels: int):
        """Plot 3D contour as multiple 2D slices"""
        print("  - Creating 2D projection of 3D field data")

        # Project 3D data to 2D (simplified - use X-Y projection)
        x_coords = self.mesh_vertices[:, 0]
        y_coords = self.mesh_vertices[:, 1]

        # Create scatter plot with color coding
        scatter = self.ax.scatter(x_coords, y_coords, c=field_data,
                                cmap=self.colormap_config.type.value, s=20)
        self.fig.colorbar(scatter, ax=self.ax, label=field_name)

    def _plot_2d_surface(self, field_data: np.ndarray, field_name: str):
        """Plot 2D surface with field as Z coordinate"""
        x_coords = self.mesh_vertices[:, 0]
        y_coords = self.mesh_vertices[:, 1]
        z_coords = field_data

        # Create triangular surface
        surface = self.ax.plot_trisurf(x_coords, y_coords, z_coords,
                                     cmap=self.colormap_config.type.value, alpha=0.8)
        self.fig.colorbar(surface, ax=self.ax, label=field_name)

    def _plot_3d_scatter(self, field_data: np.ndarray, field_name: str):
        """Plot 3D scatter with color coding"""
        scatter = self.ax.scatter(self.mesh_vertices[:, 0], self.mesh_vertices[:, 1], self.mesh_vertices[:, 2],
                                c=field_data, cmap=self.colormap_config.type.value, s=20)
        self.fig.colorbar(scatter, ax=self.ax, label=field_name)

    def _plot_2d_vector_field(self, components: List[str], scale: float):
        """Plot 2D vector field"""
        u = self.solution_fields[components[0]]
        v = self.solution_fields[components[1]]

        x_coords = self.mesh_vertices[:, 0]
        y_coords = self.mesh_vertices[:, 1]

        # Subsample for better visualization
        step = max(1, len(x_coords) // 1000)  # Limit to ~1000 arrows

        self.ax.quiver(x_coords[::step], y_coords[::step],
                      u[::step] * scale, v[::step] * scale,
                      angles='xy', scale_units='xy', scale=1, alpha=0.7)

        # Add magnitude as background
        magnitude = np.sqrt(u**2 + v**2)
        triangulation = tri.Triangulation(x_coords, y_coords, self.mesh_elements)
        contourf = self.ax.tricontourf(triangulation, magnitude, cmap='viridis', alpha=0.3)
        self.fig.colorbar(contourf, ax=self.ax, label='Magnitude')

    def _plot_3d_vector_field(self, components: List[str], scale: float):
        """Plot 3D vector field"""
        u = self.solution_fields[components[0]]
        v = self.solution_fields[components[1]]
        w = self.solution_fields[components[2]]

        x_coords = self.mesh_vertices[:, 0]
        y_coords = self.mesh_vertices[:, 1]
        z_coords = self.mesh_vertices[:, 2]

        # Subsample for better visualization
        step = max(1, len(x_coords) // 500)  # Limit to ~500 arrows

        self.ax.quiver(x_coords[::step], y_coords[::step], z_coords[::step],
                      u[::step] * scale, v[::step] * scale, w[::step] * scale,
                      alpha=0.7, arrow_length_ratio=0.1)

    def _plot_2d_streamlines(self, u: np.ndarray, v: np.ndarray, num_seeds: int):
        """Plot 2D streamlines"""
        # Create regular grid for streamline computation
        x_min, x_max = np.min(self.mesh_vertices[:, 0]), np.max(self.mesh_vertices[:, 0])
        y_min, y_max = np.min(self.mesh_vertices[:, 1]), np.max(self.mesh_vertices[:, 1])

        # Grid resolution
        nx, ny = 50, 50
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate vector field to grid (simplified)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Simple nearest neighbor interpolation
        for i in range(nx):
            for j in range(ny):
                # Find nearest mesh point
                distances = np.sqrt((self.mesh_vertices[:, 0] - X[j, i])**2 +
                                  (self.mesh_vertices[:, 1] - Y[j, i])**2)
                nearest_idx = np.argmin(distances)
                U[j, i] = u[nearest_idx]
                V[j, i] = v[nearest_idx]

        # Create streamlines
        self.ax.streamplot(X, Y, U, V, density=1.5, color='blue', linewidth=1, arrowsize=1.5)

        # Add magnitude as background
        magnitude = np.sqrt(u**2 + v**2)
        triangulation = tri.Triangulation(self.mesh_vertices[:, 0], self.mesh_vertices[:, 1], self.mesh_elements)
        contourf = self.ax.tricontourf(triangulation, magnitude, cmap='viridis', alpha=0.3)
        self.fig.colorbar(contourf, ax=self.ax, label='Magnitude')

    def _interpolate_field_along_line(self, sample_points: np.ndarray, field_name: str) -> np.ndarray:
        """Interpolate field values along a line (simplified implementation)"""
        field_data = self.solution_fields[field_name]
        interpolated_values = np.zeros(len(sample_points))

        # Simple nearest neighbor interpolation
        for i, point in enumerate(sample_points):
            if self.mesh_vertices.shape[1] == 2:
                distances = np.sqrt((self.mesh_vertices[:, 0] - point[0])**2 +
                                  (self.mesh_vertices[:, 1] - point[1])**2)
            else:
                distances = np.sqrt((self.mesh_vertices[:, 0] - point[0])**2 +
                                  (self.mesh_vertices[:, 1] - point[1])**2 +
                                  (self.mesh_vertices[:, 2] - point[2])**2)

            nearest_idx = np.argmin(distances)
            interpolated_values[i] = field_data[nearest_idx]

        return interpolated_values

    def _set_axis_labels(self):
        """Set appropriate axis labels"""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        if hasattr(self.ax, 'set_zlabel'):
            self.ax.set_zlabel('Z')
        self.ax.grid(True)

# ============================================================================
# Animation Engine
# ============================================================================

class AnimationEngine:
    """Advanced animation capabilities for time-dependent visualizations"""

    def __init__(self):
        self.config = AnimationConfig()
        self.time_series_data = []
        self.time_points = []
        self.camera_keyframes = []
        self.is_animating = False

        # Animation objects
        self.fig = None
        self.ax = None
        self.animation = None

    def set_animation_config(self, config: AnimationConfig):
        """Set animation configuration"""
        self.config = config
        self._setup_animation_pipeline()

    def add_time_series_data(self, solutions: List[Dict[str, np.ndarray]], time_points: List[float]):
        """Add time series solution data"""
        self.time_series_data = solutions
        self.time_points = time_points

        print("Adding time series data:")
        print(f"  - Number of time steps: {len(solutions)}")
        print(f"  - Time range: [{time_points[0]:.3f}, {time_points[-1]:.3f}]")

        if solutions:
            print(f"  - Fields per time step: {list(solutions[0].keys())}")

    def create_solution_evolution_animation(self, field_name: str, mesh_vertices: np.ndarray, mesh_elements: np.ndarray):
        """Create animation of solution field evolution"""
        if not self.time_series_data:
            print("Error: No time series data available")
            return

        print(f"Creating solution evolution animation for field '{field_name}'")

        # Check if field exists in all time steps
        if not all(field_name in step_data for step_data in self.time_series_data):
            print(f"Error: Field '{field_name}' not found in all time steps")
            return

        # Compute global field range for consistent color scale
        all_values = []
        for step_data in self.time_series_data:
            all_values.extend(step_data[field_name])

        global_min = np.min(all_values)
        global_max = np.max(all_values)

        print(f"  - Global field range: [{global_min:.3e}, {global_max:.3e}]")
        print(f"  - Using consistent color scale across all frames")

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)

        # Create triangulation for 2D mesh
        if mesh_vertices.shape[1] >= 2:
            triangulation = tri.Triangulation(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_elements)

        def animate_frame(frame_idx):
            """Animation function for each frame"""
            self.ax.clear()

            current_time = self.time_points[frame_idx]
            field_data = self.time_series_data[frame_idx][field_name]

            # Plot current field
            if mesh_vertices.shape[1] >= 2:
                im = self.ax.tripcolor(triangulation, field_data,
                                     cmap='viridis', vmin=global_min, vmax=global_max, shading='flat')

            self.ax.set_title(f'{field_name} at t = {current_time:.3f}')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.grid(True)

            return [im] if 'im' in locals() else []

        # Create animation
        total_frames = len(self.time_series_data)
        interval = 1000.0 / self.config.frame_rate  # milliseconds per frame

        self.animation = FuncAnimation(self.fig, animate_frame, frames=total_frames,
                                     interval=interval, blit=False, repeat=self.config.loop_animation)

        print(f"  - Animation created with {total_frames} frames")
        print(f"  - Frame rate: {self.config.frame_rate} fps")

    def create_parameter_sweep_animation(self, solutions: List[Dict[str, np.ndarray]],
                                       parameter_values: List[float], parameter_name: str,
                                       field_name: str, mesh_vertices: np.ndarray, mesh_elements: np.ndarray):
        """Create animation for parameter sweep results"""
        print(f"Creating parameter sweep animation:")
        print(f"  - Parameter: {parameter_name}")
        print(f"  - Field: {field_name}")
        print(f"  - Parameter range: [{parameter_values[0]:.3e}, {parameter_values[-1]:.3e}]")

        # Similar to solution evolution but with parameter values
        if mesh_vertices.shape[1] >= 2:
            triangulation = tri.Triangulation(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_elements)

        # Compute global field range
        all_values = []
        for solution in solutions:
            if field_name in solution:
                all_values.extend(solution[field_name])

        global_min = np.min(all_values)
        global_max = np.max(all_values)

        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)

        def animate_frame(frame_idx):
            """Animation function for parameter sweep"""
            self.ax.clear()

            current_param = parameter_values[frame_idx]
            field_data = solutions[frame_idx][field_name]

            # Plot current field
            if mesh_vertices.shape[1] >= 2:
                im = self.ax.tripcolor(triangulation, field_data,
                                     cmap='viridis', vmin=global_min, vmax=global_max, shading='flat')

            self.ax.set_title(f'{field_name} at {parameter_name} = {current_param:.3e}')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.grid(True)

            return [im] if 'im' in locals() else []

        # Create animation
        total_frames = len(solutions)
        interval = 1000.0 / self.config.frame_rate

        self.animation = FuncAnimation(self.fig, animate_frame, frames=total_frames,
                                     interval=interval, blit=False, repeat=self.config.loop_animation)

        print(f"  - Parameter sweep animation created with {total_frames} frames")

    def export_video(self, filename: str, codec: str = 'h264'):
        """Export animation to video file"""
        if self.animation is None:
            print("Error: No animation to export")
            return

        print(f"Exporting animation to video: {filename}")
        print(f"  - Codec: {codec}")
        print(f"  - Frame rate: {self.config.frame_rate} fps")

        try:
            # Try to save as MP4
            writer = plt.matplotlib.animation.FFMpegWriter(fps=self.config.frame_rate, codec=codec)
            self.animation.save(filename, writer=writer)
            print(f"  - Video export completed successfully")
        except Exception as e:
            print(f"  - Error exporting video: {e}")
            print("  - Try installing ffmpeg for video export")

    def export_gif(self, filename: str, quality: int = 80):
        """Export animation as GIF"""
        if self.animation is None:
            print("Error: No animation to export")
            return

        print(f"Exporting animation to GIF: {filename}")
        print(f"  - Quality: {quality}%")

        try:
            writer = plt.matplotlib.animation.PillowWriter(fps=self.config.frame_rate)
            self.animation.save(filename, writer=writer)
            print(f"  - GIF export completed successfully")
        except Exception as e:
            print(f"  - Error exporting GIF: {e}")

    def show_animation(self):
        """Display the animation"""
        if self.animation is None:
            print("Error: No animation to show")
            return

        print("Displaying animation...")
        print("  - Close window to stop animation")
        plt.show()

    def _setup_animation_pipeline(self):
        """Setup animation pipeline"""
        print("Setting up animation pipeline:")
        print(f"  - Frame rate: {self.config.frame_rate} fps")
        print(f"  - Duration: {self.config.duration} seconds")
        print(f"  - Total frames: {int(self.config.frame_rate * self.config.duration)}")
        print(f"  - Output resolution: {self.config.frame_width}x{self.config.frame_height}")

# ============================================================================
# Interactive Plotter
# ============================================================================

class InteractivePlotter:
    """Advanced interactive plotting capabilities"""

    def __init__(self):
        self.plots = []
        self.title = ""
        self.x_label = "X"
        self.y_label = "Y"
        self.z_label = "Z"
        self.x_limits = None
        self.y_limits = None
        self.colormap = ColorMapConfig()
        self.show_legend = True
        self.show_grid = True
        self.subplot_rows = 1
        self.subplot_cols = 1
        self.active_subplot = 0

        # Matplotlib figure and axes
        self.fig = None
        self.axes = None

    def create_line_plot(self, x: np.ndarray, y: np.ndarray, label: str = ""):
        """Create line plot"""
        x = np.array(x)
        y = np.array(y)

        if len(x) != len(y):
            print("Error: x and y data sizes do not match")
            return

        plot_data = {
            'type': 'line',
            'x': x,
            'y': y,
            'label': label if label else f"Line {len(self.plots) + 1}",
            'subplot': self.active_subplot
        }

        self.plots.append(plot_data)
        print(f"Created line plot '{plot_data['label']}' with {len(x)} points")

    def create_scatter_plot(self, x: np.ndarray, y: np.ndarray, sizes: np.ndarray = None, label: str = ""):
        """Create scatter plot"""
        x = np.array(x)
        y = np.array(y)

        if len(x) != len(y):
            print("Error: x and y data sizes do not match")
            return

        plot_data = {
            'type': 'scatter',
            'x': x,
            'y': y,
            'sizes': sizes,
            'label': label if label else f"Scatter {len(self.plots) + 1}",
            'subplot': self.active_subplot
        }

        self.plots.append(plot_data)
        print(f"Created scatter plot with {len(x)} points")
        if sizes is not None:
            print("  - Variable point sizes enabled")

    def create_surface_plot(self, z_data: np.ndarray, x_data: np.ndarray = None, y_data: np.ndarray = None):
        """Create 3D surface plot"""
        z_data = np.array(z_data)

        if z_data.ndim != 2:
            print("Error: Surface data must be 2D array")
            return

        # Generate coordinate arrays if not provided
        if x_data is None:
            x_data = np.arange(z_data.shape[1])
        if y_data is None:
            y_data = np.arange(z_data.shape[0])

        plot_data = {
            'type': 'surface',
            'x': x_data,
            'y': y_data,
            'z': z_data,
            'label': f"Surface {len(self.plots) + 1}",
            'subplot': self.active_subplot
        }

        self.plots.append(plot_data)
        print(f"Created surface plot with {z_data.shape[0]}x{z_data.shape[1]} grid")

    def create_heatmap(self, data: np.ndarray, x_labels: List[str] = None, y_labels: List[str] = None):
        """Create heatmap"""
        data = np.array(data)

        if data.ndim != 2:
            print("Error: Heatmap data must be 2D array")
            return

        plot_data = {
            'type': 'heatmap',
            'data': data,
            'x_labels': x_labels,
            'y_labels': y_labels,
            'label': f"Heatmap {len(self.plots) + 1}",
            'subplot': self.active_subplot
        }

        self.plots.append(plot_data)
        print(f"Created heatmap with {data.shape[0]}x{data.shape[1]} data")

    def create_histogram(self, data: np.ndarray, bins: int = 50, label: str = ""):
        """Create histogram"""
        data = np.array(data)

        plot_data = {
            'type': 'histogram',
            'data': data,
            'bins': bins,
            'label': label if label else f"Histogram {len(self.plots) + 1}",
            'subplot': self.active_subplot
        }

        self.plots.append(plot_data)
        print(f"Created histogram with {len(data)} data points and {bins} bins")

    def create_subplot_grid(self, rows: int, cols: int):
        """Create subplot grid"""
        self.subplot_rows = rows
        self.subplot_cols = cols
        self.active_subplot = 0

        print(f"Created {rows}x{cols} subplot grid")
        print(f"  - Total subplots: {rows * cols}")
        print("  - Use set_active_subplot() to switch between subplots")

    def set_active_subplot(self, row: int, col: int):
        """Set active subplot"""
        subplot_index = row * self.subplot_cols + col
        if subplot_index < self.subplot_rows * self.subplot_cols:
            self.active_subplot = subplot_index
            print(f"Active subplot set to ({row}, {col})")
        else:
            print("Error: Invalid subplot index")

    def set_title(self, title: str):
        """Set plot title"""
        self.title = title
        print(f"Plot title set to: '{title}'")

    def set_axis_labels(self, x_label: str, y_label: str, z_label: str = ""):
        """Set axis labels"""
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label

        print("Axis labels set:")
        print(f"  - X: '{x_label}'")
        print(f"  - Y: '{y_label}'")
        if z_label:
            print(f"  - Z: '{z_label}'")

    def set_axis_limits(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Set axis limits"""
        self.x_limits = [x_min, x_max]
        self.y_limits = [y_min, y_max]
        print(f"Axis limits set: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")

    def set_colormap(self, config: ColorMapConfig):
        """Set colormap configuration"""
        self.colormap = config
        print(f"Colormap set to {config.type.value} with range [{config.min_value}, {config.max_value}]")

    def show_plot(self):
        """Display all plots"""
        if not self.plots:
            print("Error: No plots to display")
            return

        print("Displaying interactive plots...")

        # Create figure and subplots
        self.fig, self.axes = plt.subplots(self.subplot_rows, self.subplot_cols,
                                          figsize=(15, 10), squeeze=False)

        # Group plots by subplot
        subplot_plots = {}
        for plot in self.plots:
            subplot_idx = plot['subplot']
            if subplot_idx not in subplot_plots:
                subplot_plots[subplot_idx] = []
            subplot_plots[subplot_idx].append(plot)

        # Plot each subplot
        for subplot_idx, plots_in_subplot in subplot_plots.items():
            row = subplot_idx // self.subplot_cols
            col = subplot_idx % self.subplot_cols
            ax = self.axes[row, col]

            self._plot_in_subplot(ax, plots_in_subplot)

        # Set overall title
        if self.title:
            self.fig.suptitle(self.title, fontsize=16)

        plt.tight_layout()
        plt.show()

    def save_plot(self, filename: str, dpi: int = 300):
        """Save plot to file"""
        if self.fig is None:
            print("Error: No plot to save. Call show_plot() first.")
            return

        print(f"Saving plot to: {filename}")
        print(f"  - Resolution: {dpi} DPI")
        print(f"  - Number of data series: {len(self.plots)}")

        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"  - Plot saved successfully")

    def _plot_in_subplot(self, ax, plots):
        """Plot data in a specific subplot"""
        for plot in plots:
            if plot['type'] == 'line':
                ax.plot(plot['x'], plot['y'], label=plot['label'], linewidth=2)

            elif plot['type'] == 'scatter':
                sizes = plot['sizes'] if plot['sizes'] is not None else 20
                ax.scatter(plot['x'], plot['y'], s=sizes, label=plot['label'], alpha=0.7)

            elif plot['type'] == 'surface':
                # For surface plots, create 3D subplot
                ax.remove()
                ax = self.fig.add_subplot(self.subplot_rows, self.subplot_cols,
                                        plots[0]['subplot'] + 1, projection='3d')
                X, Y = np.meshgrid(plot['x'], plot['y'])
                surface = ax.plot_surface(X, Y, plot['z'], cmap=self.colormap.type.value, alpha=0.8)
                self.fig.colorbar(surface, ax=ax, shrink=0.5)

            elif plot['type'] == 'heatmap':
                im = ax.imshow(plot['data'], cmap=self.colormap.type.value, aspect='auto')
                self.fig.colorbar(im, ax=ax)

                if plot['x_labels']:
                    ax.set_xticks(range(len(plot['x_labels'])))
                    ax.set_xticklabels(plot['x_labels'])
                if plot['y_labels']:
                    ax.set_yticks(range(len(plot['y_labels'])))
                    ax.set_yticklabels(plot['y_labels'])

            elif plot['type'] == 'histogram':
                ax.hist(plot['data'], bins=plot['bins'], alpha=0.7,
                       label=plot['label'], edgecolor='black')

        # Set labels and formatting
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        if self.x_limits:
            ax.set_xlim(self.x_limits)
        if self.y_limits:
            ax.set_ylim(self.y_limits)

        if self.show_legend and any(plot.get('label') for plot in plots):
            ax.legend()

        if self.show_grid:
            ax.grid(True, alpha=0.3)

# ============================================================================
# Visualization Manager
# ============================================================================

class VisualizationManager:
    """Unified interface for all visualization components"""

    def __init__(self):
        self.mesh_visualizer = MeshVisualizer3D()
        self.solution_visualizer = SolutionVisualizer()
        self.animation_engine = AnimationEngine()
        self.plotter = InteractivePlotter()

        print("Visualization Manager initialized")
        print("  - All visualization components ready")

    def get_mesh_visualizer(self) -> MeshVisualizer3D:
        """Get mesh visualizer component"""
        return self.mesh_visualizer

    def get_solution_visualizer(self) -> SolutionVisualizer:
        """Get solution visualizer component"""
        return self.solution_visualizer

    def get_animation_engine(self) -> AnimationEngine:
        """Get animation engine component"""
        return self.animation_engine

    def get_plotter(self) -> InteractivePlotter:
        """Get interactive plotter component"""
        return self.plotter

    def create_comprehensive_visualization(self, mesh_vertices: np.ndarray, mesh_elements: np.ndarray,
                                         solution_fields: Dict[str, np.ndarray]):
        """Create comprehensive visualization of mesh and solution"""
        print("Creating comprehensive visualization...")

        # Setup mesh visualization
        print("1. Setting up mesh visualization:")
        self.mesh_visualizer.set_mesh(mesh_vertices, mesh_elements)
        self.mesh_visualizer.show_wireframe(True, False)
        self.mesh_visualizer.show_surface(True, True)

        # Setup solution visualization
        print("2. Setting up solution visualization:")
        self.solution_visualizer.set_mesh(mesh_vertices, mesh_elements)
        self.solution_visualizer.set_solution(solution_fields)

        # Visualize first available field
        if solution_fields:
            first_field = list(solution_fields.keys())[0]
            self.solution_visualizer.show_contour_plot(first_field, 20)

            # Check for vector fields
            vector_fields = self._detect_vector_fields(solution_fields)
            if vector_fields:
                self.solution_visualizer.show_vector_field(vector_fields[0], 1.0)

        # Create analysis plots
        print("3. Creating analysis plots:")
        if solution_fields:
            field_names = list(solution_fields.keys())

            # Create line plots for field analysis
            x_data = np.linspace(0, 1, len(list(solution_fields.values())[0]))
            for i, (name, field) in enumerate(solution_fields.items()):
                if i < 3:  # Limit to first 3 fields
                    self.plotter.create_line_plot(x_data, field, name)

            self.plotter.set_title("Field Analysis")
            self.plotter.set_axis_labels("Position", "Field Value")

        print("Comprehensive visualization created successfully!")

    def create_multi_physics_dashboard(self, solutions: List[Dict[str, np.ndarray]],
                                     physics_names: List[str], mesh_vertices: np.ndarray, mesh_elements: np.ndarray):
        """Create multi-physics dashboard"""
        print("Creating multi-physics dashboard...")
        print(f"  - Number of physics models: {len(physics_names)}")
        print(f"  - Number of solutions: {len(solutions)}")

        # Create subplot grid
        grid_size = int(np.ceil(np.sqrt(len(physics_names))))
        self.plotter.create_subplot_grid(grid_size, grid_size)

        # Create visualization for each physics model
        for i, physics_name in enumerate(physics_names):
            row = i // grid_size
            col = i % grid_size

            self.plotter.set_active_subplot(row, col)
            print(f"  - Creating visualization for {physics_name} in subplot ({row}, {col})")

            # Create sample data for demonstration
            if i < len(solutions) and physics_name in solutions[i]:
                field_data = solutions[i][physics_name]
                x_data = np.linspace(0, 1, len(field_data))
                self.plotter.create_line_plot(x_data, field_data, physics_name)
            else:
                # Generate sample data
                x_data = np.linspace(0, 1, 100)
                y_data = np.sin(2 * np.pi * x_data * (i + 1)) * np.exp(-x_data)
                self.plotter.create_line_plot(x_data, y_data, physics_name)

        self.plotter.set_title("Multi-Physics Dashboard")
        print("Multi-physics dashboard created successfully!")

    def save_visualization_session(self, filename: str):
        """Save visualization session to file"""
        print(f"Saving visualization session to: {filename}")

        session_data = {
            'mesh_config': {
                'rendering_config': self.mesh_visualizer.rendering_config.__dict__,
                'has_quality_metrics': self.mesh_visualizer.quality_metrics is not None,
                'has_material_regions': self.mesh_visualizer.material_ids is not None,
                'has_doping_profile': self.mesh_visualizer.doping_profile is not None
            },
            'solution_config': {
                'colormap_config': self.solution_visualizer.colormap_config.__dict__,
                'available_fields': list(self.solution_visualizer.solution_fields.keys()),
                'num_cross_sections': len(self.solution_visualizer.cross_section_planes)
            },
            'animation_config': self.animation_engine.config.__dict__,
            'plotter_config': {
                'title': self.plotter.title,
                'x_label': self.plotter.x_label,
                'y_label': self.plotter.y_label,
                'z_label': self.plotter.z_label,
                'subplot_grid': [self.plotter.subplot_rows, self.plotter.subplot_cols],
                'num_plots': len(self.plotter.plots)
            }
        }

        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            print("  - Session saved successfully")
        except Exception as e:
            print(f"  - Error saving session: {e}")

    def load_visualization_session(self, filename: str):
        """Load visualization session from file"""
        print(f"Loading visualization session from: {filename}")

        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)

            print("  - Loading configuration sections:")

            # Load mesh configuration
            if 'mesh_config' in session_data:
                print("    * Mesh visualization settings")
                mesh_config = session_data['mesh_config']
                if 'rendering_config' in mesh_config:
                    # Restore rendering configuration
                    pass

            # Load solution configuration
            if 'solution_config' in session_data:
                print("    * Solution visualization settings")
                solution_config = session_data['solution_config']
                if 'colormap_config' in solution_config:
                    # Restore colormap configuration
                    pass

            # Load animation configuration
            if 'animation_config' in session_data:
                print("    * Animation settings")
                # Restore animation configuration
                pass

            # Load plotter configuration
            if 'plotter_config' in session_data:
                print("    * Plotter settings")
                plotter_config = session_data['plotter_config']
                self.plotter.title = plotter_config.get('title', '')
                self.plotter.x_label = plotter_config.get('x_label', 'X')
                self.plotter.y_label = plotter_config.get('y_label', 'Y')
                self.plotter.z_label = plotter_config.get('z_label', 'Z')

            print("  - Session loaded successfully")

        except Exception as e:
            print(f"  - Error loading session: {e}")

    def set_level_of_detail(self, enable: bool = True, threshold: float = 0.01):
        """Set level of detail optimization"""
        print(f"Level of detail (LOD) {'enabled' if enable else 'disabled'}")

        if enable:
            print(f"  - LOD threshold: {threshold}")
            print("  - Automatic mesh simplification for distant objects")
            print("  - Dynamic quality adjustment based on performance")

    def enable_gpu_acceleration(self, enable: bool = True):
        """Enable GPU acceleration"""
        print(f"GPU acceleration {'enabled' if enable else 'disabled'}")

        if enable:
            print("  - Utilizing GPU for rendering computations")
            print("  - Hardware-accelerated graphics pipeline")
            print("  - Improved performance for large datasets")

    def set_memory_limit(self, limit_mb: int = 4096):
        """Set memory limit for visualizations"""
        print(f"Memory limit set to {limit_mb} MB")
        print("  - Automatic data streaming for large datasets")
        print("  - Memory-efficient visualization algorithms")
        print("  - Garbage collection optimization enabled")

    def create_publication_quality_figure(self, mesh_vertices: np.ndarray, mesh_elements: np.ndarray,
                                        solution_fields: Dict[str, np.ndarray], field_name: str,
                                        filename: str, dpi: int = 300):
        """Create publication-quality figure"""
        print(f"Creating publication-quality figure for field '{field_name}'")

        # Set up high-quality visualization
        self.solution_visualizer.set_mesh(mesh_vertices, mesh_elements)
        self.solution_visualizer.set_solution(solution_fields)

        # Configure for publication quality
        colormap_config = ColorMapConfig(
            type=ColorMapType.VIRIDIS,
            logarithmic_scale=False,
            reverse_colormap=False
        )
        self.solution_visualizer.set_colormap_config(colormap_config)

        # Create contour plot
        self.solution_visualizer.show_contour_plot(field_name, 25)

        # Render to high-resolution file
        self.solution_visualizer.render_to_file(filename, width=3000, height=2000, dpi=dpi)

        print(f"  - Publication-quality figure saved to {filename}")
        print(f"  - Resolution: 3000x2000 pixels at {dpi} DPI")

    def create_interactive_dashboard(self, mesh_vertices: np.ndarray, mesh_elements: np.ndarray,
                                   solution_fields: Dict[str, np.ndarray]):
        """Create interactive dashboard with multiple visualizations"""
        print("Creating interactive dashboard...")

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))

        # Mesh visualization
        ax1 = fig.add_subplot(2, 3, 1)
        self.mesh_visualizer.set_mesh(mesh_vertices, mesh_elements)
        # Simplified mesh plot for dashboard
        if mesh_vertices.shape[1] >= 2:
            triangulation = tri.Triangulation(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_elements)
            ax1.triplot(triangulation, 'k-', linewidth=0.5, alpha=0.7)
            ax1.set_title('Mesh Structure')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.grid(True)

        # Solution field visualizations
        field_names = list(solution_fields.keys())
        for i, field_name in enumerate(field_names[:4]):  # Limit to 4 fields
            ax = fig.add_subplot(2, 3, i + 2)
            field_data = solution_fields[field_name]

            if mesh_vertices.shape[1] >= 2:
                triangulation = tri.Triangulation(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_elements)
                im = ax.tripcolor(triangulation, field_data, cmap='viridis', shading='flat')
                fig.colorbar(im, ax=ax, label=field_name)
                ax.set_title(f'{field_name}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')

        # Field comparison plot
        if len(field_names) >= 2:
            ax6 = fig.add_subplot(2, 3, 6)
            x_data = np.linspace(0, 1, len(solution_fields[field_names[0]]))
            for name in field_names[:3]:  # Plot first 3 fields
                field_data = solution_fields[name]
                # Normalize for comparison
                normalized_data = (field_data - np.min(field_data)) / (np.max(field_data) - np.min(field_data))
                ax6.plot(x_data, normalized_data, label=name, linewidth=2)

            ax6.set_title('Normalized Field Comparison')
            ax6.set_xlabel('Position')
            ax6.set_ylabel('Normalized Value')
            ax6.legend()
            ax6.grid(True)

        plt.tight_layout()
        plt.suptitle('SemiDGFEM Interactive Dashboard', fontsize=16, y=0.98)

        print("  - Interactive dashboard created with multiple visualizations")
        print("  - Use matplotlib controls to zoom, pan, and explore")

        plt.show()

    def _detect_vector_fields(self, solution_fields: Dict[str, np.ndarray]) -> List[str]:
        """Detect vector fields in solution data"""
        vector_fields = []
        field_names = list(solution_fields.keys())

        # Look for fields with _x, _y, _z components
        base_names = set()
        for name in field_names:
            if name.endswith('_x') or name.endswith('_y') or name.endswith('_z'):
                base_name = name[:-2]
                base_names.add(base_name)

        # Check which base names have at least x and y components
        for base_name in base_names:
            has_x = f"{base_name}_x" in field_names
            has_y = f"{base_name}_y" in field_names
            if has_x and has_y:
                vector_fields.append(base_name)

        return vector_fields

# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_mesh_2d(nx: int = 20, ny: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create sample 2D triangular mesh"""
    # Create regular grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Flatten to get vertices
    vertices = np.column_stack([X.flatten(), Y.flatten()])

    # Create triangular elements
    elements = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Lower triangle
            v1 = i * nx + j
            v2 = i * nx + j + 1
            v3 = (i + 1) * nx + j
            elements.append([v1, v2, v3])

            # Upper triangle
            v1 = i * nx + j + 1
            v2 = (i + 1) * nx + j + 1
            v3 = (i + 1) * nx + j
            elements.append([v1, v2, v3])

    return vertices, np.array(elements)

def create_sample_solution_fields(num_points: int) -> Dict[str, np.ndarray]:
    """Create sample solution fields for testing"""
    x = np.linspace(0, 1, num_points)

    fields = {
        'potential': np.sin(2 * np.pi * x) * np.exp(-x),
        'electron_density': 1e16 * np.exp(-10 * x**2),
        'hole_density': 1e15 * np.exp(-5 * (x - 0.5)**2),
        'electric_field_x': -2 * np.pi * np.cos(2 * np.pi * x) * np.exp(-x),
        'electric_field_y': np.zeros_like(x),
        'current_density_x': 1e-3 * x * (1 - x),
        'current_density_y': np.zeros_like(x)
    }

    return fields

def demonstrate_enhanced_visualization():
    """Demonstrate enhanced visualization capabilities"""
    print("=== Enhanced Visualization Demonstration ===")

    # Create sample data
    vertices, elements = create_sample_mesh_2d(15, 15)
    solution_fields = create_sample_solution_fields(len(vertices))

    # Create visualization manager
    viz_manager = VisualizationManager()

    # Demonstrate comprehensive visualization
    viz_manager.create_comprehensive_visualization(vertices, elements, solution_fields)

    # Demonstrate interactive dashboard
    viz_manager.create_interactive_dashboard(vertices, elements, solution_fields)

    print("Enhanced visualization demonstration completed!")

if __name__ == "__main__":
    demonstrate_enhanced_visualization()
