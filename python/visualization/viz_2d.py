from visualization.vulkan_renderer.vulkan_context import VulkanRenderer
import numpy as np

def plot_2d_potential(data, window):
    renderer = VulkanRenderer({"x": data["x"], "y": data["y"], "values": data["potential"]}, plot_type="contour")
    window.setVulkanRenderer(renderer)
    return renderer

def plot_2d_quantity(data, quantity, window):
    renderer = VulkanRenderer({"x": data["x"], "y": data["y"], "values": data[quantity]}, plot_type="contour")
    window.setVulkanRenderer(renderer)
    return renderer

def plot_current_vectors(data, window):
    Jx = data["Jn"][::2]
    Jy = data["Jn"][1::2]
    renderer = VulkanRenderer({"x": data["x"], "y": data["y"], "vectors": np.stack([Jx, Jy], axis=1)}, plot_type="vector")
    window.setVulkanRenderer(renderer)
    return renderer

def plot_mesh(data, window):
    # [Simplified]: Use contour plot for mesh outline
    renderer = VulkanRenderer({"x": data["x"], "y": data["y"], "values": np.zeros_like(data["x"])}, plot_type="contour")
    window.setVulkanRenderer(renderer)
    return renderer