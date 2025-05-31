import numpy as np
import matplotlib.pyplot as plt

def plot_2d_potential(ax, x, y, V, title):
    X, Y = np.meshgrid(x, y)
    V = np.array(V).reshape(len(y), len(x))
    c = ax.contourf(X, Y, V, cmap='viridis')
    ax.colorbar(c, label='Potential (V)')
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

def plot_2d_quantity(ax, x, y, quantity, title):
    X, Y = np.meshgrid(x, y)
    quantity = np.array(quantity).reshape(len(y), len(x))
    c = ax.contourf(X, Y, quantity, cmap='plasma')
    ax.colorbar(c, label=title)
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

def plot_current_vectors(ax, x, y, Jx, Jy, title):
    X, Y = np.meshgrid(x, y)
    Jx = np.array(Jx).reshape(len(y), len(x))
    Jy = np.array(Jy).reshape(len(y), len(x))
    ax.quiver(X, Y, Jx, Jy)
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')