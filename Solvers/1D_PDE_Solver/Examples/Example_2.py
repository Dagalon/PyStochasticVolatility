import numpy as np

from Meshes import Mesh, finite_volume_mesh, uniform_mesh

x_max = 1.0
x_min = -1.0
T = 3.0

mesh_t = Mesh(uniform_mesh, 50, 0.0, T)
mesh_x = Mesh(finite_volume_mesh, 100, x_min, x_max)

r = 0.03
q = 0.01
sigma = 0.3

S0 = 100.0
K = np.exp((r - q) * T) * S0 + 10
