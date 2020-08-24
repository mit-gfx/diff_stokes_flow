import sys
sys.path.append('../')

import numpy as np
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import Scene2d
from py_diff_stokes_flow.common.common import ndarray, print_error

np.random.seed(42)

scene = Scene2d()
# Initialize shape.
scale = 1
cell_nums = (32 * scale, 24 * scale)
control_points_lower = ndarray([
    [32, 10],
    [22, 10],
    [12, 10],
    [0, 4]
]) * scale
control_points_upper = ndarray([
    [0, 20],
    [12, 14],
    [22, 14],
    [32, 14]
]) * scale

scene.InitializeShapeComposition(cell_nums, ['spline', 'spline'],
    [control_points_lower.ravel(), control_points_upper.ravel()])

# Initialize cell.
E = 100
nu = 0.499
threshold = 1e-3
edge_sample_num = 2
scene.InitializeCell(E, nu, threshold, edge_sample_num)

# Initialize Dirichlet boundary conditions.
boundary_dofs = []
boundary_values = []
inlet_velocity = 1.0 * scale
for j in range(cell_nums[1] + 1):
    if control_points_lower[3, 1] < j < control_points_upper[0, 1]:
        boundary_dofs.append(scene.GetNodeDof([0, j], 0))
        boundary_values.append(inlet_velocity)
        boundary_dofs.append(scene.GetNodeDof([0, j], 1))
        boundary_values.append(0.0)
scene.InitializeDirichletBoundaryCondition(boundary_dofs, boundary_values)
scene.InitializeBoundaryType('no_separation')

# Solve for the velocity.
u = ndarray(scene.Solve('eigen'))

# Visualize the results.
u = u.reshape((cell_nums[0] + 1, cell_nums[1] + 1, 2))
import matplotlib.pyplot as plt
from matplotlib import collections as mc

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()
lines = []
colors = []
for i in range(cell_nums[0] + 1):
    for j in range(cell_nums[1] + 1):
        p = ndarray([i, j]) / scale
        q = p + u[i, j] / scale
        lines.append((p, q))
        colors.append('tab:blue')
ax.add_collection(mc.LineCollection(lines, colors=colors, linestyle='-', alpha=0.9))
ax.set_xlim([-1, cell_nums[0] / scale + 1])
ax.set_ylim([-1, cell_nums[1] / scale + 1])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Amplifier2D')
plt.show()