import sys
sys.path.append('../')

import numpy as np
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import Scene2d, StdRealVector
from py_diff_stokes_flow.common.common import ndarray, print_error

np.random.seed(42)

cell_nums = (32, 24)
control_points_lower = ndarray([
    [32, 10],
    [22, 10],
    [12, 10],
    [0, 4]
])
control_points_upper = ndarray([
    [0, 20],
    [12, 14],
    [22, 14],
    [32, 14]
])

def solve_forward_amplifier_2d(nu, scale):
    scene = Scene2d()
    # Initialize shape.
    scaled_cell_nums = (cell_nums[0] * scale, cell_nums[1] * scale)
    scaled_control_points_lower = control_points_lower * scale
    scaled_control_points_upper = control_points_upper * scale

    scene.InitializeShapeComposition(scaled_cell_nums, ['bezier', 'bezier'],
        [scaled_control_points_lower.ravel(), scaled_control_points_upper.ravel()])

    # Initialize cell.
    E = 100
    threshold = 1e-3
    edge_sample_num = 2
    scene.InitializeCell(E, nu, threshold, edge_sample_num)

    # Initialize Dirichlet boundary conditions.
    boundary_dofs = []
    boundary_values = []
    inlet_velocity = 1.0 * scale
    for j in range(scaled_cell_nums[1] + 1):
        if scaled_control_points_lower[3, 1] < j < scaled_control_points_upper[0, 1]:
            boundary_dofs.append(scene.GetNodeDof([0, j], 0))
            boundary_values.append(inlet_velocity)
            boundary_dofs.append(scene.GetNodeDof([0, j], 1))
            boundary_values.append(0.0)
    scene.InitializeDirichletBoundaryCondition(boundary_dofs, boundary_values)
    scene.InitializeBoundaryType('no_separation')

    # Solve for the velocity.
    u = StdRealVector(0)
    dl_du = np.zeros((scaled_cell_nums[0] + 1) * (scaled_cell_nums[1] + 1) * 2)
    dl_dparam = StdRealVector(0)
    scene.Solve('eigen', dl_du, u, dl_dparam)
    u = ndarray(u).reshape((scaled_cell_nums[0] + 1, scaled_cell_nums[1] + 1, 2)) / scale

    # Zero out velocities in the solid phase.
    for i in range(scaled_cell_nums[0] + 1):
        for j in range(scaled_cell_nums[1] + 1):
            if scene.GetSignedDistance((i, j)) >= 0:
                u[i, j] = 0
    return u

nu = [0.45, 0.47, 0.48, 0.49, 0.495, 0.499]
scale = [1, 2, 4, 8, 16, 32]
u_nu = [solve_forward_amplifier_2d(n, scale[0]) for n in nu]
u_nu_norm_max = np.max(np.sqrt(np.sum(u_nu[-1] ** 2, axis=(2,))).ravel())
u_scale = [solve_forward_amplifier_2d(nu[-1], s) for s in scale]
u_scale_norm_max = np.max(np.sqrt(np.sum(u_scale[-1] ** 2, axis=(2,))).ravel())

import matplotlib.pyplot as plt
import matplotlib.colors

plt.rc('font', size=16)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=16)

def plot_velocity_field(ax, u_field, s, u_max):
    x_size = ndarray(np.arange(u_field.shape[0])) / s
    y_size = ndarray(np.arange(u_field.shape[1])) / s
    X, Y = np.meshgrid(x_size, y_size)
    u_norm = np.sqrt(np.sum(u_field ** 2, axis=(2,)))
    strm = ax.streamplot(X, Y, u_field[:, :, 0].T, u_field[:, :, 1].T, density=(cell_nums[0] / cell_nums[1] * 3, 3),
        color=u_norm.T, norm=matplotlib.colors.Normalize(0, u_max),
        arrowstyle='->', linewidth=1.5, cmap='coolwarm')
    ax.set_xlim([0, cell_nums[0]])
    ax.set_ylim([0, cell_nums[1]])
    ax.set_xticks([])
    ax.set_yticks([])

# The under refinement experiment.
fig = plt.figure(figsize=(18, 9))
ax_low_res = fig.add_subplot(231)
plot_velocity_field(ax_low_res, u_scale[0], scale[0], u_scale_norm_max)
ax_low_res.set_xlabel('{} x {}'.format(cell_nums[0] * scale[0], cell_nums[1] * scale[0]))
ax_low_res.set_aspect('equal', 'box')

ax_mid_res = fig.add_subplot(232)
plot_velocity_field(ax_mid_res, u_scale[3], scale[3], u_scale_norm_max)
ax_mid_res.set_xlabel('{} x {}'.format(cell_nums[0] * scale[3], cell_nums[1] * scale[3]))
ax_mid_res.set_aspect('equal', 'box')

ax_high_res = fig.add_subplot(233)
plot_velocity_field(ax_high_res, u_scale[-1], scale[-1], u_scale_norm_max)
ax_high_res.set_xlabel('{} x {}'.format(cell_nums[0] * scale[-1], cell_nums[1] * scale[-1]))
ax_high_res.set_aspect('equal', 'box')

# The convergence experiment.
ax_low_nu = fig.add_subplot(234)
plot_velocity_field(ax_low_nu, u_nu[0], scale[0], u_nu_norm_max)
ax_low_nu.set_xlabel('$\\nu$ = {}'.format(nu[0]))
ax_low_nu.set_aspect('equal', 'box')

ax_mid_nu = fig.add_subplot(235)
plot_velocity_field(ax_mid_nu, u_nu[3], scale[0], u_nu_norm_max)
ax_mid_nu.set_xlabel('$\\nu$ = {}'.format(nu[3]))
ax_mid_nu.set_aspect('equal', 'box')

ax_high_nu = fig.add_subplot(236)
plot_velocity_field(ax_high_nu, u_nu[-1], scale[0], u_nu_norm_max)
ax_high_nu.set_xlabel('$\\nu$ = {}'.format(nu[-1]))
ax_high_nu.set_aspect('equal', 'box')

fig.savefig('refinement_right.pdf')
plt.show()

fig = plt.figure(figsize=(9, 9))
ax_error = fig.add_subplot(211)
error = []
for s, u_s in zip(scale, u_scale):
    # Compute the maximum discrepancy.
    max_error_s = -np.inf
    for i in range(u_s.shape[0]):
        for j in range(u_s.shape[1]):
            ii = int(i * (scale[-1] / s))
            jj = int(j * (scale[-1] / s))
            error_ij = np.max(np.abs(u_s[i, j] - u_scale[-1][ii, jj])) / u_scale_norm_max
            if error_ij > max_error_s:
                max_error_s = error_ij
    error.append(max_error_s)
ax_error.plot(scale, ndarray(error) * 100, 'o-', linewidth=2, color='tab:blue', markersize=8)
ax_error.set_xlabel('refinement')
ax_error.set_ylabel('relative error (%)')
ax_error.grid(True)

ax_error = fig.add_subplot(212)
error = [np.max(np.abs(u_n - u_nu[-1])) / u_nu_norm_max for u_n in u_nu]
ax_error.plot(nu, ndarray(error) * 100, 'o-', linewidth=2, color='tab:blue', markersize=8)
ax_error.set_xlabel('$\\nu$')
ax_error.set_ylabel('relative error (%)')
ax_error.grid(True)

fig.savefig('refinement_left.pdf')
plt.show()