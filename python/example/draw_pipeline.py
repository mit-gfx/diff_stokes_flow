import sys
sys.path.append('../')

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import collections as mc

from py_diff_stokes_flow.env.refinement_env_2d import RefinementEnv2d
from py_diff_stokes_flow.common.common import ndarray, print_error, create_folder

if __name__ == '__main__':
    folder = Path('draw_pipeline')
    create_folder(folder, exist_ok=True)

    nu = 0.45
    scale = 0.75
    env = RefinementEnv2d(0.45, scale)

    _, info = env.solve(env.sample(), False, { 'solver': 'eigen' })
    u = info[0]['velocity_field']
    node_nums = env.node_nums()
    sdf = np.zeros(node_nums)
    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            sdf[i, j] = info[0]['scene'].GetSignedDistance((i, j))
            if sdf[i, j] >= 0:
                u[i, j] = 0

    # Draw design parameters.
    plt.rc('font', size=20)
    plt.rc('axes', titlesize=24)
    plt.rc('axes', labelsize=24)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=20)
    plt.rc('figure', titlesize=20)
    face_color = ndarray([247 / 255, 247 / 255, 247 / 255])
    plt.rcParams['figure.facecolor'] = face_color
    plt.rcParams['axes.facecolor'] = face_color
    padding = 5

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    lines = []
    # Draw horizontal boundaries.
    cell_nums = [n - 1 for n in node_nums]
    for i in range(cell_nums[0]):
        v0 = ndarray([i, 0])
        v1 = ndarray([i + 1, 0])
        lines.append((v0, v1))
        v0 = ndarray([i, cell_nums[1]])
        v1 = ndarray([i + 1, cell_nums[1]])
        lines.append((v0, v1))

    def intercept(d0, d1):
        # (0, d0), (t, 0), (1, d1).
        # t / -d0 = 1 / (d1 - d0).
        return -d0 / (d1 - d0)

    # Draw vertical boundaries.
    thin_lines = []
    for j in range(cell_nums[1]):
        for i in [0, cell_nums[0]]:
            d0 = sdf[i, j]
            d1 = sdf[i, j + 1]
            v0 = ndarray([i, j])
            v1 = ndarray([i, j + 1])
            if d0 >= 0 and d1 >= 0:
                lines.append((v0, v1))
            elif d0 * d1 < 0:
                t = intercept(d0, d1)
                vt = (1 - t) * v0 + t * v1
                if d0 > 0:
                    lines.append((v0, vt))
                    thin_lines.append((vt, v1))
                else:
                    lines.append((vt, v1))
                    thin_lines.append((v0, vt))
            else:
                thin_lines.append((v0, v1))

    # Draw the interface.
    intercepts = []
    for i in range(node_nums[0]):
        ds = set()
        for j in range(cell_nums[1]):
            d0 = sdf[i, j]
            d1 = sdf[i, j + 1]
            if d0 * d1 <= 0:
                ds.add(j + intercept(d0, d1))
        ds = sorted(tuple(ds))
        assert len(ds) == 2
        intercepts.append(ds)

    interfaces = []
    for i in range(cell_nums[0]):
        for k in [0, 1]:
            v0 = ndarray([i, intercepts[i][k]])
            v1 = ndarray([i + 1, intercepts[i + 1][k]])
            interfaces.append((v0, v1))

    ax.add_collection(mc.LineCollection(interfaces, colors='k', linewidth=4.0))
    ax.add_collection(mc.LineCollection(thin_lines, colors='k', linewidth=1.0))
    ax.add_collection(mc.LineCollection(lines, colors='k', linewidth=4.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-padding, cell_nums[0] + padding])
    ax.set_ylim([-padding, cell_nums[1] + padding])
    ax.axis('off')

    fig.savefig(folder / 'design.png')
    plt.show()

    # Draw figures for the governing equations.
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    ax.add_collection(mc.LineCollection(interfaces, colors='tab:orange', linewidth=4.0))
    ax.add_collection(mc.LineCollection(thin_lines, colors='tab:blue', linewidth=1.0))
    ax.add_collection(mc.LineCollection(lines, colors='k', linewidth=4.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-padding, cell_nums[0] + padding])
    ax.set_ylim([-padding, cell_nums[1] + padding])
    ax.axis('off')

    fig.savefig(folder / 'governing_eqns.png')
    plt.show()

    # Draw the discretization figure.
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    cell_edges = []
    for i in range(cell_nums[0]):
        for j in range(cell_nums[1]):
            v00 = ndarray([i, j])
            v10 = ndarray([i + 1, j])
            v01 = ndarray([i, j + 1])
            if j != 0:
                cell_edges.append((v00, v10))
            if i != 0:
                cell_edges.append((v00, v01))

    ax.add_collection(mc.LineCollection(cell_edges, colors='k', linewidth=1.0))
    ax.add_collection(mc.LineCollection(interfaces, colors='tab:orange', linewidth=4.0))
    ax.add_collection(mc.LineCollection(thin_lines, colors='tab:blue', linewidth=1.0))
    ax.add_collection(mc.LineCollection(lines, colors='k', linewidth=4.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-padding, cell_nums[0] + padding])
    ax.set_ylim([-padding, cell_nums[1] + padding])
    ax.axis('off')

    fig.savefig(folder / 'discretization.png')
    plt.show()

    # Draw the simulation figure.
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    # How to use cmap:
    # cmap(0.0) to cmap(1.0) covers the whole range of the colormap.
    cmap = plt.get_cmap('coolwarm')
    velocities = []
    velocity_colors = []
    u_min = np.inf
    u_max = -np.inf
    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            uij = u[i, j]
            uij_norm = np.linalg.norm(uij)
            if uij_norm > 0:
                if uij_norm > u_max:
                    u_max = uij_norm
                if uij_norm < u_min:
                    u_min = uij_norm

    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            uij = u[i, j]
            uij_norm = np.linalg.norm(uij)
            if uij_norm > 0:
                v0 = ndarray([i, j])
                v1 = v0 + uij
                velocities.append((v0, v1))
                # Determine the color.
                color = cmap((uij_norm - u_min) / (u_max - u_min))
                velocity_colors.append(color)

    ax.add_collection(mc.LineCollection(velocities, colors=velocity_colors, linewidth=4.0))
    ax.add_collection(mc.LineCollection(cell_edges, colors='k', linewidth=1.0))
    ax.add_collection(mc.LineCollection(interfaces, colors='tab:orange', linewidth=4.0))
    ax.add_collection(mc.LineCollection(thin_lines, colors='tab:blue', linewidth=1.0))
    ax.add_collection(mc.LineCollection(lines, colors='k', linewidth=4.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-padding, cell_nums[0] + padding])
    ax.set_ylim([-padding, cell_nums[1] + padding])
    ax.axis('off')

    fig.savefig(folder / 'simulation.png')
    plt.show()