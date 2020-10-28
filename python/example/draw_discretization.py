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
    folder = Path('draw_discretization')
    create_folder(folder, exist_ok=True)

    nu = 0.45
    scale = 0.75
    env = RefinementEnv2d(0.45, scale)

    _, info = env.solve(env.sample(), False, { 'solver': 'eigen' })
    u = info['velocity_field']
    node_nums = env.node_nums()
    sdf = np.zeros(node_nums)
    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            sdf[i, j] = info['scene'].GetSignedDistance((i, j))
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

    # Highlight one particular cell.
    ci, cj = [3, 4]
    cell_highlight_lines = []
    v00 = ndarray([ci - 0.1, cj - 0.1])
    v01 = ndarray([ci - 0.1, cj + 1.1])
    v10 = ndarray([ci + 1.1, cj - 0.1])
    v11 = ndarray([ci + 1.1, cj + 1.1])
    cell_highlight_lines.append((v00, v10))
    cell_highlight_lines.append((v10, v11))
    cell_highlight_lines.append((v11, v01))
    cell_highlight_lines.append((v01, v00))

    ax.add_collection(mc.LineCollection(cell_edges, colors='k', linewidth=1.0))
    ax.add_collection(mc.LineCollection(interfaces, colors='tab:orange', linewidth=4.0))
    ax.add_collection(mc.LineCollection(thin_lines, colors='tab:blue', linewidth=1.0))
    ax.add_collection(mc.LineCollection(lines, colors='k', linewidth=4.0))
    ax.add_collection(mc.LineCollection(cell_highlight_lines, colors=(196 / 255, 30 / 255, 58 / 255), linewidth=4.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-padding, cell_nums[0] + padding])
    ax.set_ylim([-padding, cell_nums[1] + padding])
    ax.set_aspect('equal')
    ax.axis('off')

    fig.savefig(folder / 'discretization.png')
    plt.show()
    plt.close()

    # Plot the zoom-in version.
    subdivision_lines = []
    v0 = ndarray([ci - 1, cj + 0.5])
    v1 = ndarray([ci + 1 + 1, cj + 0.5])
    subdivision_lines.append((v0, v1))
    v0 = ndarray([ci + 0.5, cj - 1])
    v1 = ndarray([ci + 0.5, cj + 1 + 1])
    subdivision_lines.append((v0, v1))
    ax.add_collection(mc.LineCollection(subdivision_lines, colors='k', linestyle='-.', linewidth=2.0))

    # Plot the Gaussian quadratures.
    for i in range(2):
        for j in range(2):
            v = ndarray([ci + 0.25 + i * 0.5, cj + 0.25 + j * 0.5])
            ax.scatter(v[0], v[1], c='tab:blue' if sdf[ci + i, cj + j] < 0 else 'tab:orange', s=100.0)

    # Plot the polygon.
    polygon_lines = []
    # Compute the intercept first.
    a = intercepts[ci][0]
    b = intercepts[ci + 1][0]
    v0 = ndarray([ci, a])
    v1 = ndarray([ci + 1, b])
    v2 = ndarray([ci + 0.5, (a + b) / 2])
    # (x - ci) / (cj + 0.5 - a) = 1 / (b - a)
    v3 = ndarray([(cj + 0.5 - a) / (b - a) + ci, cj + 0.5])
    padding = 0.001
    # Top left polygon.
    v_top_left = ndarray([
        [ci + padding, cj + 1 - padding],
        [ci + 0.5 - padding, cj + 1 - padding],
        [v2[0] - padding, v2[1]],
        [v3[0], v3[1] + padding],
        [ci + padding, cj + 0.5 + padding]
    ])

    # Bottom left polygon.
    v_bottom_left = ndarray([
        [ci + 0.5 + padding, cj + 1 - padding],
        [ci + 1 - padding, cj + 1 - padding],
        [v1[0] - padding, v1[1]],
        [ci + 0.5 + padding, v2[1]]
    ])

    # Top right polygon.
    v_top_right = ndarray([
        [ci + padding, cj + 0.5 - padding],
        [ci + padding, v0[1]],
        [v3[0], v3[1] - padding]
    ])

    for v in [v_top_left, v_bottom_left, v_top_right]:
        for k in range(len(v)):
            polygon_lines.append((v[k], v[(k + 1) % len(v)]))

    ax.add_collection(mc.LineCollection(polygon_lines, colors='tab:blue', linewidth=2.0))
    ax.set_xlim([ci - 0.1, ci + 1.1])
    ax.set_ylim([cj - 0.1, cj + 1.1])
    ax.set_aspect('equal')
    fig.savefig(folder / 'zoom_in.png')
    plt.show()
    plt.close()