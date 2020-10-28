import sys
sys.path.append('../')

import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import collections as mc

from py_diff_stokes_flow.env.amplifier_env_2d import AmplifierEnv2d
from py_diff_stokes_flow.common.common import ndarray, print_error, create_folder

if __name__ == '__main__':
    # This script assumes the data folder exists.
    data_folder = Path('amplifier')
    cnt = 0
    while True:
        data_file_name = data_folder / '{:04d}.data'.format(cnt)
        if not os.path.exists(data_file_name):
            cnt -= 1
            break
        cnt += 1
    data_file_name = data_folder / '{:04d}.data'.format(cnt)
    opt_history = pickle.load(open(data_file_name, 'rb'))

    # Setting up the environment.
    folder = Path('draw_optimization')
    create_folder(folder, exist_ok=True)
    seed = 42
    env = AmplifierEnv2d(seed, folder)

    def draw_design(design_params, file_name):
        _, info = env.solve(design_params, False, { 'solver': 'eigen' })
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

        ax.add_collection(mc.LineCollection(interfaces, colors='k', linewidth=4.0))
        ax.add_collection(mc.LineCollection(thin_lines, colors='k', linewidth=1.0))
        ax.add_collection(mc.LineCollection(lines, colors='k', linewidth=4.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-padding, cell_nums[0] + padding])
        ax.set_ylim([-padding, cell_nums[1] + padding])
        ax.axis('off')

        fig.savefig(folder / file_name)
        plt.close()

    # Draw init guesses.
    sample_num = 4
    for i in range(sample_num):
        theta = np.random.uniform(env.lower_bound(), env.upper_bound())
        draw_design(theta, 'init_{:04d}.png'.format(i))

    # Render results.
    # 000k.png renders opt_history[k], which is also the last element in 000k.data.
    fps = 10
    cnt = len(opt_history)
    for k in range(cnt - 1):
        xk0, _, _ = opt_history[k]
        xk1, _, _ = opt_history[k + 1]
        for i in range(fps):
            t = i / fps
            xk = (1 - t) * xk0 + t * xk1
            draw_design(xk, '{:04d}.png'.format(k * fps + i))

    # Render the optimal design.
    def draw_simulation(design_params, file_name):
        _, info = env.solve(design_params, False, { 'solver': 'eigen' })
        u = info['velocity_field']
        node_nums = env.node_nums()
        sdf = np.zeros(node_nums)
        for i in range(node_nums[0]):
            for j in range(node_nums[1]):
                sdf[i, j] = info['scene'].GetSignedDistance((i, j))
                if sdf[i, j] >= 0:
                    u[i, j] = 0

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

        fig.savefig(folder / file_name)
        plt.close()

    # Render simulation results.
    draw_simulation(opt_history[-1][0], 'simulation.png')