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
    folder = Path('draw_design')
    create_folder(folder, exist_ok=True)
    seed = 42
    env = AmplifierEnv2d(seed, folder)

    create_folder(folder / 'init_design', exist_ok=True)
    def draw_init_design(design_params, file_name, draw_control_points=False):
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

        if draw_control_points:
            shape_params, _ = env._variables_to_shape_params(design_params)
            shape_params = shape_params.reshape((8, 2))
            control_lines = []
            for i in range(3):
                v0 = shape_params[i]
                v1 = shape_params[i + 1]
                control_lines.append((v0, v1))
                v0 = shape_params[4 + i]
                v1 = shape_params[4 + i + 1]
                control_lines.append((v0, v1))
            ax.add_collection(mc.LineCollection(control_lines, colors='tab:orange', linestyles='-.', linewidth=2.0))

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

    # Draw the signed distance.
    create_folder(folder / 'signed_dist', exist_ok=True)
    def draw_signed_distance(design_params, file_name):
        _, info = env.solve(design_params, False, { 'solver': 'eigen' })
        node_nums = env.node_nums()
        sdf = np.zeros(node_nums)
        for i in range(node_nums[0]):
            for j in range(node_nums[1]):
                sdf[i, j] = info['scene'].GetSignedDistance((i, j))

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
        # Draw horizontal boundaries.
        cell_nums = [n - 1 for n in node_nums]
        nx, ny = node_nums
        Y, X = np.meshgrid(np.arange(ny), np.arange(nx))
        Z = np.zeros((nx, ny))
        cs = ax.contour(X, Y, sdf, 20)
        ax.clabel(cs, fontsize=10, inline=1)
        ax.set_aspect('equal')
        ax.grid(True)

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
    sample_num = 8
    theta = [np.random.uniform(env.lower_bound(), env.upper_bound()) for _ in range(sample_num)]
    fps = 10
    for k in range(sample_num - 1):
        xk0 = theta[k]
        xk1 = theta[k + 1]
        for i in range(fps):
            t = i / fps
            xk = (1 - t) * xk0 + t * xk1
            draw_init_design(xk, 'init_design/{:04d}.png'.format(k * fps + i), draw_control_points=True)
            if k == 0 and i == 0:
                draw_init_design(xk, '{:04d}.png'.format(k * fps + i), draw_control_points=False)
            draw_signed_distance(xk, 'signed_dist/{:04d}.png'.format(k * fps + i))