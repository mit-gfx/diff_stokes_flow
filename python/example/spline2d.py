import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import Spline2d, StdIntArray2d
from py_diff_stokes_flow.common.common import ndarray, create_folder, print_error

def visualize_level_set(ls):
    _, ax = plt.subplots()
    nx = ls.node_num(0)
    ny = ls.node_num(1)
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx))
    Z = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            Z[i, j] = ls.signed_distance(StdIntArray2d((i, j)))
    cs = ax.contour(X, Y, Z, 20)
    ax.clabel(cs, fontsize=10, inline=1)
    ax.set_aspect('equal')
    ax.grid(True)
    plt.show()
    plt.close()

def test_spline2d(verbose):
    np.random.seed(42)
    folder = Path('spline2d')

    cell_nums = StdIntArray2d((32, 24))
    control_points = ndarray([
        [32, 12],
        [22, 6],
        [12, 18],
        [0, 12]
    ])
    control_points[:, 1] += np.random.normal(size=4)
    spline = Spline2d()
    spline.Initialize(cell_nums, control_points.ravel())
    sdf = ndarray(spline.signed_distances())
    sdf_master = np.load(folder / 'sdf_master.npy')
    if np.max(np.abs(sdf - sdf_master)) > 0:
        if verbose:
            print_error('Incorrect signed distance function.')
        return False

    if verbose:
        visualize_level_set(spline)

    return True

if __name__ == '__main__':
    verbose = True
    test_spline2d(verbose)