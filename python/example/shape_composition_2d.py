import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import ShapeComposition2d, StdIntArray2d, StdRealVector
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

def test_shape_composition_2d(verbose):
    np.random.seed(42)
    folder = Path('shape_composition_2d')

    cell_nums = (32, 24)
    control_points_lower = ndarray([
        [32, 10],
        [22, 10],
        [12, 4],
        [0, 4]
    ])
    control_points_upper = ndarray([
        [0, 20],
        [12, 20],
        [22, 14],
        [32, 14]
    ])

    shape = ShapeComposition2d()
    shape.AddParametricShape("spline", 8)
    shape.AddParametricShape("spline", 8)
    control_points = np.concatenate([control_points_lower.ravel(), control_points_upper.ravel()])
    control_points += np.random.normal(size=control_points.size) * 0.01
    shape.Initialize(cell_nums, control_points)
    sdf = ndarray(shape.signed_distances())
    sdf_master = np.load(folder / 'sdf_master.npy')
    if np.max(np.abs(sdf - sdf_master)) > 0:
        if verbose:
            print_error('Incorrect signed distance function.')
        return False

    if verbose:
        visualize_level_set(shape)

    # Verify the gradients.
    nx = shape.node_num(0)
    ny = shape.node_num(1)
    sdf_weight = np.random.normal(size=(nx, ny))
    def loss_and_grad(x):
        shape.Initialize(cell_nums, x)
        sdf = ndarray(shape.signed_distances())
        loss = sdf_weight.ravel().dot(sdf)
        grad = 0
        for i in range(nx):
            for j in range(ny):
                grad += sdf_weight[i, j] * ndarray(shape.signed_distance_gradient((i, j)))
        return loss, grad
    from py_diff_stokes_flow.common.grad_check import check_gradients
    return check_gradients(loss_and_grad, control_points.ravel(), verbose=verbose)

if __name__ == '__main__':
    verbose = True
    test_shape_composition_2d(verbose)