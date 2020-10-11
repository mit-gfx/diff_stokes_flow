import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import ShapeComposition2d, StdIntArray2d
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
    flag = test_shape_composition_2d_single([
        ( 'bezier',
            ndarray([
                [32, 10],
                [22, 10],
                [12, 4],
                [0, 4]
            ])
        ),
        ( 'bezier',
            ndarray([
                [0, 20],
                [12, 20],
                [22, 14],
                [32, 14]
            ])
        ),
    ], verbose)
    if not flag: return False

    flag = test_shape_composition_2d_single([
        ( 'plane', ndarray([1, 0.1, -16]) ),
        ( 'sphere', ndarray([16, 12, 8]) ),
    ], verbose)
    if not flag: return False

    flag = test_shape_composition_2d_single([
        ( 'polar_bezier',
            ndarray([
                4, 8, 4, 8, 4, 8, 4, 8,
                16, 12,
                np.pi / 3
            ])
        ),
    ], verbose)
    if not flag: return False

    return True

def test_shape_composition_2d_single(shape_info, verbose):
    np.random.seed(42)

    cell_nums = (32, 24)
    shape = ShapeComposition2d()
    params = []
    for name, param in shape_info:
        shape.AddParametricShape(name, param.size)
        params.append(ndarray(param).ravel())
    params = np.concatenate(params)
    params += np.random.normal(size=params.size) * 0.01
    shape.Initialize(cell_nums, params)

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
                grad += sdf_weight[i, j] * ndarray(shape.signed_distance_gradients((i, j)))
        return loss, grad
    from py_diff_stokes_flow.common.grad_check import check_gradients
    return check_gradients(loss_and_grad, params.ravel(), verbose=verbose)

if __name__ == '__main__':
    verbose = True
    test_shape_composition_2d(verbose)