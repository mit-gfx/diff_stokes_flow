import sys
sys.path.append('../')

import numpy as np
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import Scene2d, StdRealVector
from py_diff_stokes_flow.common.common import ndarray, print_error
from py_diff_stokes_flow.common.grad_check import check_gradients

def test_scene_2d(verbose):
    np.random.seed(42)

    cell_nums = (32, 24)
    control_points_lower = ndarray([
        [34, 10],
        [22, 10],
        [12, 10],
        [-2, 4]
    ]) + np.random.normal(size=(4, 2)) * 0.01
    control_points_upper = ndarray([
        [-2, 20],
        [12, 14],
        [22, 14],
        [34, 14]
    ]) + np.random.normal(size=(4, 2)) * 0.01
    u_weight = np.random.normal(size=(cell_nums[0] + 1, cell_nums[1] + 1, 2)).ravel()

    def loss_and_grad(x):
        scene = Scene2d()
        # Initialize shape.
        scene.InitializeShapeComposition(cell_nums, ['spline', 'spline'], [x[:8], x[8:]])

        # Initialize cell.
        E = 100
        nu = 0.499
        threshold = 1e-3
        edge_sample_num = 2
        scene.InitializeCell(E, nu, threshold, edge_sample_num)

        # Initialize Dirichlet boundary conditions.
        boundary_dofs = []
        boundary_values = []
        inlet_velocity = 1.0
        for j in range(cell_nums[1] + 1):
            if x[:8].reshape((4, 2))[3, 1] < j < x[8:].reshape((4, 2))[0, 1]:
                boundary_dofs.append(scene.GetNodeDof([0, j], 0))
                boundary_values.append(inlet_velocity)
                boundary_dofs.append(scene.GetNodeDof([0, j], 1))
                boundary_values.append(0.0)
        scene.InitializeDirichletBoundaryCondition(boundary_dofs, boundary_values)
        scene.InitializeBoundaryType('no_separation')

        # Solve for the velocity.
        u = StdRealVector(0)

        dl_dparam = StdRealVector(0)
        scene.Solve('eigen', u_weight, u, dl_dparam)
        u = ndarray(u)
        loss = u.dot(u_weight)
        grad = ndarray(dl_dparam)
        return loss, grad

    x0 = np.concatenate([control_points_lower.ravel(), control_points_upper.ravel()])
    if not check_gradients(loss_and_grad, x0, eps=1e-4, verbose=verbose):
        if verbose:
            print_error('Gradient check in scene_2d failed.')
        return False

    return True

if __name__ == '__main__':
    verbose = True
    test_scene_2d(verbose)