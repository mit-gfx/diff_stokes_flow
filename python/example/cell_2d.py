import sys
sys.path.append('../')

import numpy as np
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import Cell2d
from py_diff_stokes_flow.common.common import ndarray, print_error, print_info
from py_diff_stokes_flow.common.grad_check import check_gradients

def test_cell_2d(verbose):
    np.random.seed(42)

    cell = Cell2d()
    E = 1e5
    nu = 0.45
    threshold = 1e-3
    edge_sample_num = 3
    # Consider a line that passes (0.5, 0.5) with a slope between 1/3 and 1.
    p = ndarray([0.5, 0.5])
    k = np.random.uniform(low=1 / 3, high=1)
    # Line equation: (y - p[1]) / (x - p[0]) = k.
    # y - p[1] = kx - kp[0].
    # kx - y + p[1] - kp[0].
    line_eq = ndarray([k, -1, p[1] - k * p[0]])
    # Solid area: line_eq >= 0.
    # So, the lower part is the solid area.
    # This means corner distance from [0, 0] and [1, 0] are positive.
    sdf_at_corners = []
    for c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        sdf_at_corners.append((line_eq[0] * c[0] + line_eq[1] * c[1] + line_eq[2]) / np.linalg.norm(line_eq[:2]))
    cell.Initialize(E, nu, threshold, edge_sample_num, sdf_at_corners)

    # Check if all areas are correct.
    dx = 1 / 3
    x_intercept = (-line_eq[1] * dx - line_eq[2]) / line_eq[0]
    area_00 = x_intercept * x_intercept * k * 0.5
    area_01 = dx ** 2 - (dx - x_intercept) ** 2 * k * 0.5
    area_02 = dx ** 2
    area_10 = 0
    area_11 = dx ** 2 * 0.5
    area_12 = dx ** 2
    area_20 = 0
    area_21 = dx ** 2 - area_01
    area_22 = dx ** 2 - area_00
    area = ndarray([area_00, area_01, area_02, area_10, area_11, area_12, area_20, area_21, area_22])
    area_from_cell = ndarray(cell.sample_areas())
    if not np.allclose(area, area_from_cell):
        if verbose:
            print_error('area is inconsistent.')
        return False

    # Check if all line segments are correct.
    line_00 = np.sqrt(1 + k ** 2) * x_intercept
    line_01 = np.sqrt(1 + k ** 2) * (dx - x_intercept)
    line_02 = 0
    line_10 = 0
    line_11 = np.sqrt(1 + k ** 2) * dx
    line_12 = 0
    line_20 = 0
    line_21 = line_01
    line_22 = line_00
    line = ndarray([line_00, line_01, line_02, line_10, line_11, line_12, line_20, line_21, line_22])
    line_from_cell = ndarray(cell.sample_boundary_areas())
    if not np.allclose(line, line_from_cell):
        if verbose:
            print_error('boundary area is inconsistent.')
        return False

    # Test the gradients.
    for loss_func, grad_func, name in [
        (cell.py_normal, cell.py_normal_gradient, 'normal'),
        (cell.offset, cell.py_offset_gradient, 'offset'),
        (cell.sample_areas, cell.py_sample_areas_gradient, 'sample_areas'),
        (cell.sample_boundary_areas, cell.py_sample_boundary_areas_gradient, 'sample_boundary_areas'),
        (cell.area, cell.py_area_gradient, 'area'),
        (cell.py_energy_matrix, cell.py_energy_matrix_gradient, 'energy_matrix'),
        (cell.py_dirichlet_vector, cell.py_dirichlet_vector_gradient, 'dirichlet_vector')
    ]:
        if verbose:
            print_info('Checking loss and gradient:', name)
        dim = ndarray(loss_func()).size
        weight = np.random.normal(size=dim)
        def loss_and_grad(x):
            cell.Initialize(E, nu, threshold, edge_sample_num, x)
            loss = ndarray(loss_func()).ravel().dot(weight)
            grad = np.zeros(4)
            for i in range(4):
                grad[i] = ndarray(grad_func(i)).ravel().dot(weight)
            return loss, grad
        if not check_gradients(loss_and_grad, ndarray(sdf_at_corners), verbose=verbose):
            if verbose:
                print_error('Gradient check failed.')
            return False

    return True

if __name__ == '__main__':
    verbose = True
    test_cell_2d(verbose)