import sys
sys.path.append('../')

import numpy as np
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import Cell2d
from py_diff_stokes_flow.common.common import ndarray, print_error

def test_cell_2d(verbose):
    np.random.seed(42)

    cell = Cell2d()
    E = 1e5
    nu = 0.45
    threshold = 1e-3
    edge_sample_num = 2
    # Consider a line that passes (0.5, 0.5) with a slope between 0 and 1.
    p = ndarray([0.5, 0.5])
    k = np.random.uniform(low=0, high=1)
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
    area_00 = 0.5 * k * 0.5 / 2
    area_01 = 0.5 * 0.5
    area_10 = 0
    area_11 = 0.5 * 0.5 - 0.5 * k * 0.5 / 2
    area = ndarray([area_00, area_01, area_10, area_11])
    area_from_cell = ndarray(cell.sample_areas())
    if not np.allclose(area, area_from_cell):
        if verbose:
            print_error('area is inconsistent.')
        return False

    # Check if all line segments are correct.
    line_00 = np.sqrt(1 + k ** 2) * 0.5
    line_01 = 0
    line_10 = 0
    line_11 = line_00
    line = ndarray([line_00, line_01, line_10, line_11])
    line_from_cell = ndarray(cell.sample_boundary_areas())
    if not np.allclose(line, line_from_cell):
        if verbose:
            print_error('boundary area is inconsistent.')
        return False

if __name__ == '__main__':
    verbose = True
    test_cell_2d(verbose)