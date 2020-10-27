import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

class RefinementEnv2d(EnvBase):
    def __init__(self, nu, scale):
        cell_nums = (int(32 * scale), int(24 * scale))
        E = 100
        vol_tol = 1e-3
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, None)

        # Initial condition.
        control_points_lower = ndarray([
            [32, 10],
            [22, 10],
            [12, 10],
            [0, 4]
        ]) * scale
        control_points_upper = ndarray([
            [0, 20],
            [12, 14],
            [22, 14],
            [32, 14]
        ]) * scale
        self._sample = np.concatenate([control_points_lower.ravel(), control_points_upper.ravel()])

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 8), ('bezier', 8) ]
        # Initialize the node conditions.
        self._node_boundary_info = []
        inlet_velocity = 1.0
        for j in range(cell_nums[1] + 1):
            if control_points_lower[3, 1] < j < control_points_upper[0, 1]:
                self._node_boundary_info.append(((0, j, 0), inlet_velocity))
                self._node_boundary_info.append(((0, j, 1), 0))
        # Initialize the interface.
        self._interface_boundary_type = 'free-slip'

        self._scale = scale

    def _loss_and_grad_on_velocity_field(self, u):
        return 0, np.zeros(u.shape)

    def sample(self):
        return self._sample

    def lower_bound(self):
        return ndarray(np.full(self.parameter_dim(), -100))

    def upper_bound(self):
        return ndarray(np.full(self.parameter_dim(), 100))