import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

class RandomEnv2d(EnvBase):
    def __init__(self, seed):
        cell_nums = (32, 24)
        E = 100
        nu = 0.499
        vol_tol = 1e-3
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, None)

        np.random.seed(seed)
        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 8), ('bezier', 8) ]
        # Initialize the node conditions.
        self._node_boundary_info = []
        inlet_velocity = 1.0
        for j in range(cell_nums[1] + 1):
            if 4.1 < j < 19.9:
                self._node_boundary_info.append(((0, j, 0), inlet_velocity))
                self._node_boundary_info.append(((0, j, 1), 0))
        # Initialize the interface.
        self._interface_boundary_type = 'free-slip'

        self.__u_weight = np.random.normal(size=(cell_nums[0] + 1, cell_nums[1] + 1, 2)).ravel()

    def _loss_and_grad(self, scene, u):
        param_size = self._variables_to_shape_params(self.lower_bound())[0].size
        grad_param = ndarray(np.zeros(param_size))

        loss = self.__u_weight.dot(u)
        return loss, self.__u_weight, grad_param

    def sample(self):
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
        x0 = np.concatenate([control_points_lower.ravel(), control_points_upper.ravel()])
        return x0

    def lower_bound(self):
        return ndarray(np.full(self.parameter_dim(), -100))

    def upper_bound(self):
        return ndarray(np.full(self.parameter_dim(), 100))