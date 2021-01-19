import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

class AmplifierEnv2d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (64, 48)
        E = 100
        nu = 0.499
        vol_tol = 1e-3
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 8), ('bezier', 8) ]
        # Initialize the node conditions.
        self._node_boundary_info = []
        inlet_velocity = 1
        inlet_range = ndarray([0.125, 0.875])
        inlet_lb, inlet_ub = inlet_range * cell_nums[1]
        for j in range(cell_nums[1] + 1):
            if inlet_lb < j < inlet_ub:
                self._node_boundary_info.append(((0, j, 0), inlet_velocity))
                self._node_boundary_info.append(((0, j, 1), 0))
        # Initialize the interface.
        self._interface_boundary_type = 'free-slip'

        # Other data members.
        self._inlet_velocity = inlet_velocity
        self._outlet_velocity = 3 * inlet_velocity
        self._inlet_range = inlet_range

    def _variables_to_shape_params(self, x):
        x = ndarray(x).copy().ravel()
        assert x.size == 5

        cx, cy = self._cell_nums
        # Convert x to the shape parameters.
        lower = ndarray([
            [1, x[4]],
            x[2:4],
            x[:2],
            [0, self._inlet_range[0]],
        ])
        lower[:, 0] *= cx
        lower[:, 1] *= cy
        upper = ndarray([
            [0, self._inlet_range[1]],
            [x[0], 1 - x[1]],
            [x[2], 1 - x[3]],
            [1, 1 - x[4]],
        ])
        upper[:, 0] *= cx
        upper[:, 1] *= cy
        params = np.concatenate([lower.ravel(), upper.ravel()])

        # Jacobian.
        J = np.zeros((params.size, x.size))
        J[1, 4] = 1
        J[2, 2] = 1
        J[3, 3] = 1
        J[4, 0] = 1
        J[5, 1] = 1
        J[10, 0] = 1
        J[11, 1] = -1
        J[12, 2] = 1
        J[13, 3] = -1
        J[15, 4] = -1
        # Scale it by cx and cy.
        J[:, 0] *= cx
        J[:, 2] *= cx
        J[:, 1] *= cy
        J[:, 3] *= cy
        J[:, 4] *= cy
        return ndarray(params).copy(), ndarray(J).copy()

    def _loss_and_grad(self, scene, u):
        param_size = self._variables_to_shape_params(self.lower_bound())[0].size
        grad_param = ndarray(np.zeros(param_size))

        u_field = self.reshape_velocity_field(u)
        grad = np.zeros(u_field.shape)
        cnt = 0
        loss = 0
        for j, (ux, uy) in enumerate(u_field[-1]):
            if ux > 0:
                cnt += 1
                u_diff = ndarray([ux, uy]) - ndarray([self._outlet_velocity, 0])
                loss += u_diff.dot(u_diff)
                grad[-1, j] += 2 * u_diff
        loss /= cnt
        grad /= cnt
        return loss, ndarray(grad).ravel(), grad_param

    def sample(self):
        return np.random.uniform(low=self.lower_bound(), high=self.upper_bound())

    def lower_bound(self):
        return ndarray([0.16, 0.05, 0.49, 0.05, 0.05])

    def upper_bound(self):
        return ndarray([0.49, 0.49, 0.83, 0.49, 0.49])