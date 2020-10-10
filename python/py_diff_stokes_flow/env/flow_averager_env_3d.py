import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

class FlowAveragerEnv3d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (64, 64, 4)
        E = 100
        nu = 0.499
        vol_tol = 1e-2
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 11), ('bezier', 11), ('bezier', 11), ('bezier', 11) ]
        # Initialize the node conditions.
        self._node_boundary_info = []

        inlet_range = ndarray([
            [0.1, 0.4],
            [0.6, 0.9],
        ])
        outlet_range = ndarray([
            [0.2, 0.4],
            [0.6, 0.8]
        ])
        cx, cy, _ = self.cell_nums()
        nx, ny, nz = self.node_nums()
        inlet_bd = inlet_range * cy
        outlet_bd = outlet_range * cy
        for j in range(ny):
            for k in range(nz):
                # Set the inlet at i = 0.
                if inlet_bd[0, 0] < j < inlet_bd[0, 1]:
                    self._node_boundary_info.append(((0, j, k, 0), 1))
                    self._node_boundary_info.append(((0, j, k, 1), 0))
                    self._node_boundary_info.append(((0, j, k, 2), 0))
                if inlet_bd[1, 0] < j < inlet_bd[1, 1]:
                    self._node_boundary_info.append(((0, j, k, 0), 0))
                    self._node_boundary_info.append(((0, j, k, 1), 0))
                    self._node_boundary_info.append(((0, j, k, 2), 0))
        # Set the top and bottom plane.
        for i in range(nx):
            for j in range(ny):
                for k in [0, nz - 1]:
                    self._node_boundary_info.append(((i, j, k, 2), 0))
        # Initialize the interface.
        self._interface_boundary_type = 'free-slip'

        # Other data members.
        self._inlet_range = inlet_range
        self._outlet_range = outlet_range
        self._inlet_bd = inlet_bd
        self._outlet_bd = outlet_bd

    def _variables_to_shape_params(self, x):
        x = ndarray(x).copy().ravel()
        assert x.size == 8

        cx, cy, _ = self._cell_nums
        lower = ndarray([
            [1, self._outlet_range[0, 0]],
            x[2:4],
            x[:2],
            [0, self._inlet_range[0, 0]],
        ])
        right = ndarray([
            [1, self._outlet_range[1, 0]],
            [x[4], 1 - x[5]],
            x[4:6],
            [1, self._outlet_range[0, 1]],
        ])
        upper = ndarray([
            [0, self._inlet_range[1, 1]],
            [x[0], 1 - x[1]],
            [x[2], 1 - x[3]],
            [1, self._outlet_range[1, 1]],
        ])
        left = ndarray([
            [0, self._inlet_range[0, 1]],
            x[6:8],
            [x[6], 1 - x[7]],
            [0, self._inlet_range[1, 0]],
        ])
        cxy = ndarray([cx, cy])
        lower *= cxy
        right *= cxy
        upper *= cxy
        left *= cxy
        params = np.concatenate([lower.ravel(),
            [0, -0.01, 1],
            right.ravel(),
            [0.01, 0, 1],
            upper.ravel(),
            [0, 0.01, 1],
            left.ravel(),
            [-0.01, 0, 1]
        ])

        # Jacobian.
        J = np.zeros((params.size, x.size))
        J[2, 2] = J[3, 3] = 1
        J[4, 0] = J[5, 1] = 1

        J[13, 4] = 1
        J[14, 5] = -1
        J[15, 4] = J[16, 5] = 1

        J[24, 0] = 1
        J[25, 1] = -1
        J[26, 2] = 1
        J[27, 3] = -1

        J[35, 6] = J[36, 7] = 1
        J[37, 6] = 1
        J[38, 7] = -1
        J[:, ::2] *= cx
        J[:, 1::2] *= cy
        return ndarray(params).copy(), ndarray(J).copy()

    def _loss_and_grad_on_velocity_field(self, u):
        u_field = self.reshape_velocity_field(u)
        grad = np.zeros(u_field.shape)
        nx, ny, nz = self.node_nums()
        loss = 0
        cnt = 0
        for j in range(ny):
            for k in range(nz):
                if self._outlet_bd[0, 0] < j < self._outlet_bd[0, 1] or \
                    self._outlet_bd[1, 0] < j < self._outlet_bd[1, 1]:
                    cnt += 1
                    u_diff = u_field[nx - 1, j, k] - ndarray([0.5, 0, 0])
                    loss += u_diff.dot(u_diff)
                    grad[nx - 1, j, k] += 2 * u_diff
        loss /= cnt
        grad /= cnt
        return loss, ndarray(grad).ravel()

    def _color_velocity(self, u):
        return float(np.linalg.norm(u) / 3)

    def sample(self):
        return np.random.uniform(low=self.lower_bound(), high=self.upper_bound())

    def lower_bound(self):
        return ndarray([.01, .01, .49, .01, .49, .01, .01, .01])

    def upper_bound(self):
        return ndarray([.49, .49, .99, .49, .99, .49, .49, .49])