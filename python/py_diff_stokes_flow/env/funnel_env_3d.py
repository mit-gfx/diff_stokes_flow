import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

class FunnelEnv3d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (64, 64, 4)
        E = 100
        nu = 0.499
        vol_tol = 1e-2
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 11), ('bezier', 11), ('bezier', 11) ]
        # Initialize the node conditions.
        self._node_boundary_info = []

        inlet_range = ndarray([0.4, 0.6])
        outlet_range = 0.8
        cx, cy, _ = self.cell_nums()
        assert cx == cy
        nx, ny, nz = self.node_nums()
        inlet_bd = inlet_range * cx
        outlet_bd = outlet_range * cx
        inlet_velocity = ndarray([1.0, 0.0])
        for j in range(ny):
            for k in range(nz):
                # Set the inlet at i = 0.
                if inlet_bd[0] < j < inlet_bd[1]:
                    self._node_boundary_info.append(((0, j, k, 0), inlet_velocity[0]))
                    self._node_boundary_info.append(((0, j, k, 1), 0))
                    self._node_boundary_info.append(((0, j, k, 2), 0))

                    self._node_boundary_info.append(((j, 0, k, 0), inlet_velocity[1]))
                    self._node_boundary_info.append(((j, 0, k, 1), 0))
                    self._node_boundary_info.append(((j, 0, k, 2), 0))
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
        self._inlet_velocity = inlet_velocity

    def _variables_to_shape_params(self, x):
        x = ndarray(x).copy().ravel()
        assert x.size == 5

        cx, cy, _ = self._cell_nums
        assert cx == cy
        lower_left = ndarray([
            [self._inlet_range[0], 0],
            [x[4], x[0]],
            [x[0], x[4]],
            [0, self._inlet_range[0]]
        ])
        right = ndarray([
            [1., self._outlet_range],
            [x[2], x[3]],
            [self._inlet_range[1], x[1]],
            [self._inlet_range[1], 0]
        ])
        upper = ndarray([
            [0, self._inlet_range[1]],
            [x[1], self._inlet_range[1]],
            [x[3], x[2]],
            [self._outlet_range, 1.]
        ])
        lower_left *= cx
        right *= cx
        upper *= cx

        params = np.concatenate([lower_left.ravel(),
            [-0.01, -0.01, 1],
            right.ravel(),
            [0.01, 0, 1],
            upper.ravel(),
            [0, 0.01, 1],
        ])

        # Jacobian.
        J = np.zeros((params.size, x.size))
        J[2, 4] = J[3, 0] = 1
        J[4, 0] = J[5, 4] = 1

        J[13, 2] = J[14, 3] = 1
        J[16, 1] = 1

        J[24, 1] = J[26, 3] = J[27, 2] = 1
        J *= cx
        return ndarray(params).copy(), ndarray(J).copy()

    def _loss_and_grad_on_velocity_field(self, u):
        u_field = self.reshape_velocity_field(u)
        grad = np.zeros(u_field.shape)
        nx, ny, nz = self.node_nums()
        assert nx == ny
        loss = 0
        cnt = 0
        outlet_bd = int(np.ceil(self._outlet_range * nx))
        eps = 1e-8
        outlet_idx = []
        for j in range(outlet_bd, ny):
            for k in range(nz):
                outlet_idx.append((nx - 1, j, k))
        for i in range(outlet_bd, nx):
            for k in range(nz):
                outlet_idx.append((i, ny - 1, k))
        for i, j, k in outlet_idx:
            cnt += 1
            ux, uy, _ = u_field[i, j, k]
            angle = np.arctan2(uy, ux)
            loss += np.abs(angle - np.pi / 4)
            dangle_ux = -uy / (ux ** 2 + uy ** 2 + eps)
            dangle_uy = ux / (ux ** 2 + uy ** 2 + eps)
            grad[i, j, k] += ndarray([
                np.sign(angle - np.pi / 4) * dangle_ux,
                np.sign(angle - np.pi / 4) * dangle_uy,
                0
            ])
        loss /= cnt
        grad /= cnt
        return loss, ndarray(grad).ravel()

    def _color_velocity(self, u):
        return float(np.clip(np.arctan2(u[1], u[0]), 0, np.pi / 2))

    def sample(self):
        return np.random.uniform(low=self.lower_bound(), high=self.upper_bound())

    def lower_bound(self):
        return ndarray([.01, .01, .59, .01, .01])

    def upper_bound(self):
        return ndarray([.39, .59, .99, .99, .59])