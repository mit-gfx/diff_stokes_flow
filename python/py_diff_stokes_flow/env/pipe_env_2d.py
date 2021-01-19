import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray

class PipeEnv2d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (50, 50)
        E = 100
        nu = 0.499
        vol_tol = 1e-3
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 8), ('bezier', 8) ]
        # Initialize the node conditions.
        self._node_boundary_info = []
        cx, cy = cell_nums
        assert cx == cy
        nx, ny = cx + 1, cy + 1
        self._velocity_type = 'quadratic'   # 'quadratic' or 'const'.
        # Initialize the interface.
        self._interface_boundary_type = 'no-slip'   # Choose 'no-slip' or 'free-slip'.

        inlet_velocity = 1
        inlet_range = ndarray([0.7, 0.9])
        inlet_lb, inlet_ub = inlet_range * cy
        inlet_lb = int(np.floor(inlet_lb))
        inlet_ub = int(np.ceil(inlet_ub))
        outlet_velocity = 1
        outlet_range = ndarray([0.7, 0.9])
        outlet_lb, outlet_ub = outlet_range * cx
        outlet_lb = int(np.floor(outlet_lb))
        outlet_ub = int(np.ceil(outlet_ub))

        for j in range(cy + 1):
            if inlet_lb <= j <= inlet_ub:
                # Compute the quadratic profile.
                # j = inlet_lb: nj = -1.
                # j = inlet_ub: nj = 1.
                if self._velocity_type == 'const':
                    v = inlet_velocity
                elif self._velocity_type == 'quadratic':
                    nj = (j / cy - 0.8) / 0.1
                    v = (1 - nj ** 2) * inlet_velocity
                else:
                    raise NotImplementedError
                self._node_boundary_info.append(((0, j, 0), v))
                self._node_boundary_info.append(((0, j, 1), 0))

        # Other data members.
        self._inlet_velocity = inlet_velocity
        self._inlet_lb = inlet_lb
        self._inlet_ub = inlet_ub
        self._outlet_lb = outlet_lb
        self._outlet_ub = outlet_ub

    def _variables_to_shape_params(self, x):
        x = ndarray(x).copy().ravel()
        assert x.size == 2

        cx, cy = self._cell_nums
        assert cx == cy
        inlet_lb_ratio = (self._inlet_lb - 0.5) / cy
        inlet_ub_ratio = (self._inlet_ub + 0.5) / cy
        lower = ndarray([
            [inlet_lb_ratio, 0],
            [inlet_lb_ratio, x[0]],
            [x[0], inlet_lb_ratio],
            [0, inlet_lb_ratio]
        ])
        upper = ndarray([
            [0, inlet_ub_ratio],
            [x[1], inlet_ub_ratio],
            [inlet_ub_ratio, x[1]],
            [inlet_ub_ratio, 0]
        ])
        lower *= cx
        upper *= cx
        params = np.concatenate([lower.ravel(), upper.ravel()])

        # Jacobian.
        J = np.zeros((params.size, x.size))
        J[3, 0] = 1
        J[4, 0] = 1
        J[10, 1] = 1
        J[13, 1] = 1
        # Scale it by cx and cy.
        J *= cx
        return ndarray(params).copy(), ndarray(J).copy()

    def _loss_and_grad_on_velocity_field(self, u):
        u = ndarray(u).copy().ravel()
        u_field = self.reshape_velocity_field(u)
        grad = np.zeros(u_field.shape)
        cnt = 0
        loss = 0
        nx, _ = self._node_nums
        cx, cy = self._cell_nums
        for i in range(nx):
            if self._outlet_lb <= i <= self._outlet_ub:
                cnt += 1
                # Compute the quadratic profile.
                if self._velocity_type == 'const':
                    v = self._inlet_velocity
                elif self._velocity_type == 'quadratic':
                    nj = (i / cy - 0.8) / 0.1
                    v = (1 - nj ** 2) * self._inlet_velocity
                else:
                    raise NotImplementedError
                u_diff = ndarray(u_field[i, 0]) - ndarray([0, -v])
                loss += u_diff.dot(u_diff)
                grad[i, 0] += 2 * u_diff
        loss /= cnt
        grad /= cnt
        return loss, ndarray(grad).ravel()

    def sample(self):
        return np.random.uniform(low=self.lower_bound(), high=self.upper_bound())

    def lower_bound(self):
        return ndarray([0.01, 0.01])

    def upper_bound(self):
        return ndarray([0.69, 0.89])