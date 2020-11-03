import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray
from py_diff_stokes_flow.core.py_diff_stokes_flow_core import ShapeComposition2d, StdIntArray2d

class FluidicTwisterEnv3d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (16, 16, 8)
        E = 100
        nu = 0.499
        vol_tol = 1e-2
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('polar_bezier-6', 51)]
        # Initialize the node conditions.
        self._node_boundary_info = []

        inlet_radius = 0.3
        outlet_radius = 0.3
        inlet_velocity = 1.0
        outlet_velocity = 2.0
        cx, cy, _ = self.cell_nums()
        assert cx == cy
        nx, ny, nz = self.node_nums()

        def get_bezier(radius):
            bezier = ShapeComposition2d()
            params = np.concatenate([
                np.full(8, radius) * cx,
                ndarray([0.5 * cx, 0.5 * cy, 0])
            ])
            bezier.AddParametricShape('polar_bezier', params.size)
            cxy = StdIntArray2d((int(cx), int(cy)))
            bezier.Initialize(cxy, params, True)
            return bezier
        inlet_bezier = get_bezier(inlet_radius)
        outlet_bezier = get_bezier(outlet_radius)

        for i in range(nx):
            for j in range(ny):
                if inlet_bezier.signed_distance((i, j)) > 0:
                    self._node_boundary_info.append(((i, j, 0, 0), 0))
                    self._node_boundary_info.append(((i, j, 0, 1), 0))
                    self._node_boundary_info.append(((i, j, 0, 2), inlet_velocity))

        # Initialize the interface.
        self._interface_boundary_type = 'free-slip'

        # Compute the target velocity field (for rendering purposes only)
        desired_omega = 2 * outlet_velocity / (cx * outlet_radius)
        target_velocity_field = np.zeros((nx, ny, 3))
        for i in range(nx):
            for j in range(ny):
                if outlet_bezier.signed_distance((i, j)) > 0:
                    x, y = i / cx, j / cy
                    # u = (-(j - ny / 2), (i - nx / 2), 0) * c.
                    # ux_pos = (-j, i + 1, 0) * c.
                    # uy_pos = (-j - 1, i, 0) * c.
                    # curl = (i + 1) * c + (j + 1) * c - i * c - j * c.
                    #      = (i + j + 2 - i - j) * c = 2 * c.
                    # c = outlet_vel / (num_cells[0] * outlet_radius).
                    c = desired_omega / 2
                    target_velocity_field[i, j] = ndarray([
                        -(y - 0.5) * c,
                        (x - 0.5) * c,
                        0
                    ])

        # Other data members.
        self._inlet_radius = inlet_radius
        self._outlet_radius = outlet_radius
        self._inlet_velocity = inlet_velocity
        self._target_velocity_field = target_velocity_field
        self._inlet_bezier = inlet_bezier
        self._outlet_bezier = outlet_bezier
        self._desired_omega = desired_omega

    def _variables_to_shape_params(self, x):
        x = ndarray(x).copy().ravel()
        assert x.size == 32

        cx, cy, _ = self._cell_nums
        assert cx == cy
        params = np.concatenate([
            np.full(8, self._inlet_radius),
            x,
            np.full(8, self._outlet_radius),
            ndarray([0.5, 0.5, 0]),
        ])
        params[:-1] *= cx

        # Jacobian.
        J = np.zeros((params.size, x.size))
        for i in range(x.size):
            J[8 + i, i] = cx
        return ndarray(params).copy(), ndarray(J).copy()

    def _loss_and_grad_on_velocity_field(self, u):
        u_field = self.reshape_velocity_field(u)
        grad = np.zeros(u_field.shape)
        nx, ny, nz = self.node_nums()
        assert nx == ny
        loss = 0
        cnt = 0
        for i in range(nx):
            for j in range(ny):
                if self._outlet_bezier.signed_distance((i, j)) > 0:
                    cnt += 1
                    uxy = u_field[i, j, nz - 1, :2]
                    ux_pos = u_field[i + 1, j, nz - 1, :2]
                    uy_pos = u_field[i, j + 1, nz - 1, :2]
                    # Compute the curl.
                    curl = ux_pos[1] - uy_pos[0] - uxy[1] + uxy[0]
                    loss += (curl - self._desired_omega) ** 2
                    # ux_pos[1]
                    grad[i + 1, j, nz - 1, 1] += 2 * (curl - self._desired_omega)
                    grad[i, j + 1, nz - 1, 0] += -2 * (curl - self._desired_omega)
                    grad[i, j, nz - 1, 1] += -2 * (curl - self._desired_omega)
                    grad[i, j, nz - 1, 0] += 2 * (curl - self._desired_omega)
        loss /= cnt
        grad /= cnt
        return loss, ndarray(grad).ravel()

    def _color_velocity(self, u):
        return float(np.linalg.norm(u) / 2)

    def sample(self):
        return np.random.uniform(low=self.lower_bound(), high=self.upper_bound())

    def lower_bound(self):
        return np.full(32, 0.1)

    def upper_bound(self):
        return np.full(32, 0.4)