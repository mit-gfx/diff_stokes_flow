import time
import numpy as np
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import StdRealVector, Scene2d, Scene3d
from py_diff_stokes_flow.common.common import ndarray, create_folder

# For each derived class, it only needs to initialize three data members in __init__ and overlead the loss and grad
# function _loss_and_grad_on_velocity_field. A typical derived may look like this:
#
# class EnvDerived:
#     def __init__(self, ...):
#         EnvBase.__init__(self, ...)
#
#         self._parametric_shape_info = {}
#         self._node_boundary_info = []
#         self._interface_boundary_type = 'free-slip'
#
#     def _loss_and_grad_on_velocity_field(self, u):
#         u_tensor = self.reshape_velocity_field(u)
#         loss = ...
#         grad = ...
#         return loss, grad
#
#     def sample(self):
#         return np.random.uniform(self.lower_bound(), self.upper_bound())
#
#     def lower_bound(self):
#         return ndarray([-np.inf, -np.inf, ...])
#
#     def upper_bound(self):
#         return ndarray([np.inf, np.inf, ...])
#
# The derived class is typically used as follows:
# env = EnvDerived(...)
# loss, grad, info = env.solve(x)
# scene = info['scene']

class EnvBase:
    def __init__(self,
        cell_nums,          # 2D or 3D int array.
        E,                  # Young's modulus.
        nu,                 # Poisson's ratio.
        vol_tol,            # vol_tol <= mixed cell <= 1 - vol_tol.
        edge_sample_num,    # The number of samples inside each cell's axis.
        folder              # The folder that stores temporary files.
    ):
        # The following data members are provided:
        # - _cell_nums and _node_nums;
        # - _dim;
        # - _cell_options;
        # - _folder;
        # - _parametric_shape_info: a list of (string, int);
        # - _node_boundary_info: a list of (int, float) or ((int, int, int), float) (2D) or
        #   ((int, int, int, int), float) (3D) that describes the boundary conditions.
        # - _interface_boundary_type: either 'no-slip' or 'free-slip' (default).

        # Sanity check the inputs.
        cell_nums = ndarray(cell_nums).astype(np.int32)
        assert cell_nums.size in [2, 3]

        # The value of E does not quite matter as it simply scale the QP problem by a constant factor.
        E = float(E)
        assert E > 0

        nu = float(nu)
        assert 0 < nu < 0.5

        vol_tol = float(vol_tol)
        assert 0 < vol_tol < 0.5

        edge_sample_num = int(edge_sample_num)
        assert edge_sample_num > 1

        if folder is not None:
            folder = Path(folder)
            create_folder(folder, exist_ok=False)

        # Create data members.
        self._cell_nums = np.copy(cell_nums)
        self._node_nums = ndarray([n + 1 for n in cell_nums]).astype(np.int32)
        self._dim = self._cell_nums.size
        self._cell_options = {
            'E': E, 'nu': nu, 'vol_tol': vol_tol, 'edge_sample_num': edge_sample_num
        }
        self._folder = folder

        # The remaining initialization is left to the subclass:
        # - Initialize the parametric shapes.
        # - Initialize the boundary conditions.
        # Each derived class needs to fill in the following data members:
        self._parametric_shape_info = []
        self._node_boundary_info = []
        self._interface_boundary_type = 'free-slip'

    # Each scene must present a loss and grad function defined on a velocity field.
    # Input:
    # - u: a 1D array that stores the whole velocity.
    # Output:
    # - loss: a scalar (float).
    # - grad: a 1D array that stores grad loss/grad u.
    def _loss_and_grad_on_velocity_field(self, u):
        raise NotImplementedError

    # If u is an 1D array, reshape it to [*node_nums] x dim.
    # If it is a tensor of size [*node_nums] x dim, flatten it as a 1D array.
    def reshape_velocity_field(self, u):
        u_array = ndarray(u)
        assert len(u_array.shape) in [1, self._dim + 1]
        assert u_array.size == np.prod(self._node_nums) * self._dim
        if len(u_array.shape) == 1:
            return np.copy(u_array).reshape(tuple(self._node_nums) + (self._dim,))
        else:
            return np.copy(u_array).ravel()

    # The core of this class.
    # - x: shape parameters flattened into a 1D array.
    # - options:
    #   - 'solver': 'eigen' or 'pardiso'.
    def solve(self, x, options):
        scene = Scene2d() if self._dim == 2 else Scene3d()

        # Initialize shapes.
        assert self._parametric_shape_info
        x = ndarray(x).ravel()
        assert x.size == self.parameter_dim()
        cnt = 0
        names = []
        vals = []
        for k, v in self._parametric_shape_info:
            xk = x[cnt:cnt + v]
            cnt += v
            names.append(k)
            vals.append(xk)
        
        scene.InitializeShapeComposition([int(n) for n in self._cell_nums], names, vals)

        # Initialize cells.
        scene.InitializeCell(
            self._cell_options['E'],
            self._cell_options['nu'],
            self._cell_options['vol_tol'],
            self._cell_options['edge_sample_num']
        )

        # Initialize the Dirichlet boundary conditions.
        node_bnd_dict = {}
        for dof, val in self._node_boundary_info:
            dof = scene.GetNodeDof(dof[:-1], dof[-1])
            node_bnd_dict[dof] = val
        node_bnd_dofs = []
        node_bnd_vals = []
        for dof, val in node_bnd_dict.items():
            dof = int(dof)
            assert 0 <= dof < np.prod(self._node_nums)
            val = float(val)
            node_bnd_dofs.append(dof)
            node_bnd_vals.append(val)
        scene.InitializeDirichletBoundaryCondition(node_bnd_dofs, node_bnd_vals)

        # Initialize the interface boundary type.
        name_map = {
            'no-slip': 'no_slip',
            'free-slip': 'no_separation'
        }
        assert self._interface_boundary_type in name_map
        scene.InitializeBoundaryType(name_map[self._interface_boundary_type])

        # Solve for the velocity.
        u = StdRealVector(0)
        dl_dparam = StdRealVector(0)
        assert 'solver' in options and options['solver'] in ['eigen', 'pardiso']

        # Forward simulation to obtain the velocity field.
        solver = options['solver']
        x = scene.Forward(solver)
        u = ndarray(scene.GetVelocityFieldFromForward(x))

        # Backpropagation.
        loss, grad = self._loss_and_grad_on_velocity_field(u)
        grad = ndarray(scene.Backward(solver, x, grad))
        return loss, grad, { 'scene': scene, 'u': u }

    # Lower bounds, upper bounds, and initial configuration.
    def sample(self):
        raise NotImplementedError

    def lower_bound(self):
        raise NotImplementedError

    def upper_bound(self):
        raise NotImplementedError

    def parameter_dim(self):
        return np.sum([v for _, v in self._parametric_shape_info])

    def cell_nums(self):
        return np.copy(self._cell_nums)

    def node_nums(self):
        return np.copy(self._node_nums)

    def dim(self):
        return self._dim

    def cell_options(self):
        return self._cell_options

    def folder(self):
        return self._folder

    def parametric_shape_info(self):
        return self._parametric_shape_info

    def node_boundary_info(self):
        return self._node_boundary_info

    def interface_boundary_type(self):
        return self._interface_boundary_type