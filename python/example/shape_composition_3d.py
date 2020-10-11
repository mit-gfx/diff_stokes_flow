import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import ShapeComposition3d
from py_diff_stokes_flow.common.common import ndarray, create_folder, print_error

def test_shape_composition_3d(verbose):
    return test_shape_composition_3d_single([
        ( 'polar_bezier3',
            ndarray([
                1, 4, 1, 4, 1, 4, 1, 4,
                3, 3, 3, 3, 3, 3, 3, 3,
                4, 1, 4, 1, 4, 1, 4, 1,
                5, 5, np.pi / 3
            ])
        )
    ], verbose)

def test_shape_composition_3d_single(shape_info, verbose):
    np.random.seed(42)

    cell_nums = (10, 10, 3)
    shape = ShapeComposition3d()
    params = []
    for name, param in shape_info:
        shape.AddParametricShape(name, param.size)
        params.append(ndarray(param).ravel())
    params = np.concatenate(params)
    params += np.random.normal(size=params.size) * 0.01
    shape.Initialize(cell_nums, params)

    # Verify the gradients.
    nx = shape.node_num(0)
    ny = shape.node_num(1)
    nz = shape.node_num(2)
    shape.Initialize(cell_nums, params.ravel())
    sdf = ndarray(shape.signed_distances())
    sdf = sdf.reshape((nx, ny, nz))

    if verbose:
        # Visualize the mesh.
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        verts, faces, _, _ = measure.marching_cubes_lewiner(sdf, 0)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        cx, cy, cz = cell_nums
        ax.set_xlim(0, cx)
        ax.set_ylim(0, cy)
        ax.set_zlim(0, cz)

        plt.tight_layout()
        plt.show()

    sdf_weight = np.random.normal(size=(nx, ny, nz))
    def loss_and_grad(x):
        shape.Initialize(cell_nums, x)
        sdf = ndarray(shape.signed_distances())
        loss = sdf_weight.ravel().dot(sdf)
        grad = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    grad += sdf_weight[i, j, k] * ndarray(shape.signed_distance_gradients((i, j, k)))
        return loss, grad
    from py_diff_stokes_flow.common.grad_check import check_gradients
    return check_gradients(loss_and_grad, params.ravel(), verbose=verbose)

if __name__ == '__main__':
    verbose = True
    test_shape_composition_3d(verbose)