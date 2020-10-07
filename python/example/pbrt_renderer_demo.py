import sys
sys.path.append('../')

from pathlib import Path
import shutil
import os
import numpy as np

from py_diff_stokes_flow.common.renderer import PbrtRenderer
from py_diff_stokes_flow.common.common import print_info, create_folder
from py_diff_stokes_flow.common.project_path import root_path

if __name__ == '__main__':
    folder = Path('pbrt_renderer_demo')
    create_folder(folder)

    # Render.
    options = {
        'file_name': str(folder / 'demo.png'),
        'light_map': 'uffizi-large.exr',
        'sample': 16,
        'max_depth': 4,
        'camera_pos': (0, -2, 0.8),
        'camera_lookat': (0, 0, 0),
    }
    renderer = PbrtRenderer(options)
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
        texture_img='chkbd_24_0.7')
    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/bunny.obj',
        transforms=[
            ('s', 0.4),
        ],
    color=(.2, .7, .3))
    renderer.add_shape_mesh({ 'name': 'sphere', 'radius': 0.1, 'center': [0.2, -0.4, 0.1] }, color=(.7, .3, .2))
    renderer.render(verbose=True)

    # Display.
    os.system('eog {}'.format(folder / 'demo.png'))