# Functional Optimization of Fluidic Devices with Differentiable Stokes Flow (Siggraph Asia 2020)

[![Travis CI Status](https://travis-ci.org/mit-gfx/diff_stokes_flow.svg?branch=master)](https://travis-ci.org/mit-gfx/diff_stokes_flow)

## System requirement
- Ubuntu 18.04
- (Mini)conda 4.7.12 or higher

## Installation
```
git clone --recursive https://github.com/mit-gfx/diff_stokes_flow.git
cd diff_stokes_flow
conda env create -f environment.yml
conda activate diff_stokes_flow
./install.sh
```
### (Optional) Configuring Pardiso
Let `<PARDISO_HOME>` be the folder that you saved your Pardiso license file and the binary file. For example, if `/home/pardiso/pardiso.lic` and `/home/pardiso/libpardiso600-GNU720-X86-64.so` are your license and binary files, then `<PARDISO_HOME>=/home/pardiso`.
- Set `PARDISO_LIC_PATH` and `OMP_NUM_THREADS`:
```
export OMP_NUM_THREADS=4
export PARDISO_LIC_PATH=<PARDISO_HOME>
export PARDISOLICMESSAGE=1
```
- Pardiso requires `lapack` and `blas`:
```
sudo apt-get install liblapack-dev
sudo apt-get install libblas-dev
```
As of the date this README is written, the version we use is `3.7.1-4ubuntu1`:
```
Reading package lists... Done
Building dependency tree
Reading state information... Done
libblas-dev is already the newest version (3.7.1-4ubuntu1).
liblapack-dev is already the newest version (3.7.1-4ubuntu1).
0 upgraded, 0 newly installed, 0 to remove and 132 not upgraded.
```
- Recompile the codebase with an optional `pardiso` argument:
```
./install.sh pardiso
```

## Examples
Navigate to the `python/example` folder before you run any of the following scripts.

### Results in the paper
Run `python run_demo.py [demo_name]` where `[demo_name]` can be any of the following;
- `amplifier`: run the `Amplifier` demo (Fig. 2) in the paper.
- `flow_averager`: run the `Flow Averager` demo (Fig. 4) in the paper.

<img src="asset/video/flow_averager.gif" alt="flow_averager" style="display:block;margin-left:auto;margin-right:auto;width:50%;">

- `superposition_gate`: run the `Superposition Gate` demo (Fig. 4) in the paper.

<img src="asset/video/superposition_gate.gif" alt="superposition_gate" style="display:block;margin-left:auto;margin-right:auto;width:50%;">

- `funnel`: run the `Funnel` demo (Fig. 4) in the paper.

<img src="asset/video/funnel.gif" alt="funnel" style="display:block;margin-left:auto;margin-right:auto;width:50%;">

- `fluidic_twister`: run the `Fluidic Twister` demo (Fig. 5) in the paper. Note that the original result was run with a resolution of 64 by 64 by 32, and we downsampled it in this codebase for a faster demonstration.

<img src="asset/video/fluidic_twister.gif" alt="fluidic_twister" style="display:block;margin-left:auto;margin-right:auto;width:50%;">

- `fluidic_switch`: run the `Fluidic Switch` demo (Fig. 1) in the paper. Note that the original result was run with a resolution of 64 by 64 by 32, and we downsampled it in this codebase for a faster demonstration. The two figures below show the optimized solution in two modes:

<img src="asset/image/fluidic_switch_mode_0.png" alt="fluidic_switch_mode_0" style="display:block;margin-left:auto;margin-right:auto;width:50%;">

<img src="asset/image/fluidic_switch_mode_1.png" alt="fluidic_switch_mode_1" style="display:block;margin-left:auto;margin-right:auto;width:50%;">

### Numerical tests
Run `python [script_name].py` where `[script_name]` can be any of the following:
- `bezier_2d`: show the level-set of a Bezier curve and check the implementation of gradients.
- `cell_2d`: check if all quantities in `Cell2d` are implemented correctly.
- `scene_2d`: check the gradients of loss defined in a 2-dimensional scene.
- `shape_composition_2d` and `shape_composition_3d`: check the gradients of composing multiple primitive level-sets.
Finally, if you would like to run all these numerical experiments above, you can simply call `run_all_tests`:
- `run_all_tests`: this will sequentially run all the aforementioned numerical tests.

Run `python refinement.py` to generate the numerical results in Fig. 8 in the paper.

### Rendering
Run `python pbrt_renderer_demo.py` to see how to use the Python wrapper of pbrt-v3.

Additionally, we have also provided a few Python scripts for generating figures in the presentation:
- `python draw_pipeline.py`: generate each subfigure in the whole pipeline.
- `python draw_design.py`: generate pictures for the design section.
- `python draw_discretization.py`: generate pictures for the discretization section.
- `python draw_optimization.py`: generate video and pictures for the optimization section.

## How to cite
```text
@article{du2020stokes,
    author = {Du, Tao and Wu, Kui and Spielberg, Andrew and Matusik, Wojciech and Zhu, Bo and Sifakis, Eftychios},
    title = {Functional Optimization of Fluidic Devices with Differentiable Stokes Flow},
    year = {2020},
    issue_date = {December 2020},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {39},
    number = {6},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3414685.3417795},
    doi = {10.1145/3414685.3417795},
    journal = {ACM Trans. Graph.},
    month = dec,
    articleno = {197},
    numpages = {15},
    keywords = {physically-based simulation, fluid simulation, computational design optimization}
}
```

## Contact
If you have any questions about the paper or the codebase, please feel free to contact taodu@csail.mit.edu.
