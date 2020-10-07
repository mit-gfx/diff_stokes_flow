cd external/
mkdir -p pbrt_build
cd pbrt_build
cmake ../pbrt-v3
make -j4
cd ../../

# Generate python binding.
cd cpp/core/src
swig -c++ -python py_diff_stokes_flow_core.i

# Compile c++ code.
cd ../../
mkdir -p build
cd build
if [ $# -eq 0 ]
then
    cmake -DPARDISO_AVAILABLE=OFF ..
else
    cmake -DPARDISO_AVAILABLE=ON ..
fi
make -j4
./diff_stokes_flow_demo

# Python binding.
cd ../core/src/
mv py_diff_stokes_flow_core.py ../../../python/py_diff_stokes_flow/core
mv ../../build/libpy_diff_stokes_flow_core.so ../../../python/py_diff_stokes_flow/core/_py_diff_stokes_flow_core.so

# Log absolute path.
cd ../../../
root_path=$(pwd)
printf "root_path = '%s'\n" "$root_path" > python/py_diff_stokes_flow/common/project_path.py