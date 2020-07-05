%module py_diff_stokes_flow_core
%{
#include "../include/shape/parametric_shape.h"
%}

%exception {
    try {
        $action
    } catch (const std::runtime_error& e) {
        PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
        SWIG_fail;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error.");
        SWIG_fail;
    }
}

%include <std_array.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_map.i>
%include "../include/common/config.h"
%include "../include/shape/parametric_shape.h"

namespace std {
    %template(StdRealArray2d) array<real, 2>;
    %template(StdRealArray3d) array<real, 3>;
    %template(StdIntArray4d) array<int, 4>;
    %template(StdIntArray8d) array<int, 8>;
    %template(StdRealVector) vector<real>;
    %template(StdIntVector) vector<int>;
    %template(StdRealMatrix) vector<vector<real>>;
    %template(StdMap) map<string, real>;
}

%template(ParametricShape2d) ParametricShape<2>;
%template(ParametricShape3d) ParametricShape<3>;