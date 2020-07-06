%module py_diff_stokes_flow_core
%{
#include "../include/shape/parametric_shape.h"
#include "../include/shape/spline2d.h"
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
    %template(StdIntArray2d) array<int, 2>;
    %template(StdIntArray3d) array<int, 3>;
    %template(StdRealArray2d) array<real, 2>;
    %template(StdRealArray3d) array<real, 3>;
    %template(StdRealVector) vector<real>;
}

%template(ParametricShape2d) ParametricShape<2>;
%template(ParametricShape3d) ParametricShape<3>;

%include "../include/shape/spline2d.h"