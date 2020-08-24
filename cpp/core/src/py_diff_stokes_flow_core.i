%module py_diff_stokes_flow_core
%{
#include "../include/shape/parametric_shape.h"
#include "../include/shape/spline.h"
#include "../include/shape/shape_composition.h"
#include "../include/cell/cell.h"
#include "../include/scene/scene.h"
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
%include "../include/shape/shape_composition.h"
%include "../include/cell/cell.h"
%include "../include/scene/scene.h"

namespace std {
    %template(StdIntArray2d) array<int, 2>;
    %template(StdIntArray3d) array<int, 3>;
    %template(StdRealArray2d) array<real, 2>;
    %template(StdRealArray3d) array<real, 3>;
    %template(StdStringVector) vector<string>;
    %template(StdIntVector) vector<int>;
    %template(StdRealVector) vector<real>;
    %template(StdRealVectorVector) vector<vector<real>>;
}

%template(ParametricShape2d) ParametricShape<2>;
%template(ParametricShape3d) ParametricShape<3>;
%template(ShapeComposition2d) ShapeComposition<2>;
%template(ShapeComposition3d) ShapeComposition<3>;
%template(Cell2d) Cell<2>;
%template(Cell3d) Cell<3>;
%template(Scene2d) Scene<2>;
%template(Scene3d) Scene<3>;

%include "../include/shape/spline.h"