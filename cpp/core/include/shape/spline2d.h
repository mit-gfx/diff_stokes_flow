#ifndef SHAPE_SPLINE2D_H
#define SHAPE_SPLINE2D_H

#include "shape/parametric_shape.h"

// A spline is defined by 4 control points:
// s(t) = (1 - t)^3 * p0 + 3 (1 - t)^2 t * p1 + 3 (1 - t) t^2 p2 + t^3 p3.
// We assume that t from 0 to 1 equals looping over the solid (positive) region in the counter-clockwise order.
class Spline2d : public ParametricShape<2> {
public:
    void Initialize(const std::array<int, 2>& cell_nums, const std::vector<real>& params);
    void Backward(const std::vector<real>& dl_dsigned_distances, std::vector<real>& dl_dparams) const;
    const real ComputeSignedDistance(const std::array<real, 2>& point);

private:
    const Vector2r GetSplinePoint(const real t) const;
    const Vector2r GetSplineDerivative(const real t) const;

    Matrix4r A_;
    Eigen::Matrix<real,4, 3> B_;
    Eigen::Matrix<real, 2, 4> cA_;
    Eigen::Matrix<real, 2, 3> cAB_;
    Vector6r c_;
};

#endif