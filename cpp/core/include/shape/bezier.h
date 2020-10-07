#ifndef SHAPE_BEZIER_H
#define SHAPE_BEZIER_H

#include "shape/parametric_shape.h"

// A Bezier curve is defined by 4 control points:
// s(t) = (1 - t)^3 * p0 + 3 (1 - t)^2 t * p1 + 3 (1 - t) t^2 p2 + t^3 p3.
// We assume that t from 0 to 1 equals looping over the solid (positive) region in the counter-clockwise order.
class Bezier2d : public ParametricShape<2> {
public:
    const real ComputeSignedDistanceAndGradients(const std::array<real, 2>& point,
        std::vector<real>& grad) const override;

private:
    void InitializeCustomizedData() override;
    const Vector2r GetBezierPoint(const real t) const;
    const Vector2r GetBezierDerivative(const real t) const;

    Matrix4r A_;
    Eigen::Matrix<real, 4, 3> B_;
    Eigen::Matrix<real, 2, 4> cA_;
    Eigen::Matrix<real, 2, 3> cAB_;
    Vector6r c_;

    // Gradients.
    std::array<Eigen::Matrix<real, 2, 4>, 8> cA_gradients_;
    std::array<Eigen::Matrix<real, 2, 3>, 8> cAB_gradients_;
    Eigen::Matrix<real, 6, 8> c_gradients_;
};

// We assume that Bezier3d is obtained by extruding a Bezier2d sketch on the xy plane along the z axis.
class Bezier3d : public ParametricShape<3> {
public:
    const real ComputeSignedDistanceAndGradients(const std::array<real, 3>& point,
        std::vector<real>& grad) const override;

private:
    void InitializeCustomizedData() override;

    std::shared_ptr<Bezier2d> sketch_;
};

#endif