#include "shape/bezier.h"
#include "common/common.h"
#include "unsupported/Eigen/Polynomials"

void Bezier2d::InitializeCustomizedData() {
    CheckError(param_num() == 8, "Inconsistent number of parameters.");
    // s(t) = control_points * A * [1, t, t^2, t^3].
    // s'(t) = control_points * A * [0, 1, 2 * t, 3 * t * t].
    //       = control_points * A * B * [1, t, t^2]
    // s(t) = (1 - t)^3 p0 + 3 (1 - t)^2 t p1 + 3 (1 - t) t^2 p2 + t^3 p3.
    //      = (1 - 3t + 3t^2 - t^3) p0 +
    //      = (3t - 6t^2 + 3t^3) p1 +
    //      = (3t^2 - 3t^3) p2 +
    //      = t^3 p3.
    control_points_ = Eigen::Map<const Eigen::Matrix<real, 2, 4>>(params().data(), 2, 4);
    A_ << 1, -3, 3, -1,
        0, 3, -6, 3,
        0, 0, 3, -3,
        0, 0, 0, 1;
    B_ << 0, 0, 0,
        1, 0, 0,
        0, 2, 0,
        0, 0, 3;
    cA_ = control_points_ * A_;
    cAB_ = control_points_ * A_ * B_;
    // When solving for the minimal distance from a point to the bezier curve, we solve:
    // s'(t)^T * (s(t) - point) = 0.
    // [1, t, t^2] * B^T * A^T * control_points^T * control_points * A * [1, t, t^2, t^3] -
    // [1, t, t^2] * B^T * A^T * control_points^T * point = 0.
    // [1, t, t^2, t^3, t^4, t^5] * c_ - [1, t, t^2] * (cAB_^T * point) = 0.
    c_.setZero();
    const Eigen::Matrix<real, 3, 4> C = cAB_.transpose() * cA_;
    // [1, t, t^2] * C * [1, t, t^2, t^3] - [1, t, t^2] * (cAB_^T * point) = 0.
    c_(0) = C(0, 0);
    c_(1) = C(0, 1) + C(1, 0);
    c_(2) = C(0, 2) + C(1, 1) + C(2, 0);
    c_(3) = C(0, 3) + C(1, 2) + C(2, 1);
    c_(4) = C(1, 3) + C(2, 2);
    c_(5) = C(2, 3);

    // Gradients.
    c_gradients_.setZero();
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 2; ++i) {
            const int idx = j * 2 + i;
            cA_gradients_[idx].setZero();
            cA_gradients_[idx].row(i) = A_.row(j);
            cAB_gradients_[idx].setZero();
            cAB_gradients_[idx].row(i) = (A_ * B_).row(j);
            const Eigen::Matrix<real, 3, 4> C_gradients = cAB_gradients_[idx].transpose() * cA_
                + cAB_.transpose() * cA_gradients_[idx];
            c_gradients_(0, idx) = C_gradients(0, 0);
            c_gradients_(1, idx) = C_gradients(0, 1) + C_gradients(1, 0);
            c_gradients_(2, idx) = C_gradients(0, 2) + C_gradients(1, 1) + C_gradients(2, 0);
            c_gradients_(3, idx) = C_gradients(0, 3) + C_gradients(1, 2) + C_gradients(2, 1);
            c_gradients_(4, idx) = C_gradients(1, 3) + C_gradients(2, 2);
            c_gradients_(5, idx) = C_gradients(2, 3);
        }
}

const real Bezier2d::ComputeSignedDistanceAndGradients(const std::array<real, 2>& point,
    std::vector<real>& grad) const {
    const Vector2r p(point[0], point[1]);
    Vector6r coeff = c_;
    coeff.head(3) -= p.transpose() * cAB_;

    // Now solve [1, t, t^2, t^3, t^4, t^5] * coeff = 0.
    Eigen::PolynomialSolver<real, 5> dist_solver;
    dist_solver.compute(coeff);
    std::vector<real> ts_full;
    dist_solver.realRoots(ts_full);

    // Add two end points - this concludes the candidate set.
    std::vector<real> ts{ 0, 1 };
    for (const real& t : ts_full)
        if (0 <= t && t <= 1)
            ts.push_back(t);

    // Pick the minimal distance among them.
    real min_dist = std::numeric_limits<real>::infinity();
    real min_t = 0;
    Vector2r min_proj(0, 0);
    for (const real t : ts) {
        Vector2r proj = GetBezierPoint(t);
        const real dist = (proj - p).norm();
        if (dist < min_dist) {
            min_dist = dist;
            min_proj = proj;
            min_t = t;
        }
    }

    // Determine the sign.
    const Vector2r min_tangent = GetBezierDerivative(min_t);
    if (min_tangent.norm() == 0) {
        std::cout << "The Bezier curve is singular. Control points are probably duplicated." << std::endl;
        std::cout << control_points_ << std::endl;
        CheckError(false, "Singular Bezier curve.");
    }
    const Vector2r q = min_proj - p;
    // Consider the sign of q x min_tangent: positive = interior.
    const real z = q.x() * min_tangent.y() - q.y() * min_tangent.x();
    const real sign = z >= 0 ? 1.0 : -1.0;

    // Compute the gradient.
    // control_point -> coeff -> min_t -> min_proj -> min_dist.
    // According to the envelope theorem, we can safely assume min_t does not change during the gradient computation.
    // min_proj = GetBezierPoint(t) = cA_ * ts.
    const real min_t2 = min_t * min_t;
    const real min_t3 = min_t * min_t2;
    const Vector4r min_ts(1, min_t, min_t2, min_t3);
    Eigen::Matrix<real, 2, 8> min_proj_gradients; min_proj_gradients.setZero();
    for (int i = 0; i < 8; ++i) {
        min_proj_gradients.col(i) = cA_gradients_[i] * min_ts;
    }
    // min_proj -> min_dist: min_dist = |min_proj - p|.
    const real eps = Epsilon();
    Vector2r q_unit = Vector2r::Zero();
    if (min_dist > eps) q_unit = q / min_dist;
    const Vector8r grad_vec(q_unit.transpose() * min_proj_gradients);
    grad.resize(8);
    for (int i = 0; i < 8; ++i) grad[i] = sign * grad_vec(i);
    return sign * min_dist;
}

const Vector2r Bezier2d::GetBezierPoint(const real t) const {
    const Vector4r ts(1, t, t * t, t * t * t);
    return cA_ * ts;
}

const Vector2r Bezier2d::GetBezierDerivative(const real t) const {
    const Vector3r ts(1, t, t * t);
    return cAB_ * ts;
}

const real Bezier3d::ComputeSignedDistanceAndGradients(const std::array<real, 3>& point,
    std::vector<real>& grad) const {
    // (point + dir_ * t)[2] = 0.
    // This is an approximation.
    const real t = -point[2] / dir_(2);
    std::array<real, 2> sketch_point{ point[0] + t * dir_(0), point[1] + t * dir_(1) };
    return sketch_->ComputeSignedDistanceAndGradients(sketch_point, grad);
}

void Bezier3d::InitializeCustomizedData() {
    CheckError(param_num() == 11, "Inconsistent number of parameters.");
    CheckError(params()[10] != 0, "Invalid extrusion direction.");

    sketch_ = std::make_shared<Bezier2d>();
    std::array<int, 2> sketch_cell_nums;
    for (int i = 0; i < 2; ++i) sketch_cell_nums[i] = cell_num(i);
    const std::vector<real> sketch_params(params().data(), params().data() + 8);
    sketch_->Initialize(sketch_cell_nums, sketch_params);
    for (int i = 0; i < 3; ++i) dir_(i) = params()[8 + i];
}