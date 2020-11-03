#include "shape/polar_bezier.h"
#include "common/common.h"

PolarBezier2d::PolarBezier2d(const bool flip)
    : flip_(flip) {}

const real PolarBezier2d::ComputeSignedDistanceAndGradients(const std::array<real, 2>& point,
    std::vector<real>& grad) const {
    // Determine the curve that covers point.
    const real x = point[0], y = point[1];
    real theta_xy = std::atan2(y - center_.y(), x - center_.x()) - angle_offset_;
    // theta_xy is in the range of (-pi - angle_offset_, pi - angle_offset_) now.
    // Now we would like to shift theta_xy to the region (0, 2pi).
    while (theta_xy < 0) theta_xy += 2 * Pi();
    CheckError(0 <= theta_xy && theta_xy < 2 * Pi(), "Something went wrong with your usage of atan2.");
    const int bezier_idx = static_cast<int>(theta_xy / (angle_step_size_ * 3)) % bezier_num_;

    // Compute the distance.
    real min_abs_dist = std::numeric_limits<real>::infinity();
    int min_idx = -1;
    VectorXr min_grad;
    real sign = 1.0;
    for (int i = 0; i < bezier_num_; ++i) {
        const auto& curve = bezier_curves_[i];
        std::vector<real> g;
        const real d = curve->ComputeSignedDistanceAndGradients(point, g);
        const real d_abs = d > 0 ? d : -d;
        if (d_abs < min_abs_dist) {
            min_abs_dist = d_abs;
            min_idx = i;
            min_grad = ToEigenVector(g) * (d > 0 ? 1 : -1);
        }
        if (i == bezier_idx) sign = (d > 0 ? 1 : -1);
    }
    // Compute gradients.
    VectorXr g = VectorXr::Zero(param_num());
    for (int k = 0; k < 4; ++k) {
        const int idx = (min_idx * 3 + k) % (bezier_num_ * 3);
        g += VectorXr(RowVector2r(min_grad.segment(k * 2, 2)) * control_points_gradients_[idx]);
    }
    grad = ToStdVector(sign * g * (flip_ ? -1 : 1));
    return sign * min_abs_dist * (flip_ ? -1 : 1);
}

void PolarBezier2d::InitializeCustomizedData() {
    // rho, cx, cy, angle_offset.
    bezier_num_ = (param_num() - 3) / 2;
    CheckError(bezier_num_ >= 3 && bezier_num_ * 2 + 3 == param_num(), "Inconsistent number of parameters.");

    rho_.resize(bezier_num_ * 2);
    for (int i = 0; i < bezier_num_ * 2; ++i) rho_[i] = params()[i];
    center_ = Vector2r(params()[bezier_num_ * 2], params()[bezier_num_ * 2 + 1]);
    angle_offset_ = params()[bezier_num_ * 2 + 2];
    angle_step_size_ = Pi() * 2 / (bezier_num_ * 3);

    control_points_.resize(bezier_num_ * 3);
    control_points_gradients_.resize(bezier_num_ * 3);
    for (int i = 0; i < bezier_num_; ++i) {
        for (int j = 1; j < 3; ++j) {
            const real theta = angle_offset_ + angle_step_size_ * (i * 3 + j);
            const real c = std::cos(theta), s = std::sin(theta);
            control_points_[i * 3 + j] = Vector2r(
                c * rho_[i * 2 + j - 1],
                s * rho_[i * 2 + j - 1]
            ) + center_;
            // Compute gradients of the control_points above w.r.t. rho, center, and angle_offset.
            control_points_gradients_[i * 3 + j] = Matrix2Xr::Zero(2, param_num());
            control_points_gradients_[i * 3 + j](0, i * 2 + j - 1) = c;
            control_points_gradients_[i * 3 + j](1, i * 2 + j - 1) = s;
            control_points_gradients_[i * 3 + j](0, bezier_num_ * 2) = 1;
            control_points_gradients_[i * 3 + j](1, bezier_num_ * 2 + 1) = 1;
            control_points_gradients_[i * 3 + j](0, bezier_num_ * 2 + 2) = -s * rho_[i * 2 + j - 1];
            control_points_gradients_[i * 3 + j](1, bezier_num_ * 2 + 2) = c * rho_[i * 2 + j - 1];
        }
    }
    for (int i = 0; i < bezier_num_; ++i) {
        const real theta = angle_offset_ + i * 3 * angle_step_size_;
        const real c = std::cos(theta);
        const real s = std::sin(theta);
        const Vector2r d(c, s);
        Matrix2Xr d_grad = Matrix2Xr::Zero(2, param_num());
        d_grad(0, bezier_num_ * 2 + 2) = -s;
        d_grad(1, bezier_num_ * 2 + 2) = c;
        const int p0_idx = (3 * i - 1 + bezier_num_ * 3) % (bezier_num_ * 3);
        const int p1_idx = (3 * i + 1) % (bezier_num_ * 3);
        const Vector2r& p0 = control_points_[p0_idx];
        const Matrix2Xr p0_grad = control_points_gradients_[p0_idx];
        const Vector2r& p1 = control_points_[p1_idx];
        const Matrix2Xr p1_grad = control_points_gradients_[p1_idx];
        // p0 + t * (p1 - p0) = d * u + center.
        // [p1 - p0, -d] * [t, u] = center - p0.
        Matrix2r A;
        A.col(0) = p1 - p0;
        A.col(1) = -d;
        const Matrix2r A_inv = A.inverse();
        const Vector2r tu = A_inv * (center_ - p0);
        control_points_[3 * i] = d * tu(1) + center_;
        Matrix2Xr center_grad = Matrix2Xr::Zero(2, param_num());
        center_grad(0, bezier_num_ * 2) = 1;
        center_grad(1, bezier_num_ * 2 + 1) = 1;
        control_points_gradients_[3 * i] = Matrix2Xr::Zero(2, param_num());
        for (int j = 0; j < param_num(); ++j) {
            Matrix2r A_grad = Matrix2r::Zero();
            A_grad.col(0) = p1_grad.col(j) - p0_grad.col(j);
            A_grad.col(1) = -d_grad.col(j);
            // A * A_inv = I.
            // A' * A_inv + A * A_inv' = 0.
            // A * A_inv' = -A' * A_inv.
            // A_inv' = -A_inv * A' * A_inv.
            const Matrix2r A_inv_grad = -A_inv * A_grad * A_inv;
            const Vector2r tu_grad = A_inv_grad * (center_ - p0) + A_inv * (center_grad.col(j) - p0_grad.col(j));
            control_points_gradients_[3 * i].col(j) = d * tu_grad(1) + d_grad.col(j) * tu(1) + center_grad.col(j);
        }
    }

    bezier_curves_.clear();
    bezier_curves_.resize(bezier_num_, nullptr);
    for (int i = 0; i < bezier_num_; ++i) {
        std::vector<real> bezier_params(8, 0);
        for (int k = 0; k < 4; ++k) {
            bezier_params[2 * k] = control_points_[(3 * i + k) % (bezier_num_ * 3)](0);
            bezier_params[2 * k + 1] = control_points_[(3 * i + k) % (bezier_num_ * 3)](1);
        }
        bezier_curves_[i] = std::make_shared<Bezier2d>();
        bezier_curves_[i]->Initialize(cell_nums(), bezier_params, false);
    }
}

PolarBezier3d::PolarBezier3d(const bool flip, const int z_level_num)
    : flip_(flip), z_level_num_(z_level_num) {
    CheckError(z_level_num_ >= 2, "Inconsistent z_level_num.");
}

const real PolarBezier3d::ComputeSignedDistanceAndGradients(const std::array<real, 3>& point,
    std::vector<real>& grad) const {
    // Use z to determine the rho level.
    std::vector<real> rho_single(2 * bezier_num_ + 3, 0);
    const real z = point[2];
    const real t0 = z / cell_nums()[2];
    RowVectorXr t = RowVectorXr::Zero(z_level_num_);
    t(0) = 1;
    for (int j = 1; j < z_level_num_; ++j) t(j) = t(j - 1) * t0;
    const VectorXr t_coeff(t * coeffs_);
    for (int i = 0; i < 2 * bezier_num_; ++i) {
        VectorXr p = VectorXr::Zero(z_level_num_);
        for (int j = 0; j < z_level_num_; ++j) {
            p(j) = rho_[j][i];
        }
        rho_single[i] = t_coeff.dot(p);
    }
    PolarBezier2d polar_bezier(flip_);
    std::array<int, 2> xy_cell_nums{ cell_nums()[0], cell_nums()[1] };
    rho_single[2 * bezier_num_] = center_.x();
    rho_single[2 * bezier_num_ + 1] = center_.y();
    rho_single[2 * bezier_num_ + 2] = angle_offset_;
    polar_bezier.Initialize(xy_cell_nums, rho_single, false);

    std::array<real, 2> xy_point{ point[0], point[1] };
    std::vector<real> xy_grad;
    const real dist = polar_bezier.ComputeSignedDistanceAndGradients(xy_point, xy_grad);

    // Compute gradients.
    grad.clear();
    grad.resize(param_num(), 0);
    for (int i = 0; i < z_level_num_; ++i)
        for (int j = 0; j < 2 * bezier_num_; ++j) {
            grad[i * 2 * bezier_num_ + j] += xy_grad[j] * t_coeff(i);
        }
    for (int i = 0; i < 3; ++i)
        grad[z_level_num_ * 2 * bezier_num_ + i] += xy_grad[2 * bezier_num_ + i];

    return dist;
}

void PolarBezier3d::InitializeCustomizedData() {
    // - rho: 2 * bezier_num_ * z_level_num.
    // - cx, cy.
    // - angle_offset.
    // bezier_num_ * 2 * z_level_num + 2 + 1 == param_num().
    bezier_num_ = (param_num() - 3) / (2 * z_level_num_);
    CheckError(bezier_num_ * 2 * z_level_num_ + 3 == param_num(), "Inconsistent param_num.");

    rho_.clear();
    rho_.resize(z_level_num_);
    for (int z = 0; z < z_level_num_; ++z) {
        rho_[z].clear();
        rho_[z].resize(bezier_num_ * 2);
        for (int k = 0; k < bezier_num_ * 2; ++k)
            rho_[z][k] = params()[z * 2 * bezier_num_ + k];
    }
    center_ = Vector2r(params()[bezier_num_ * 2 * z_level_num_], params()[bezier_num_ * 2 * z_level_num_ + 1]);
    angle_offset_ = params()[bezier_num_ * 2 * z_level_num_ + 2];
    angle_step_size_ = Pi() * 2 / (bezier_num_ * 3);
    dz_ = ToReal(cell_nums()[2]) / (z_level_num_ - 1);

    // Derived data.
    // coeffs_ are used for interpolating curves vertically.
    coeffs_ = MatrixXr::Zero(z_level_num_, z_level_num_);
    const int m = z_level_num_ - 1;
    // p(t) = (1 - t)^m p0 + ... + t^m p_m.
    //      = \sum C_{m, k} (1 - t)^{m-k}t^k p_k.    k = 0, 1, 2, ..., m.
    int cmk = 1;
    // Fill in the first column: (1 - t)^m p_0.
    for (int i = 0; i < m + 1; ++i) {
        coeffs_(i, 0) = cmk;
        cmk *= -(m - i);
        cmk /= (i + 1);
        // i = 0 => cmk = -m;
        // i = 1 => cmk = m * (m - 1) / 2.
        // i = 2 => cmk = -m * (m - 1) * (m - 2) / (3 * 2).
    }
    cmk = m;
    for (int k = 1; k < m + 1; ++k) {
        // Contribution of p_k --- This is column k of coeffs_.
        // cmk * (1 - t)^(m - k)t^k.
        int cmj = 1;
        for (int j = 0; j < m - k + 1; ++j) {
            coeffs_(j + k, k) = cmk * cmj;
            cmj *= -(m - k - j);
            cmj /= (j + 1);
        }
        // Expand C_{m, k} (1 - t)^{m - k}t^k.
        cmk = cmk * (m - k) / (k + 1);
    }
    // For sanity check:
    // num_z_level_ = 2:
    // p(t) = (1 - t)p0 + tp1.
    //      = (1, t) * (p0, p1 - p0).
    //      = (1, t) * [1, 0; -1, 1] * [p0, p1].
    // num_z_level_ = 3:
    // p(t) = (1 - t)^2p0 + 2(1 - t)tp1 + t^2p2.
    //      = (1, -2t, t^2)p0 + (2t, -2t^2)p1 + t^2p2.
    //      = (1, t, t^2) * (p0, -2p0 + 2p1, p0 - 2p1 + p2).
    //      = (1, t, t^2) * [1, 0, 0; -2, 2, 0; 1, -2, 1] * [p0, p1, p2].
    // num_z_level_ = 4:
    // p(t) = (1 - t)^3p0 + 3 (1-t)^2 tp1 + 3(1-t)t^2p2 + t^3p3.
    //      = (1 - 3t + 3t^2 - t^3) p0 + 3(t - 2t^2 + t^3) p1 + (3t^2 - 3t^3) p2 + t^3 p3.
    //      = (1, t, t^2, t^3) * [p0, -3p0 + 3p1; 3p0 - 6p1 + 3p2; -p0 + 3p1 - 3p2 + p3].
}