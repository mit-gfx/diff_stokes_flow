#include "cell/cell.h"
#include "common/common.h"

template<int dim>
void Cell<dim>::Initialize(const real E, const real nu, const real threshold, const int edge_sample_num,
    const std::vector<real>& sdf_at_corners) {
    corner_num_prod_ = 1;
    for (int i = 0; i < dim; ++i) {
        corner_num_prod_ *= 2;
        corner_nums_[i] = 2;
    }

    E_ = E;
    nu_ = nu;
    la_ = E * nu / (1 + nu) / (1 - 2 * nu);
    mu_ = E / 2 / (1 + nu);
    threshold_ = threshold;

    // Compute normal_ and offset_.
    const Eigen::Matrix<real, dim + 1, 1> plane_eq = FitBoundary(sdf_at_corners);
    normal_ = plane_eq.head(dim);
    offset_ = plane_eq(dim);

    // Compute areas.
    edge_sample_num_ = edge_sample_num;
    sample_num_prod_ = 1;
    for (int i = 0; i < dim; ++i) {
        sample_num_prod_ *= edge_sample_num_;
        sample_nums_[i] = edge_sample_num_;
    }
    sample_areas_.clear();
    sample_areas_.resize(sample_num_prod_, 0);
    sample_boundary_areas_.clear();
    sample_boundary_areas_.resize(sample_num_prod_, 0);
    for (int i = 0; i < sample_num_prod_; ++i)
        ComputeSampleAreaAndBoundaryArea(i, sample_areas_[i], sample_boundary_areas_[i]);

    area_ = 0;
    for (const real a : sample_areas_) area_ += a;

    // Compute energy_matrix_ and dirichlet_matrix_.
    energy_matrix_ = ComputeEnergyMatrix();
    dirichlet_vector_ = ComputeDirichletVector();
}

template<int dim>
const real Cell<dim>::sample_area(const int sample_idx) const {
    CheckError(0 <= sample_idx && sample_idx < static_cast<int>(sample_areas_.size()), "sample_idx out of range.");
    return sample_areas_[sample_idx];
}

template<int dim>
const real Cell<dim>::sample_area(const std::array<int, dim>& sample_idx) const {
    return sample_areas_[GetIndex(sample_idx, sample_nums_)];
}

template<int dim>
const real Cell<dim>::sample_boundary_area(const int sample_idx) const {
    CheckError(0 <= sample_idx && sample_idx < static_cast<int>(sample_boundary_areas_.size()),
        "sample_idx out of range.");
    return sample_boundary_areas_[sample_idx];
}

template<int dim>
const real Cell<dim>::sample_boundary_area(const std::array<int, dim>& sample_idx) const {
    return sample_boundary_areas_[GetIndex(sample_idx, sample_nums_)];
}

template<int dim>
const Eigen::Matrix<real, dim + 1, 1> Cell<dim>::FitBoundary(const std::vector<real>& sdf_at_corners) const {
    MatrixXr corners(corner_num_prod_, dim + 1);
    for (int i = 0; i < corner_num_prod_; ++i) {
        const auto idx = GetIndex(i, corner_nums_);
        for (int j = 0; j < dim; ++j) corners(i, j) = idx[j];
        corners(i, dim) = 1;
    }
    CheckError(static_cast<int>(sdf_at_corners.size()) == corner_num_prod_, "Incompatible sdf_at_corners.size.");
    const VectorXr sdf = Eigen::Map<const VectorXr>(sdf_at_corners.data(), corner_num_prod_);
    // min \|corners * x - sdf\|.
    // x = (corners.T * corners)^(-1) * (corners.T * sdf).
    const Eigen::Matrix<real, dim + 1, dim + 1> CtC = corners.transpose() * corners;
    const Eigen::Matrix<real, dim + 1, 1> plane_eq = CtC.ldlt().solve(corners.transpose() * sdf);
    return plane_eq;
}

template<int dim>
void Cell<dim>::ComputeSampleAreaAndBoundaryArea(const int sample_idx, real& area, real& boundary_area) const {
    // Input: normal_ and offset_: the plane equation in the cell coordinates ([0, 1]^dim).
    // Output: area: the area of the intersection between the subcell sample_idx and normal_.dot(x) + offset_ < 0.
    // normal_.dot(dx) + doffset = 0.
    // -doffset / |normal_| * boundary_area = darea.
    // boundary_area = -\partial area / \partial offset * |normal_|

    const auto idx = GetIndex(sample_idx, sample_nums_);
    const real dx = 1 / ToReal(edge_sample_num_);
    // Tranform the coordinates.
    Eigen::Matrix<real, dim, 1> origin;
    for (int i = 0; i < dim; ++i) origin(i) = idx[i] * dx;
    // Plane equation in the new coordinates:
    // (p * dx + origin).dot(normal_) + offset_ >= 0.
    // p * dot(dx * normal_) + origin.dot(normal_) + offset_ >= 0.
    // Since we care about the fluid area, we flip the sign.
    const Eigen::Matrix<real, dim, 1> scaled_normal = -dx * normal_;
    const real scaled_offset = -(origin.dot(normal_) + offset_);
    const real scaled_offset_derivative = -1;

    // Now compute the intersected area between a unit cube and p * scaled_normal + scaled_offset >= 0.
    const real a = scaled_normal.prod();
    const real eps = std::numeric_limits<real>::epsilon();
    CheckError(a > eps || a < -eps, "Singular case: boundaries are axis-aligned.");
    const real inv_a = 1 / a;
    real d_factorial = 1;
    for (int i = 0; i < dim; ++i) d_factorial *= (i + 1);
    const real inv_d_factorial = 1 / d_factorial;

    area = 0;
    real area_derivative = 0;
    for (int i = 0; i < corner_num_prod_; ++i) {
        const auto corner_idx = GetIndex(i, corner_nums_);
        // Check if corner_idx is inside the halfspace.
        real dist = scaled_offset;
        real dist_derivative = scaled_offset_derivative;
        for (int j = 0; j < dim; ++j) {
            dist += corner_idx[j] * scaled_normal(j);
        }
        if (dist < 0) continue;

        // Compute the contribution of this corner.
        int zero_num = 0;
        for (int j = 0; j < dim; ++j)
            if (corner_idx[j] == 0) ++zero_num;
        const real sign = zero_num % 2 ? -1 : 1;
        area += sign * inv_a * inv_d_factorial * std::pow(dist, dim);
        area_derivative += sign * inv_a * inv_d_factorial * dim * std::pow(dist, dim - 1) * dist_derivative;
    }

    // Scale it back to the original coordinate.
    const real factor = std::pow(dx, dim);
    area *= factor;
    area_derivative *= factor;
    boundary_area = -area_derivative * normal_.norm();
}

template<int dim>
const MatrixXr Cell<dim>::ComputeEnergyMatrix() const {
    MatrixXr energy_matrix(dim * corner_num_prod_, dim * corner_num_prod_);
    energy_matrix.setZero();
    const real dx = ToReal(1) / edge_sample_num_;
    for (int i = 0; i < sample_num_prod_; ++i) {
        const auto sample_idx = GetIndex(i, sample_nums_);
        // Compute the sample point.
        Eigen::Matrix<real, dim, 1> coord_sample;
        for (int j = 0; j < dim; ++j) coord_sample(j) = ToReal(sample_idx[j] + 0.5) * dx;

        // Write F, the deformation gradient, as F = u_to_F * u + F_const.
        const MatrixXr u_to_F = VelocityToDeformationGradient(coord_sample);

        // Define the strain tensor eps.
        // eps = 0.5 * (F + F.T) - I
        MatrixXr u_to_eps(u_to_F);
        for (int ii = 0; ii < dim; ++ii)
            for (int jj = 0; jj < dim; ++jj)
                u_to_eps.row(ii * dim + jj) += u_to_F.row(jj * dim + ii);
        u_to_eps *= 0.5;
        // Now eps = u_to_eps * u.

        // Material model.
        // \Psi(F) = mu * (eps : eps) + lambda * 0.5 * tr(eps)^2.
        MatrixXr energy_matrix_sample = 2 * mu_ * u_to_eps.transpose() * u_to_eps;
        // Part II: lambda * 0.5 * tr(eps)^2.
        // = lambda * 0.5 * (u_to_eps[0] * u + u_to_eps[3] * u)^2
        RowVectorXr trace = RowVectorXr::Zero(dim * corner_num_prod_);
        for (int ii = 0; ii < dim; ++ii)
            trace += u_to_eps.row(ii * dim + ii);
        energy_matrix_sample += la_ * trace.transpose() * trace;
        energy_matrix += energy_matrix_sample * sample_areas_[i];
    }
    return energy_matrix;
}

template<int dim>
const VectorXr Cell<dim>::ComputeDirichletVector() const {
    VectorXr dirichlet_vector = VectorXr::Zero(corner_num_prod_);
    const real dx = ToReal(1) / edge_sample_num_;
    const real eps = std::numeric_limits<real>::epsilon();
    for (int i = 0; i < sample_num_prod_; ++i) {
        const auto sample_idx = GetIndex(i, sample_nums_);
        Eigen::Matrix<real, dim, 1> sample;
        for (int j = 0; j < dim; ++j) sample(j) = ToReal(sample_idx[j] + 0.5) * dx;
        // Project sample onto the boundary plane.
        // (sample + normal * t).dot(normal) + offset = 0.
        // normal.dot(normal) * t + sample.dot(normal) + offset = 0.
        const real a = sample.dot(normal_) + offset_;
        const real b = normal_.dot(normal_);
        CheckError(b > eps, "Singular boundary.");
        const real t = -a / b;
        const auto projected = sample + normal_ * t;

        // Compute the contribution of this sample.
        for (int j = 0; j < corner_num_prod_; ++j) {
            const auto corner_idx = GetIndex(j, corner_nums_);
            real contrib = 1;
            for (int k = 0; k < dim; ++k) contrib *= corner_idx[k] ? projected(k) : (1 - projected(k));
            dirichlet_vector(j) += contrib * sample_boundary_areas_[i];
        }
    }
    return dirichlet_vector;
}

template<>
const MatrixXr Cell<2>::VelocityToDeformationGradient(const Vector2r& material_coordinates) const {
    MatrixXr u_to_F(4, 8);
    u_to_F.setZero();
    // X = (a, b) = material_coordinates \in [0, 1]^2.
    // Xij = (i, j), i, j \in \{0, \1}.
    // \phi(X) = (X00 + u00) (1 - a)(1 - b) + (X01 + u01) (1 - a)b +
    //           (X10 + u10) a(1 - b) + (X11 + u11) ab.
    //         = (X00 + u00) (1 - a - b + ab) + (X01 + u01) (b - ab) +
    //           (X10 + u10) (a - ab) + (X11 + u11) ab.
    //         = X + u00 (1 - a - b + ab) + u01 (b - ab) + u10 (a - ab) + u11 ab.
    // F = \partial \phi(X) / \partial X
    //   = I + u00 * (-grad_a - grad_b + grad_ab) + u01 * (grad_b - grad_ab)
    //       + u10 * (grad_a - grad_ab) + u11 * grad_ab.
    const Vector2r grad_a(1, 0);
    const Vector2r grad_b(0, 1);
    const Vector2r grad_ab(material_coordinates.y(), material_coordinates.x());
    int cnt = 0;
    for (int ii = 0; ii < 2; ++ii)
        for (int jj = 0; jj < 2; ++jj) {
            u_to_F(cnt, ii) = -grad_a(jj) - grad_b(jj) + grad_ab(jj);
            u_to_F(cnt, 2 + ii) = grad_b(jj) - grad_ab(jj);
            u_to_F(cnt, 4 + ii) = grad_a(jj) - grad_ab(jj);
            u_to_F(cnt, 6 + ii) = grad_ab(jj);
            ++cnt;
        }
    return u_to_F;
}

template<>
const MatrixXr Cell<3>::VelocityToDeformationGradient(const Vector3r& material_coordinates) const {
    MatrixXr u_to_F(9, 24);
    u_to_F.setZero();
    // X = (a, b, c) = material_coordinates \in [0, 1]^3.
    // Xijk = (i, j, k), i, j, k \in \{0, \1}.
    // \phi(X) = X + u000 (1 - a)(1 - b)(1 - c) + u001 (1 - a)(1 - b)c
    //             + u010 (1 - a)b(1 - c) + u011 (1 - a)bc
    //             + u100 a(1 - b)(1 - c) + u101 a(1 - b)c
    //             + u110 ab(1 - c) + u111 abc.
    const Vector3r grad_a(1, 0, 0);
    const Vector3r grad_b(0, 1, 0);
    const Vector3r grad_c(0, 0, 1);
    const Vector3r grad_ab(material_coordinates.y(), material_coordinates.x(), 0);
    const Vector3r grad_ac(material_coordinates.z(), 0, material_coordinates.x());
    const Vector3r grad_bc(0, material_coordinates.z(), material_coordinates.y());
    const Vector3r grad_abc(material_coordinates.y() * material_coordinates.z(),
        material_coordinates.x() * material_coordinates.z(),
        material_coordinates.x() * material_coordinates.y());
    // F(X) = I + GRAD u000 (-a - b - c + ab + ac + bc - abc) +
    //        u001 (c - ac - bc + abc) +
    //        u010 (b - ab - bc + abc) +
    //        u011 (bc - abc) +
    //        u100 (a - ab - ac + abc) +
    //        u101 (ac - abc) +
    //        u110 (ab - abc) +
    //        u111 abc.
    int cnt = 0;
    std::array<Vector3r, 8> grad_coeff{
        -grad_a - grad_b - grad_c + grad_ab + grad_ac + grad_bc - grad_abc,
        grad_c - grad_ac - grad_bc + grad_abc,
        grad_b - grad_ab - grad_bc + grad_abc,
        grad_bc - grad_abc,
        grad_a - grad_ab - grad_ac + grad_abc,
        grad_ac - grad_abc,
        grad_ab - grad_abc,
        grad_abc
    };
    for (int ii = 0; ii < 3; ++ii)
        for (int jj = 0; jj < 3; ++jj) {
            for (int kk = 0; kk < 8; ++kk) {
                u_to_F(cnt, kk * 3 + ii) = grad_coeff[kk](jj);
            }
            ++cnt;
        }
    return u_to_F;
}

template class Cell<2>;
template class Cell<3>;