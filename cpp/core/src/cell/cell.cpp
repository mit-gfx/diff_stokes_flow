#include "cell/cell.h"
#include "common/common.h"

template<int dim>
Cell<dim>::Cell() {
    corner_num_prod_ = 1;
    for (int i = 0; i < dim; ++i) {
        corner_num_prod_ *= 2;
        corner_nums_[i] = 2;
    }
}

template<int dim>
void Cell<dim>::Initialize(const real E, const real nu, const real threshold, const int edge_sample_num,
    const std::vector<real>& sdf_at_corners) {
    E_ = E;
    nu_ = nu;
    la_ = E * nu / (1 + nu) / (1 - 2 * nu);
    mu_ = E / 2 / (1 + nu);
    threshold_ = threshold;

    // Compute normal_ and offset_.
    FitBoundary(sdf_at_corners, normal_, offset_, normal_gradients_, offset_gradients_);

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
    sample_areas_gradients_ = MatrixXr::Zero(sample_num_prod_, corner_num_prod_);
    sample_boundary_areas_gradients_ = MatrixXr::Zero(sample_num_prod_, corner_num_prod_);
    for (int i = 0; i < sample_num_prod_; ++i) {
        VectorXr sample_area_gradients, sample_boundary_area_gradients;
        ComputeSampleAreaAndBoundaryArea(i, sample_areas_[i], sample_boundary_areas_[i], sample_area_gradients,
            sample_boundary_area_gradients);
        sample_areas_gradients_.row(i) = sample_area_gradients;
        sample_boundary_areas_gradients_.row(i) = sample_boundary_area_gradients;
    }

    area_ = 0;
    for (const real a : sample_areas_) area_ += a;
    area_gradients_ = VectorXr(sample_areas_gradients_.colwise().sum());

    // Compute energy_matrix_ and dirichlet_matrix_.
    ComputeEnergyMatrix(energy_matrix_, energy_matrix_gradients_);
    ComputeDirichletVector(dirichlet_vector_, dirichlet_vector_gradients_);
}

template<int dim>
const std::array<real, dim> Cell<dim>::py_normal() const {
    std::array<real, dim> normal;
    for (int i = 0; i < dim; ++i) normal[i] = normal_(i);
    return normal;
}

template<int dim>
const std::vector<std::vector<real>> Cell<dim>::py_energy_matrix() const {
    const int rows = static_cast<int>(energy_matrix_.rows());
    const int cols = static_cast<int>(energy_matrix_.cols());
    std::vector<std::vector<real>> K(rows, std::vector<real>(cols, 0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            K[i][j] = energy_matrix_(i, j);
    return K;
}

template<int dim>
const std::vector<real> Cell<dim>::py_dirichlet_vector() const {
    std::vector<real> b(corner_num_prod_, 0);
    for (int i = 0; i < corner_num_prod_; ++i) b[i] = dirichlet_vector_(i);
    return b;
}

template<int dim>
const std::array<real, dim> Cell<dim>::py_normal_gradient(const int corner_idx) const {
    CheckError(0 <= corner_idx && corner_idx < corner_num_prod_, "corner_idx out of bound.");
    std::array<real, dim> n;
    for (int i = 0; i < dim; ++i) n[i] = normal_gradients_(i, corner_idx);
    return n;
}

template<int dim>
const real Cell<dim>::py_offset_gradient(const int corner_idx) const {
    CheckError(0 <= corner_idx && corner_idx < corner_num_prod_, "corner_idx out of bound.");
    return offset_gradients_(corner_idx);
}

template<int dim>
const std::vector<real> Cell<dim>::py_sample_areas_gradient(const int corner_idx) const {
    CheckError(0 <= corner_idx && corner_idx < corner_num_prod_, "corner_idx out of bound.");
    std::vector<real> v(sample_num_prod_);
    for (int i = 0; i < sample_num_prod_; ++i) v[i] = sample_areas_gradients_(i, corner_idx);
    return v;
}

template<int dim>
const std::vector<real> Cell<dim>::py_sample_boundary_areas_gradient(const int corner_idx) const {
    CheckError(0 <= corner_idx && corner_idx < corner_num_prod_, "corner_idx out of bound.");
    std::vector<real> v(sample_num_prod_);
    for (int i = 0; i < sample_num_prod_; ++i) v[i] = sample_boundary_areas_gradients_(i, corner_idx);
    return v;
}

template<int dim>
const real Cell<dim>::py_area_gradient(const int corner_idx) const {
    CheckError(0 <= corner_idx && corner_idx < corner_num_prod_, "corner_idx out of bound.");
    return area_gradients_(corner_idx);
}

template<int dim>
const std::vector<std::vector<real>> Cell<dim>::py_energy_matrix_gradient(const int corner_idx) const {
    CheckError(0 <= corner_idx && corner_idx < corner_num_prod_, "corner_idx out of bound.");
    const int rows = static_cast<int>(energy_matrix_.rows());
    const int cols = static_cast<int>(energy_matrix_.cols());
    std::vector<std::vector<real>> K(rows, std::vector<real>(cols, 0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            K[i][j] = energy_matrix_gradients_[corner_idx](i, j);
    return K;
}

template<int dim>
const std::vector<real> Cell<dim>::py_dirichlet_vector_gradient(const int corner_idx) const {
    CheckError(0 <= corner_idx && corner_idx < corner_num_prod_, "corner_idx out of bound.");
    std::vector<real> v(corner_num_prod_);
    for (int i = 0; i < corner_num_prod_; ++i) v[i] = dirichlet_vector_gradients_(i, corner_idx);
    return v;
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
void Cell<dim>::FitBoundary(const std::vector<real>& sdf_at_corners, Eigen::Matrix<real, dim, 1>& normal, real& offset,
    Eigen::Matrix<real, dim, -1>& normal_gradients, VectorXr& offset_gradients) const {
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
    const Eigen::Matrix<real, dim + 1, -1> Ct = corners.transpose();
    const Eigen::Matrix<real, dim + 1, dim + 1> CtC = Ct * corners;
    const Eigen::Matrix<real, dim + 1, 1> plane_eq = CtC.ldlt().solve(Ct * sdf);
    normal = plane_eq.head(dim);
    offset = plane_eq(dim);

    // Compute gradients.
    const Eigen::Matrix<real, dim + 1, -1> jac = CtC.ldlt().solve(Ct);
    normal_gradients = jac.topRows(dim);
    offset_gradients = VectorXr(jac.row(dim));
}

template<int dim>
void Cell<dim>::ComputeSampleAreaAndBoundaryArea(const int sample_idx, real& area, real& boundary_area,
    VectorXr& area_gradients, VectorXr& boundary_area_gradients) const {
    // Input: normal_ and offset_: the plane equation in the cell coordinates ([0, 1]^dim).
    // Output: area: the area of the intersection between the subcell sample_idx and normal_.dot(x) + offset_ < 0.
    // normal_.dot(dx) + doffset = 0.
    // -doffset / |normal_| * boundary_area = darea.
    // boundary_area = -\partial area / \partial offset * |normal_|
    area_gradients = VectorXr::Zero(corner_num_prod_);
    boundary_area_gradients = VectorXr::Zero(corner_num_prod_);

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
    const Eigen::Matrix<real, dim, -1> scaled_normal_gradients = -dx * normal_gradients_;
    const real scaled_offset = -(origin.dot(normal_) + offset_);
    const VectorXr scaled_offset_gradients = -VectorXr(origin.transpose() * normal_gradients_) - offset_gradients_;
    const real scaled_offset_derivative = -1;

    // Now compute the intersected area between a unit cube and p * scaled_normal + scaled_offset >= 0.
    const real a = scaled_normal.prod();
    VectorXr a_gradients = VectorXr::Zero(corner_num_prod_);
    for (int i = 0; i < dim; ++i) {
        // a = \Pi scaled_normal(i)
        // a' = \Sum scaled_normal_gradient(i) * scaled_normal(j)
        VectorXr ai_gradients = VectorXr::Ones(corner_num_prod_);
        for (int j = 0; j < dim; ++j) {
            if (i == j) {
                ai_gradients = VectorXr(ai_gradients.array() * VectorXr(scaled_normal_gradients.row(i)).array());
            } else {
                ai_gradients *= scaled_normal(j);
            }
        }
        a_gradients += ai_gradients;
    }
    const real eps = Epsilon();
    CheckError(a > eps || a < -eps, "Singular case: boundaries are axis-aligned.");
    const real inv_a = 1 / a;
    const VectorXr inv_a_gradients = -a_gradients * inv_a * inv_a;
    real d_factorial = 1;
    for (int i = 0; i < dim; ++i) d_factorial *= (i + 1);
    const real inv_d_factorial = 1 / d_factorial;

    area = 0;
    real area_derivative = 0;
    VectorXr area_derivative_gradients = VectorXr::Zero(corner_num_prod_);
    for (int i = 0; i < corner_num_prod_; ++i) {
        const auto corner_idx = GetIndex(i, corner_nums_);
        // Check if corner_idx is inside the halfspace.
        real dist = scaled_offset;
        VectorXr dist_gradients = scaled_offset_gradients;
        real dist_derivative = scaled_offset_derivative;
        for (int j = 0; j < dim; ++j) {
            dist += corner_idx[j] * scaled_normal(j);
            dist_gradients += VectorXr(corner_idx[j] * scaled_normal_gradients.row(j));
        }
        if (dist < 0) continue;

        // Compute the contribution of this corner.
        int zero_num = 0;
        for (int j = 0; j < dim; ++j)
            if (corner_idx[j] == 0) ++zero_num;
        const real sign = zero_num % 2 ? -1 : 1;
        const real dist_dim_power = std::pow(dist, dim);
        const real dist_dim_minus_one_power = std::pow(dist, dim - 1);
        area += sign * inv_a * inv_d_factorial * dist_dim_power;
        area_gradients += sign * inv_d_factorial * (inv_a * dim * dist_dim_minus_one_power * dist_gradients
            + inv_a_gradients * dist_dim_power);
        area_derivative += sign * inv_a * inv_d_factorial * dim * dist_dim_minus_one_power * dist_derivative;
        area_derivative_gradients += sign * inv_d_factorial * dim * dist_derivative * (
            inv_a_gradients * dist_dim_minus_one_power + inv_a * (dim - 1) * std::pow(dist, dim - 2) * dist_gradients);
    }

    // Scale it back to the original coordinate.
    const real factor = std::pow(dx, dim);
    area *= factor;
    area_gradients *= factor;
    area_derivative *= factor;
    area_derivative_gradients *= factor;
    const real normal_len = normal_.norm();
    CheckError(normal_len > eps, "Singular normal.");
    const VectorXr normal_len_gradients = VectorXr(normal_.transpose() * normal_gradients_) / normal_len;
    boundary_area = -area_derivative * normal_len;
    boundary_area_gradients = -(area_derivative_gradients * normal_len + area_derivative * normal_len_gradients);
}

template<int dim>
void Cell<dim>::ComputeEnergyMatrix(MatrixXr& energy_matrix, std::vector<MatrixXr>& energy_matrix_gradients) const {
    energy_matrix = MatrixXr::Zero(dim * corner_num_prod_, dim * corner_num_prod_);
    energy_matrix_gradients.clear();
    energy_matrix_gradients.resize(corner_num_prod_, MatrixXr::Zero(dim * corner_num_prod_, dim * corner_num_prod_));
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

        // Compute energy_matrix_gradients_.
        for (int j = 0; j < corner_num_prod_; ++j) {
            energy_matrix_gradients[j] += energy_matrix_sample * sample_areas_gradients_(i, j);
        }
    }
}

template<int dim>
void Cell<dim>::ComputeDirichletVector(VectorXr& dirichlet_vector, MatrixXr& dirichlet_vector_gradients) const {
    dirichlet_vector = VectorXr::Zero(corner_num_prod_);
    dirichlet_vector_gradients = MatrixXr::Zero(corner_num_prod_, corner_num_prod_);
    const real dx = ToReal(1) / edge_sample_num_;
    const real eps = Epsilon();
    for (int i = 0; i < sample_num_prod_; ++i) {
        const auto sample_idx = GetIndex(i, sample_nums_);
        Eigen::Matrix<real, dim, 1> sample;
        for (int j = 0; j < dim; ++j) sample(j) = ToReal(sample_idx[j] + 0.5) * dx;
        // Project sample onto the boundary plane.
        // (sample + normal * t).dot(normal) + offset = 0.
        // normal.dot(normal) * t + sample.dot(normal) + offset = 0.
        const real a = sample.dot(normal_) + offset_;
        const VectorXr a_gradients = VectorXr(sample.transpose() * normal_gradients_) + offset_gradients_;
        const real b = normal_.dot(normal_);
        const VectorXr b_gradients = 2 * VectorXr(normal_.transpose() * normal_gradients_);
        CheckError(b > eps, "Singular boundary.");
        const real t = -a / b;
        const VectorXr t_gradients = (a * b_gradients - a_gradients * b) / (b * b);
        const auto projected = sample + normal_ * t;
        const MatrixXr projected_gradients = normal_gradients_ * t + normal_ * t_gradients.transpose();

        // Compute the contribution of this sample.
        for (int j = 0; j < corner_num_prod_; ++j) {
            const auto corner_idx = GetIndex(j, corner_nums_);
            real contrib = 1;
            VectorXr contrib_gradients = VectorXr::Zero(corner_num_prod_);
            for (int k = 0; k < dim; ++k) {
                // contrib = contrib * projected(k) or contrib * (1 - projected(k)).
                if (corner_idx[k]) {
                    contrib_gradients = VectorXr(contrib * projected_gradients.row(k))
                        + contrib_gradients * projected(k);
                    contrib *= projected(k);
                } else {
                    contrib_gradients = VectorXr(contrib * -projected_gradients.row(k))
                        + contrib_gradients * (1 - projected(k));
                    contrib *= 1 - projected(k);
                }
            }
            dirichlet_vector(j) += contrib * sample_boundary_areas_[i];
            dirichlet_vector_gradients.row(j) += VectorXr(contrib * sample_boundary_areas_gradients_.row(i))
                + contrib_gradients * sample_boundary_areas_[i];
        }
    }
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