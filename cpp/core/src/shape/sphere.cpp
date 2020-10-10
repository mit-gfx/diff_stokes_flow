#include "shape/sphere.h"
#include "common/common.h"

template<int dim>
const real Sphere<dim>::ComputeSignedDistanceAndGradients(const std::array<real, dim>& point,
    std::vector<real>& grad) const {
    Eigen::Matrix<real, dim, 1> p;
    for (int i = 0; i < dim; ++i) p(i) = point[i];
    // dist = r - |c - p|.
    real cp_norm = (center_ - p).norm();
    const real dist = radius_ - cp_norm;

    // Compute gradients.
    Eigen::Matrix<real, dim + 1, 1> radius_grad, cp_norm_grad;
    radius_grad.setZero(); radius_grad(dim) = 1;
    cp_norm_grad.setZero();
    // Avoid division-by-zero.
    const real eps = Epsilon();
    if (cp_norm <= eps) cp_norm += eps;
    cp_norm_grad.head(dim) = (center_ - p) / cp_norm;

    grad = ToStdVector(radius_grad - cp_norm_grad);
    return dist;
}

template<int dim>
void Sphere<dim>::InitializeCustomizedData() {
    CheckError(ParametricShape<dim>::param_num() == dim + 1, "Inconsistent number of parameters.");
    for (int i = 0; i < dim; ++i) center_(i) = ParametricShape<dim>::params()[i];
    radius_ = ParametricShape<dim>::params()[dim];
}

template class Sphere<2>;
template class Sphere<3>;