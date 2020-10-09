#include "shape/plane.h"
#include "common/common.h"

template<int dim>
const real Plane<dim>::ComputeSignedDistanceAndGradients(const std::array<real, dim>& point,
    std::vector<real>& grad) const {
    Eigen::Matrix<real, dim, 1> p;
    for (int i = 0; i < dim; ++i) p(i) = point[i];
    const real normal_len = normal_.norm();
    const real eps = std::numeric_limits<real>::epsilon();
    CheckError(normal_len > eps, "Singular normal length from the plane equation.");
    const real f = p.dot(normal_) + offset_;
    const real g = normal_len;
    const real dist = f / g;

    // Compute gradients.
    grad.clear();
    grad.resize(dim + 1);

    // Gradient w.r.t. normal.
    // Redefine dist = f / g.
    // dist' = (f' g - f g') / g^2
    Eigen::Matrix<real, dim + 1, 1> f_grad, g_grad;
    f_grad.head(dim) = p;
    f_grad(dim) = 1;
    g_grad.head(dim) = normal_ / normal_len;
    g_grad(dim) = 0;

    // Gradient w.r.t. offset.
    grad = ToStdVector((f_grad * g - f * g_grad) / (g * g));
    return dist;
}

template<int dim>
void Plane<dim>::InitializeCustomizedData() {
    CheckError(ParametricShape<dim>::param_num() == dim + 1, "Inconsistent number of parameters.");
    for (int i = 0; i < dim; ++i) normal_(i) = ParametricShape<dim>::params()[i];
    offset_ = ParametricShape<dim>::params()[dim];
}

template class Plane<2>;
template class Plane<3>;