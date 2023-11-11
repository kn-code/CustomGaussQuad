//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef CUSTOM_GAUSS_QUAD_H
#define CUSTOM_GAUSS_QUAD_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace CustomGaussQuad
{

// Matrix of the form
//  a0, a1, a2, a3, ...
//  a1, b0,  0,  0, ...
//  a2,  0, b1,  0, ...
//  a3,  0,  0, b2, ...
//  ...
template <typename Scalar_>
struct WeightMatrix
{
    using Scalar = Scalar_;
    using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

    Vector a;
    Vector b;

    int rows() const noexcept
    {
        return a.rows();
    }

    int cols() const noexcept
    {
        return a.cols();
    }

    Vector operator*(const Vector& x) const
    {
        int n = a.size();
        assert(n > 0);
        assert(n == b.size() + 1);
        assert(x.size() == n);

        Vector returnValue(n);
        returnValue[0]          = a.adjoint() * x;
        returnValue.tail(n - 1) = a.tail(n - 1) * x[0] + b.cwiseProduct(x.tail(n - 1));
        return returnValue;
    };
};

template <typename Scalar>
struct LanczosResult
{
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

    Vector alpha;
    Vector beta;
    Matrix V;

    Matrix tridiagonalMatrix() const
    {
        Matrix returnValue       = Matrix::Zero(alpha.size(), alpha.size());
        returnValue.diagonal()   = alpha;
        returnValue.diagonal(1)  = beta;
        returnValue.diagonal(-1) = beta;
        return returnValue;
    }
};

enum Reorthogonalization
{
    None,
    // Partial, TODO implement
    Full
};

template <int Reorthogonalization_, typename MatrixType>
LanczosResult<typename std::decay_t<MatrixType>::Scalar> lanczos(MatrixType&& A, int n)
{
    using Scalar = typename std::decay_t<MatrixType>::Scalar;
    using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    int dimA = A.rows();

    assert(dimA > 1);
    assert(n > 0);
    assert(n <= dimA);

    Vector alpha(n);
    Vector beta(n - 1);
    Matrix V(dimA, n);

    V.col(0) = Vector::Zero(dimA);
    V(0, 0)  = 1.0;

    Vector w = A * V.col(0);
    alpha[0] = w.adjoint() * V.col(0);
    w        -= alpha[0] * V.col(0);

    for (int i = 1; i < n; ++i)
    {
        beta[i - 1] = w.norm();
        assert(beta.isZero() == false);

        V.col(i) = w / beta[i - 1];

        // Modified Gram-Schmidt to orthogonalize V.col(i) wrt. all previous columns
        if constexpr (Reorthogonalization_ == Reorthogonalization::Full)
        {
            for (int j = 0; j < i; ++j)
            {
                V.col(i) -= (V.col(j).adjoint() * V.col(i)) * V.col(j);
            }
            V.col(i) /= V.col(i).norm();
        }

        w        = A * V.col(i);
        alpha[i] = w.adjoint() * V.col(i);
        w        -= alpha[i] * V.col(i) + beta[i - 1] * V.col(i - 1);
    }

    return LanczosResult<Scalar>{.alpha = std::move(alpha), .beta = std::move(beta), .V = std::move(V)};
}

template <typename Scalar, int Size, typename WeightFunction>
WeightMatrix<Scalar> constructWeightMatrix(
    const Eigen::Vector<Scalar, Size>& x,
    const Eigen::Vector<Scalar, Size>& w,
    WeightFunction&& weightFunc)
{
    using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

    int n = x.size();
    assert(w.size() == n);

    WeightMatrix<Scalar> returnValue;
    returnValue.a = Vector::Zero(n + 1);
    returnValue.b = x;

    returnValue.a[0] = 1.0;
    for (int i = 0; i < n; ++i)
    {
        returnValue.a[i + 1] = w[i] * weightFunc(x[i]);
    }

    returnValue.a = returnValue.a.cwiseSqrt();

    return returnValue;
}

template <typename Scalar>
struct QuadRule
{
    using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

    Vector weights;
    Vector abscissae;

    template <typename Func>
    auto integrate(Func&& f)
    {
        typename std::invoke_result<Func, Scalar>::type returnValue = weights[0] * f(abscissae[0]);
        for (int i = 1; i < weights.size(); ++i)
        {
            returnValue += weights[i] * f(abscissae[i]);
        }
        return returnValue;
    }

    template <typename Scalar2>
    QuadRule<Scalar2> cast() const
    {
        QuadRule<Scalar2> returnValue;
        returnValue.weights.resize(weights.size());
        returnValue.abscissae.resize(abscissae.size());

        for (int i = 0; i < weights.size(); ++i)
        {
            returnValue.weights[i] = static_cast<Scalar2>(weights[i]);
        }

        for (int i = 0; i < abscissae.size(); ++i)
        {
            returnValue.abscissae[i] = static_cast<Scalar2>(abscissae[i]);
        }

        return returnValue;
    }
};

// Returns a double exponential quadrature rule with 2*N+1 points for I = \int_a^b dx f(x).
// The integral is transformed using x(y) = 0.5*(b+a) + 0.5*(b-a)*tanh(sinh(y)).
// leading to I = \int_{-\infty}^{\infty} dy dx/dy(y) f(x(y))
// and approximated using 2N+1 samples of y \in [-yMax, yMax].
// As a rule of thumb, if f(x) only has mild singularities at the endpoints (logarithmic),
// then typically yMax=4.0 is sufficient (in double precision, see Numerical Recipes).
// Stronger singularities requires increasing yMax.
template <typename Scalar>
QuadRule<Scalar> computeDoubleExponentialRule(Scalar a, Scalar b, int N, Scalar yMax)
{
    using std::cosh;
    using std::exp;
    using std::sinh;

    Eigen::Vector<Scalar, Eigen::Dynamic> weights(2 * N + 1);
    Eigen::Vector<Scalar, Eigen::Dynamic> abscissae(2 * N + 1);

    Scalar qk, dk;
    Scalar delta = yMax / N;
    weights[0]   = 0.5 * delta * (b - a);
    abscissae[0] = 0.5 * (b + a);
    for (int k = 0; k < N; ++k)
    {
        qk                   = exp(-2 * sinh((k + 1) * delta));
        dk                   = (b - a) * qk / (1 + qk);
        weights[2 * k + 1]   = 2 * delta * dk / (1 + qk) * cosh((k + 1) * delta);
        weights[2 * k + 2]   = weights[2 * k + 1];
        abscissae[2 * k + 1] = a + dk;
        abscissae[2 * k + 2] = b - dk;
    }

    return QuadRule<Scalar>{.weights = std::move(weights), .abscissae = std::move(abscissae)};
}

// Main algorithm described in https://arxiv.org/abs/2305.01621
// The main difference is that we are using a double exponential "base grid" instead of Gauss-Legendre points
// to compute the quadrature rule. This seems to perform better for singular integrands.
template <typename Scalar, int Size, typename WeightFunction>
QuadRule<Scalar> computeGaussRule(
    WeightFunction&& weightFunc,
    int n,
    const Eigen::Vector<Scalar, Size>& w,
    const Eigen::Vector<Scalar, Size>& x,
    bool normalizeWeights)
{
    using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    assert(n > 0);
    assert(n <= x.size());

    WeightMatrix<Scalar> W = constructWeightMatrix(x, w, std::forward<WeightFunction>(weightFunc));
    auto lanczosResult     = lanczos<Reorthogonalization::Full>(W, n + 1);
    Scalar b0              = lanczosResult.beta[0] * lanczosResult.beta[0];

    Eigen::SelfAdjointEigenSolver<Matrix> es;
    es.computeFromTridiagonal(lanczosResult.alpha.tail(n), lanczosResult.beta.tail(n - 1), Eigen::ComputeEigenvectors);

    Vector weights   = es.eigenvectors().row(0).transpose();
    weights          = weights.cwiseProduct(weights);
    weights          *= b0;
    Vector abscissae = es.eigenvalues();

    if (normalizeWeights == true)
    {
        for (int i = 0; i < weights.size(); ++i)
        {
            weights[i] /= weightFunc(abscissae[i]);
        }
    }

    return QuadRule<Scalar>{std::move(weights), std::move(abscissae)};
}

template <typename Scalar, typename WeightFunction>
QuadRule<Scalar> computeGaussRule(
    WeightFunction&& weightFunc,
    int n,
    Scalar a,
    Scalar b,
    int N,
    Scalar yMax           = 4.0,
    bool normalizeWeights = false)
{
    QuadRule<Scalar> doubleExpRule = computeDoubleExponentialRule(a, b, N, yMax);
    return computeGaussRule(
        std::forward<WeightFunction>(weightFunc), n, doubleExpRule.weights, doubleExpRule.abscissae, normalizeWeights);
}

} // namespace CustomGaussQuad

#endif // CUSTOM_GAUSS_QUAD_H
