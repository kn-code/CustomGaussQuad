# CustomGaussQuad

_CustomGaussQuad_ is a C++20 library for computing Gaussian quadrature rules for arbitrary (real, positive) weight functions. This means that given a fixed weight function $w(x)\geq0$, this library computes _optimal_ weights $w_i$ and abscissae $x_i$ such that

$$ \int_a^b w(x) f(x) dx \approx \sum_{i=1}^n w_i f(x_i). $$

The weights and abscissae are optimal in the sense that the degree of exactness is _maximal_: If $f(x) \in \text{span}(1,x,x^2,\dots,x^{2n-1})$, then the approximation is exact. See the [Wikipedia entry](https://en.wikipedia.org/wiki/Gaussian_quadrature) for more information.

## Basic usage

Say we are interested in computing integrals of the form

$$ \int_0^1 \log(x)^2 f(x) dx $$

for many different smooth functions $f(x)$. In this example $w(x)=\log(x)^2$ is our weight function. We can compute a Gaussian quadrature rule as follows:

```cpp
#include <cmath>
#include <iomanip>
#include <iostream>

#include <custom_gauss_quad.h>

using std::log, std::pow, std::sin;
using namespace CustomGaussQuad;

int main()
{
    // Weight function
    auto w = [](double x) { return pow(log(x), 2); };

    // Compute optimal quadrature rule with 20 weights/abscissae for integrals
    // of the form \int_0^1 w(x) f(x) dx.
    // Internally a non-optimal rule with 100 points is used for the computation.
    QuadRule gauss = computeGaussRule(w, 20, 0.0, 1.0, 100);

    // Example: integrate log(x)^2 * sin(10*x)
    // Exact result from Mathematica: 5/2 HypergeometricPFQ[{1,1,1},{3/2,2,2,2},-25]
    double result = gauss.integrate([](double x){ return sin(10*x); });
    double exact  = 0.74526762034248593146;

    std::cout << std::setprecision(16)
              << "result = " << result << "\n"
              << "error  = " << result-exact << "\n";

    return 0;
}
```

```sh
result = 0.7452676203424863
error  = 3.33066907387547e-16
```

With the Gaussian rule in hand we just need 20 evaluations to compute the integral with an error of order $\approx 10^{-16}$. For comparison, we also integrate the same function using ordinary adaptive quadrature as implemented in `scipy.integrate.quad`:

```python
import numpy as np
from scipy import integrate
 
count = 0
def f(x):
    global count
    count += 1
    return pow(np.log(x), 2)*np.sin(10*x)

exact = 0.74526762034248593146
I = integrate.quad(f, a=0, b=1, epsrel=1e-15, epsabs=1e-15, limit=1000)
print('result =', I[0])
print('error  =', I[0]-exact)
print('Computed with', count, 'function evaluations.')
```
```
result = 0.7452676203424862
error  = 2.220446049250313e-16
Computed with 651 function evalations.
```

Now to achieve a comparable error 651 function evaluations are needed!

## Installing

The library consists of a single header, and thus nothing has to be compiled. Tests can be compiled and run with

```console
$ meson build
$ cd build/
$ ninja test
```

## Dependencies

This library uses some C++-20 features and therefore needs a not too old compiler. The [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page) is used for linear algebra. Tests are implemented using [μt](https://github.com/boost-ext/ut/). If the Eigen library cannot be found by meson when building the tests, it will be automatically downloaded.


## References

1. A. D. Fernandes, W. R. Atchley. Gaussian Quadrature Formulae for Arbitrary Positive Measures. Evolutionary Bioinformatics. 2006;2. [doi:10.1177/117693430600200010](https://journals.sagepub.com/doi/10.1177/117693430600200010)

2. R. M. Parrish. Simple Computation of the MultiExp Gaussian Quadrature in Double Precision. [arXiv:2305.01621](https://arxiv.org/abs/2305.01621)

## License

This library is licensed under the [Mozilla Public License (MPL) version 2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).