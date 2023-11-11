//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <boost/ut.hpp>

#include <custom_gauss_quad.h>
#include <unsupported/Eigen/MPRealSupport>

using boost::ut::expect;
using boost::ut::lt;
using namespace boost::ut::literals;

boost::ut::suite mpfrQuadratureTest = []
{
    using Real = mpfr::mpreal;
    mpfr::mpreal::set_default_prec(std::ceil(100 * 3.33));

    "Integrate moments of w(x)=log(x)^2"_test = []
    {
        auto weightFunc = [](Real x)
        {
            return pow(log(x), 2);
        };
        int n      = 50;
        Real a     = 0;
        Real b     = 1;
        Real yMax  = 6;
        auto gauss = CustomGaussQuad::computeGaussRule(weightFunc, n, a, b, 250, yMax);

        // \int_0^1 log(x)^2 * x^k dx
        auto moments = [](int k) -> Real
        {
            return Real(2.0) / pow(Real(k) + 1, 3);
        };

        for (int k = 0; k <= 2 * n - 1; ++k)
        {
            auto f = [k](Real x)
            {
                return pow(x, k);
            };

            Real result = gauss.integrate(f);
            Real exact  = moments(k);

            Real epsilon("5e-99");
            expect(lt(abs(result - exact), epsilon)) << "k =" << k;
        }
    };
};

int main()
{
    return 0;
}