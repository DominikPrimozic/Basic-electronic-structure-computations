#include "gaussian.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Gaussian::Gaussian(double alpha, std::vector<double> coords, int x, int y, int z, double coeff)
    : a(alpha), r(coords), c(coeff), l({x, y, z}) {
    std::vector<int> l={x,y,z};
    this->l = l;
    double N = std::pow(2 * alpha / M_PI, 3.0 / 4.0);
    double up = std::pow(4 * alpha, (x + y + z) / 2.0);
    double down = std::sqrt(factorial(2 * x - 1) * factorial(2 * y - 1) * factorial(2 * z - 1));
    this->N = N * up / down;

}

double Gaussian::factorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n * factorial(n - 2);
}
