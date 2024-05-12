#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <vector>

class Gaussian {
public:
    double a;
    std::vector<double> r;
    double c;
    std::vector<int> l;
    double N;

public:
    Gaussian(double a, std::vector<double> coords, int x, int y, int z, double coeff = 1);

    double factorial(int n);
};

#endif // GAUSSIAN_H
