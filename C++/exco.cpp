#include "exco.h"
#define _USE_MATH_DEFINES 
#include <cmath>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#include <eigen_lib/Eigen/Dense>
#include <cmath>

using namespace Eigen;


    VectorXd wigner_seitz_r(VectorXd n) {
        n.array() += (n.array() == 0).cast<double>() * 1e-120;
        return pow(3.0 / (4.0 * M_PI * n.array()), 1.0 / 3.0);
    }

    VectorXd Vexchange_LDA(VectorXd n) {
        return -pow(3.0 * n.array() / M_PI, 1.0 / 3.0);
    }

    VectorXd Vcorrelation_LDA(VectorXd rs) {
    VectorXd e_c = Ecorrelation_LDA(rs);
    double a = (log(2) - 1) / (2 * M_PI * M_PI);
    double b = 20.4562557;
    VectorXd de_c = -a * b * (rs.array() + 2) / (rs.array() * (rs.array() * rs.array() + b * rs.array() + b));
    return e_c.array() - (1.0 / 3.0) * rs.array() * de_c.array();
}

    VectorXd Eexchange_LDA(VectorXd n) {
    return -3.0 / 4.0 * pow(3.0 * n.array() / M_PI, 1.0 / 3.0);
}

    VectorXd Ecorrelation_LDA(VectorXd rs) {
    double a = (log(2) - 1) / (2 * M_PI * M_PI);
    double b = 20.4562557;
    return a * log(1 + b / rs.array() + b / (rs.array() * rs.array()));
}
