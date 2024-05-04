#ifndef EXCO_H
#define EXCO_H
#include <cmath>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#include <eigen_lib/Eigen/Dense>
#include <cmath>

using namespace Eigen;

VectorXd wigner_seitz_r(VectorXd n);
VectorXd Vexchange_LDA(VectorXd n);
VectorXd Vcorrelation_LDA(VectorXd rs);
VectorXd Eexchange_LDA(VectorXd n);
VectorXd Ecorrelation_LDA(VectorXd rs);

#endif // EXCO_H