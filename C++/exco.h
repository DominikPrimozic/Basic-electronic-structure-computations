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
VectorXd Ecorrelation_WVN(VectorXd rs, int limit = 0);
VectorXd Vcorrelation_WVN(VectorXd rs, int limit = 0);
VectorXd V_correlation(VectorXd rs,std::string functional="MP", int limit=0);
VectorXd E_correlation(VectorXd rs,std::string functional="MP", int limit=0);

#endif // EXCO_H