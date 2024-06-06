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



VectorXd Ecorrelation_WVN(VectorXd rs, int limit) {
   
    double a, x0, b, c, b1, b2, b3;
    if (limit == 0) {
        a = 0.0621814;
        x0 = -0.409286;
        b = 13.0720;
        c = 42.7198;
        b1 = (b * x0 - c) / (c * x0);
        b2 = (x0 - b) / (c * x0);
        b3 = -1 / (c * x0);
    } else if (limit == 1) {
        a = 0.0310907;
        x0 = -0.743294;
        b = 20.1231;
        c = 101.578;
        b1 = (b * x0 - c) / (c * x0);
        b2 = (x0 - b) / (c * x0);
        b3 = -1 / (c * x0);
    } else {
        throw std::invalid_argument("limit must be 0 or 1");
    }

    
    VectorXd Xr = rs.array() + b * rs.array().sqrt() + c;
    double Xx = x0 * x0 + b * sqrt(x0 * x0) + c;
    double Q = sqrt(4 * c - b * b);
    VectorXd t2 = Q / (2 * rs.array().sqrt() + b);
    VectorXd t1 = (log((rs.array().sqrt() - x0).square() / Xr.array())
                    + 2 * (b + 2 * x0) / Q * atan(t2.array())).matrix();
    VectorXd t = (log(rs.array() / Xr.array())
                    + 2 * b / Q * atan(t2.array())
                    - b * x0 * t1.array() / Xx).matrix();

    
    return a * t;
}

VectorXd Vcorrelation_WVN(VectorXd rs, int limit) {
    
    double a, x0, b, c, b1, b2, b3;
    if (limit == 0) {
        a = 0.0621814;
        x0 = -0.409286;
        b = 13.0720;
        c = 42.7198;
        b1 = (b * x0 - c) / (c * x0);
        b2 = (x0 - b) / (c * x0);
        b3 = -1 / (c * x0);
    } else if (limit == 1) {
        a = 0.0310907;
        x0 = -0.743294;
        b = 20.1231;
        c = 101.578;
        b1 = (b * x0 - c) / (c * x0);
        b2 = (x0 - b) / (c * x0);
        b3 = -1 / (c * x0);
    } else {
        throw std::invalid_argument("limit must be 0 or 1");
    }

    
    VectorXd Xr = rs.array() + b * rs.array().sqrt() + c;
    double Xx = x0 * x0 + b * sqrt(x0 * x0) + c;
    double Q = sqrt(4 * c - b * b);
    VectorXd t2 = Q / (2 * rs.array().sqrt() + b);
    VectorXd t1 = (log((rs.array().sqrt() - x0).square() / Xr.array())
                    + 2 * (b + 2 * x0) / Q * atan(t2.array())).matrix();
    VectorXd t = (log(rs.array() / Xr.array())
                    + 2 * b / Q * atan(t2.array())
                    - b * x0 * t1.array() / Xx).matrix();

    
    return Ecorrelation_WVN(rs, limit).array() - a / 3 * (1 + b1 * rs.array().sqrt())/(1 + b1 * rs.array().sqrt() + b2 * rs.array() + b3 * rs.array().pow(3.0/2.0));
}

VectorXd V_correlation(VectorXd rs,std::string functional, int limit){
    if (functional=="MP"){return Vcorrelation_LDA(rs);}
    else if (functional=="WVN"){return Vcorrelation_WVN(rs,limit);}
    else {throw std::invalid_argument("not defined functional");}
}

VectorXd E_correlation(VectorXd rs,std::string functional, int limit){
    if (functional=="MP"){return Ecorrelation_LDA(rs);}
    else if (functional=="WVN"){return Ecorrelation_WVN(rs,limit);}
    else {throw std::invalid_argument("not defined functional");}
}