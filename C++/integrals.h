#ifndef INTEGRALS
#define INTEGRALS

#include <cmath>
#include <iostream>
#include <limits>
#include <gaussian.h>
#include <molecule.h>
#include <angular.h>
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib\unsupported\Eigen\CXX11\Tensor>
#include <boost/math/special_functions/gamma.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class MolecularIntegralsS {
public:
    Molecule molecule;
    Eigen::MatrixXd S;
    Eigen::MatrixXd T;
    Eigen::MatrixXd V_ne;
    Eigen::Tensor<double,4> V_ee;
    double E_nn;
    Eigen::MatrixXd H_core;
public:
    MolecularIntegralsS(std::vector<std::vector<std::vector<double>>> alphas, std::vector<std::vector<double>> centers, std::vector<std::vector<std::vector<int>>> l, std::vector<int> Z, std::vector<std::vector<std::vector<double>>> coefficients) ; 
    void overlap_matrix();
    void kinetic_matrix();
    void electron_nuclear_matrix();
    void electron_electron_repulsion_matrix();
    void nuclear_nuclear_repulsion();
};
#endif // INTEGRALS