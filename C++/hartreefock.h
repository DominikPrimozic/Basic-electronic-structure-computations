#ifndef HartreeFock
#define HartreeFock

#include <cmath>
#include <iostream>
#include <limits>
#include <gaussian.h>
#include <molecule.h>
#include <angular.h>
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib/Eigen/Eigenvalues>
#include <eigen_lib\unsupported\Eigen\CXX11\Tensor>
#include <boost/math/special_functions/gamma.hpp>
#include <integrals.h>

class HF {
public:
    Eigen::MatrixXd F;
    Eigen::MatrixXd C;
    Eigen::VectorXd o_ene;
    Eigen::MatrixXd D;
    double E_ee;
    double E_total;
public:
     HF(const MolecularIntegralsS& molint, double convergence, int max_steps) ;
     void fock_matrix(const MolecularIntegralsS& molint);
     void orbital_coefficients(MolecularIntegralsS molint);
     void density_matrix(MolecularIntegralsS molint);
     void SCF(MolecularIntegralsS molint, double convergence, int max_steps);
     void total_energy(MolecularIntegralsS molint);

};
#endif // HartreeFock