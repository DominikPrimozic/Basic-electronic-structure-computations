#ifndef densityfunctional
#define densityfunctional

#include <iostream>
#include <vector>
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib/Eigen/Eigenvalues>
#include <boost/math/special_functions/gamma.hpp>
#include <integrals.h>
#include <diis.h>
#include <exco.h>
#include <grid.h>


class DFT {
public:
    int n_occ;
    Eigen::MatrixXd ao;
    Eigen::MatrixXd C;
    Eigen::MatrixXd D;
    Eigen::VectorXd rho;
    Eigen::MatrixXd V_xc;
    double E_xc;
    std::vector<double> gridx;
    std::vector<double> gridy;
    std::vector<double> gridz;
    std::vector<double> weights;
    double Eks;
    double E_ee;
    double E_total;
    double dipoleX,dipoleY,dipoleZ;
    Eigen::MatrixXd Oij;
public:
    DFT(MolecularIntegralsS molint, std::tuple<double, double, int> scf_param, int ty=1, std::string functional="MP", int limit=0);
    void get_grid(MolecularIntegralsS molint,int ty);
    void grid_atomic_orbitals(std::vector<std::vector<Gaussian>> basis, std::vector<double> gridx,std::vector<double> gridy,std::vector<double> gridz);
    Eigen::VectorXd evaluate_density();
    Eigen::MatrixXd matrix_xc_potential(Eigen::VectorXd vxc);
    double xc_energy(Eigen::VectorXd exc);
    void density_matrix();
    void SCF(std::tuple<double, double, int> scf_param,MolecularIntegralsS molint, std::string functional, int limit);
    void total_energy(MolecularIntegralsS molint);
    void dipole();
    void quadrupole();
    Eigen::VectorXd nucleardipole(MolecularIntegralsS molint);
    Eigen::MatrixXd nuclearquadrupole(MolecularIntegralsS molint);

};








#endif //densityfunctional