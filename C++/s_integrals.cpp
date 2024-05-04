#include <cmath>
#include <iostream>
#include <limits>
#include <gaussian.h>
#include <molecule.h>
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib\unsupported\Eigen\CXX11\Tensor>
#include <boost/math/special_functions/gamma.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/*
double factorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n * factorial(n - 2);
}

double boys(double x, int n) {
    if (x == 0) {
        return 1.0 / (2 * n + 1);
    } else {
        return boost::math::tgamma_lower(n+0.5,x)/(2*std::pow(x,n+0.5));
    }
}

double overlap(Gaussian gi, Gaussian gj) {
    double ai = gi.a;
    double aj = gj.a;
    double overlap = gi.N * gj.N * gi.c * gj.c * std::pow(M_PI, 1.5) / std::pow(ai + aj, 1.5);
    overlap *= std::exp(-ai * aj / (ai + aj) * (std::pow(gi.r[0] - gj.r[0], 2) + std::pow(gi.r[1] - gj.r[1], 2) + std::pow(gi.r[2] - gj.r[2], 2)));
    return overlap;
}

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
    MolecularIntegralsS(std::vector<std::vector<std::vector<double>>> alphas, std::vector<std::vector<double>> centers, std::vector<std::vector<std::vector<int>>> l, std::vector<int> Z, std::vector<std::vector<std::vector<double>>> coefficients) 
        : molecule(alphas, centers, l, Z, coefficients) {   
        // Initialize molecule
        //Molecule molecule(alphas, centers,l, Z, coefficients);
        // Calculate integrals
        //overlap_matrix();
        //kinetic_matrix();
        //electron_nuclear_matrix();
        //electron_electron_repulsion_matrix();
        //nuclear_nuclear_repulsion();
        //H_core = T + V_ne; //need to reshape H_core i think
    }

    void overlap_matrix() {
        int n_mol_basis =molecule.mol.size();
        
        S.resize(n_mol_basis, n_mol_basis);
        S.setZero();

        for (int i = 0; i < n_mol_basis; ++i) {
            for (int j = 0; j < n_mol_basis; ++j) {
                int n_exp_basis_i = molecule.mol[i].size();
                int n_exp_basis_j = molecule.mol[j].size();
                for (int p = 0; p < n_exp_basis_i; ++p) {
                    for (int q = 0; q < n_exp_basis_j; ++q) {
                        
                        S(i,j) += overlap(molecule.mol[i][p], molecule.mol[j][q]);
                    }
                }
            }
        }
    }

    void kinetic_matrix() {
    int n_mol_basis = molecule.mol.size();
    T.resize(n_mol_basis, n_mol_basis);

    for (int i = 0; i < n_mol_basis; ++i) {
        for (int j = 0; j < n_mol_basis; ++j) {
            int n_exp_basis_i = molecule.mol[i].size();
            int n_exp_basis_j = molecule.mol[j].size();

            for (int p = 0; p < n_exp_basis_i; ++p) {
                for (int q = 0; q < n_exp_basis_j; ++q) {
                    double Spq = overlap(molecule.mol[i][p], molecule.mol[j][q]);

                    double ap = molecule.mol[i][p].a;
                    double aq = molecule.mol[j][q].a;
                    std::vector<double> Rp = molecule.mol[i][p].r;
                    std::vector<double> Rq = molecule.mol[j][q].r;

                    double Rpq = 0.0;
                    for (size_t k = 0; k < Rp.size(); ++k) {
                        Rpq += (ap * Rp[k] + aq * Rq[k]) / (ap + aq);
                    }

                    double term = ap * aq / (ap + aq) * (3 - 2 * ap * aq / (ap + aq) * (std::pow(Rp[0] - Rq[0], 2) + std::pow(Rp[1] - Rq[1], 2) + std::pow(Rp[2] - Rq[2], 2)));
                    T(i,j) += term * Spq;
                }
            }
        }
    }
}

void electron_nuclear_matrix() {
    int n_mol_basis = molecule.mol.size();
    int n_atoms = molecule.charges.size();
    V_ne.resize(n_mol_basis, n_mol_basis);

    for (int atom = 0; atom < n_atoms; ++atom) {
        for (int i = 0; i < n_mol_basis; ++i) {
            for (int j = 0; j < n_mol_basis; ++j) {
                int n_exp_basis_i = molecule.mol[i].size();
                int n_exp_basis_j = molecule.mol[j].size();

                for (int p = 0; p < n_exp_basis_i; ++p) {
                    for (int q = 0; q < n_exp_basis_j; ++q) {

                        double Spq = overlap(molecule.mol[i][p], molecule.mol[j][q]);

                        double ap = molecule.mol[i][p].a;
                        double aq = molecule.mol[j][q].a;
                        std::vector<double> Rp = molecule.mol[i][p].r;
                        std::vector<double> Rq = molecule.mol[j][q].r;
                        
                        std::vector<double> Rpq(3);
                        for (int iii=0;iii<3 ;++iii)
                        {
                            Rpq[iii]=(ap*Rp[iii]+aq*Rq[iii]) /(ap+aq);
                        }
                        
                        
                        std::vector<double> Ra = molecule.position[atom];
                        //std::cout << Ra[0]<<Ra[1]<<Ra[2] << std::endl;
                        double factor = -molecule.charges[atom] * 2 * std::sqrt((ap + aq) / M_PI);
                        std::vector<double> Rapq(3);
                        for (int iii=0;iii<3 ;++iii)
                        {
                            Rapq[iii]=Ra[iii]-Rpq[iii];
                        }
                        double term =(ap+aq) * (Rapq[0]*Rapq[0] + Rapq[1]*Rapq[1] + Rapq[2]*Rapq[2]);
                        V_ne(i,j) += factor * boys(term,0) * Spq;
                    }
                }
            }
        }
    }
}

void electron_electron_repulsion_matrix() {
    int n_mol_basis = molecule.mol.size();
    V_ee.resize(n_mol_basis, n_mol_basis,n_mol_basis,n_mol_basis);
    V_ee.setZero();
    for (int i = 0; i < n_mol_basis; ++i) {
        for (int j = 0; j < n_mol_basis; ++j) {
            for (int k = 0; k < n_mol_basis; ++k) {
                for (int l = 0; l < n_mol_basis; ++l) {
                    int n_exp_basis_i = molecule.mol[i].size();
                    int n_exp_basis_j = molecule.mol[j].size();
                    int n_exp_basis_k = molecule.mol[k].size();
                    int n_exp_basis_l = molecule.mol[l].size();

                    for (int p = 0; p < n_exp_basis_i; ++p) {
                        for (int q = 0; q < n_exp_basis_j; ++q) {
                            for (int r = 0; r < n_exp_basis_k; ++r) {
                                for (int v = 0; v < n_exp_basis_l; ++v)
                                 {
                                    double Spq = overlap(molecule.mol[i][p], molecule.mol[j][q]);
                                    double Srv = overlap(molecule.mol[k][r], molecule.mol[l][v]);

                                    double ap = molecule.mol[i][p].a;
                                    double aq = molecule.mol[j][q].a;
                                    double ar = molecule.mol[k][r].a;
                                    double av = molecule.mol[l][v].a;
                                    std::vector<double> Rp = molecule.mol[i][p].r;
                                    std::vector<double> Rq = molecule.mol[j][q].r;
                                    std::vector<double> Rr = molecule.mol[k][r].r;
                                    std::vector<double> Rv = molecule.mol[l][v].r;

                                    std::vector<double> Rpq(3);
                                    for (int iii=0;iii<3 ;++iii)
                                    {
                                        Rpq[iii]=(ap*Rp[iii]+aq*Rq[iii]) /(ap+aq);
                                    }
                                    std::vector<double> Rrv(3);
                                    for (int iii=0;iii<3 ;++iii)
                                    {
                                        Rrv[iii]=(ar*Rr[iii]+av*Rv[iii]) /(ar+av);
                                    }

                                    std::vector<double> R(3);
                                    for (int iii=0;iii<3 ;++iii)
                                    {
                                        R[iii]=Rpq[iii]-Rrv[iii];
                                    }
                                    double term =(ap+aq)*(ar+av)/(ap+aq+ar+av) * (R[0]*R[0] + R[1]*R[1] + R[2]*R[2]);

                                    double term2 = 2 / std::sqrt(M_PI) * std::sqrt(ap + aq) * std::sqrt(ar + av) / std::sqrt(ap + aq + ar + av);
                                    V_ee(i,j,k,l) +=term2* boys(term,0) * Spq * Srv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


    void nuclear_nuclear_repulsion() {
        double E_nn = 0.0;
        for (size_t atom1 = 0; atom1 < molecule.position.size(); ++atom1) {
        for (size_t atom2 = atom1 + 1; atom2 < molecule.position.size(); ++atom2) {
            // Compute the difference vector between atom1 and atom2
            Eigen::VectorXd diff = Eigen::Map<const Eigen::VectorXd>(molecule.position[atom1].data(), molecule.position[atom1].size()) -
                                   Eigen::Map<const Eigen::VectorXd>(molecule.position[atom2].data(), molecule.position[atom2].size());

            // Compute the distance between atom1 and atom2
            double dist = diff.norm();

            // Add the contribution to the nuclear-nuclear repulsion energy
            E_nn += molecule.charges[atom1] * molecule.charges[atom2] / dist;
        }
    }
        this->E_nn = E_nn;
    }
};
int main(){
    std::vector<std::vector<std::vector<double>>> alphas={{{0.8021420155E+01,0.1467821061E+01,0.4077767635E+00,0.1353374420E+00}},{{0.8021420155E+01,0.1467821061E+01,0.4077767635E+00,0.1353374420E+00}}};
     std::vector<std::vector<double>> centers = {{0,0,0},{0,0,1.4}};
      std::vector<std::vector<std::vector<int>>> l= {{{0, 0, 0}}, {{0, 0, 0}}};
      std::vector<int> Z={1,1};
       std::vector<std::vector<std::vector<double>>> coefficients={{{0.5675242080E-01,0.2601413550E+00,0.5328461143E+00,0.2916254405E+00}},{{0.5675242080E-01,0.2601413550E+00,0.5328461143E+00,0.2916254405E+00}}};
          
    //Molecule molecule(alphas, centers,l, Z, coefficients);
    
    //std::cout << molecule.mol[0][0].a<< std::endl;
    //std::cout << molecule.charges[0]<< std::endl;
    //std::cout << molecule.charges[1]<< std::endl;
    //std::cout << molecule.position[1][2]<< std::endl;
    MolecularIntegralsS molS(alphas,centers,l,Z,coefficients);
    //std::cout << molS.molecule.mol[0][1].N<< std::endl;
    //std::cout << molS.molecule.mol[0][0].l[1]<< std::endl;
    molS.overlap_matrix();
    molS.kinetic_matrix();
    molS.electron_nuclear_matrix();
    molS.electron_electron_repulsion_matrix();
    molS.nuclear_nuclear_repulsion();
    //std::cout << boys(5,0) << std::endl;
    std::cout << molS.E_nn << std::endl;
    return 0;
}
*/