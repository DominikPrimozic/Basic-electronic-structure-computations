#include <cmath>
#include <iostream>
#include <limits>
#include <gaussian.h>
#include <molecule.h>
#include <angular.h>
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib\unsupported\Eigen\CXX11\Tensor>
#include <boost/math/special_functions/gamma.hpp>
#include <integrals.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif




MolecularIntegralsS::MolecularIntegralsS(std::vector<std::vector<std::vector<double>>> alphas, std::vector<std::vector<double>> centers, std::vector<std::vector<std::vector<int>>> l, std::vector<int> Z, std::vector<std::vector<std::vector<double>>> coefficients) 
        : molecule(alphas, centers, l, Z, coefficients) {   
        // Initialize molecule
        //Molecule molecule(alphas, centers,l, Z, coefficients);
        // Calculate integrals
        overlap_matrix();
        kinetic_matrix();
        electron_nuclear_matrix();
        electron_electron_repulsion_matrix();
        nuclear_nuclear_repulsion();
        H_core = T + V_ne; //need to reshape H_core i think
    }

void MolecularIntegralsS:: overlap_matrix() {
    int n_mol_basis =molecule.mol.size();
    
    S.resize(n_mol_basis, n_mol_basis);
    S.setZero();

    for (int i = 0; i < n_mol_basis; ++i) {
        for (int j = 0; j < n_mol_basis; ++j) {
            int n_exp_basis_i = molecule.mol[i].size();
            int n_exp_basis_j = molecule.mol[j].size();
            for (int p = 0; p < n_exp_basis_i; ++p) {
                for (int q = 0; q < n_exp_basis_j; ++q) {
                    
                    S(i,j) += molecule.mol[i][p].N * molecule.mol[j][q].N * molecule.mol[i][p].c* molecule.mol[j][q].c * overlap(molecule.mol[i][p], molecule.mol[j][q]);
                }
            }
        }
    }
}

void MolecularIntegralsS::kinetic_matrix() {
    int n_mol_basis = molecule.mol.size();
    T.resize(n_mol_basis, n_mol_basis);

    for (int i = 0; i < n_mol_basis; ++i) {
        for (int j = 0; j < n_mol_basis; ++j) {
            int n_exp_basis_i = molecule.mol[i].size();
            int n_exp_basis_j = molecule.mol[j].size();

            for (int p = 0; p < n_exp_basis_i; ++p) {
                for (int q = 0; q < n_exp_basis_j; ++q) {
                    T(i,j) += molecule.mol[i][p].N * molecule.mol[j][q].N * molecule.mol[i][p].c* molecule.mol[j][q].c * kinetic(molecule.mol[i][p], molecule.mol[j][q]);
                }
            }
        }
    }
}

void MolecularIntegralsS::electron_nuclear_matrix() {
    int n_mol_basis = molecule.mol.size();
    int n_atoms = molecule.charges.size();
    V_ne.resize(n_mol_basis, n_mol_basis);

    for (int atom = 0; atom < n_atoms; ++atom) {
        for (int i = 0; i < n_mol_basis; ++i) {
            for (int j = 0; j < n_mol_basis; ++j) {
                std::vector<Gaussian> gto1=molecule.mol[i];
                std::vector<Gaussian> gto2=molecule.mol[j];
                std::vector<int> la=gto1[0].l;
                std::vector<int> lb=gto2[0].l;
                std::vector<double> Ra = molecule.position[atom];

                V_ne(i,j) +=-molecule.charges[atom] * nuclearHRR(gto1,gto2, la,lb,Ra);
               
            }
        }
    }
}

void MolecularIntegralsS::electron_electron_repulsion_matrix() {
int n_mol_basis = molecule.mol.size();
V_ee.resize(n_mol_basis, n_mol_basis,n_mol_basis,n_mol_basis);
V_ee.setZero();
for (int i = 0; i < n_mol_basis; ++i) {
    for (int j = 0; j < n_mol_basis; ++j) {
        for (int k = 0; k < n_mol_basis; ++k) {
            for (int l = 0; l < n_mol_basis; ++l) {
                std::vector<Gaussian> gto1=molecule.mol[i];
                std::vector<Gaussian> gto2=molecule.mol[j];
                std::vector<Gaussian> gto3=molecule.mol[k];
                std::vector<Gaussian> gto4=molecule.mol[l];
                std::vector<int> la=gto1[0].l;
                std::vector<int> lb=gto2[0].l;
                std::vector<int> lc=gto3[0].l;
                std::vector<int> ld=gto4[0].l;
                
                V_ee(i,j,k,l) +=HRR(gto1,gto2,gto3,gto4,la,lb,lc,ld);
                            
                }
            }
        }
    }
}



void MolecularIntegralsS:: nuclear_nuclear_repulsion() {
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


