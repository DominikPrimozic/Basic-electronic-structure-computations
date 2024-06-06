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
#include <hartreefock.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double compute_electronic_energy(Eigen::MatrixXd& D, Eigen::MatrixXd& H, Eigen::MatrixXd& F) {
    double E=0;
    for (int p=0; p<D.rows(); ++p){
        for (int q=0; q<D.cols(); ++q){

            E+=0.5*D(p,q)*( H(p,q)+F(p,q) );
        }

    }

    return E; 
}



    HF::HF(const MolecularIntegralsS& molint, double convergence, int max_steps) 
        {   
          SCF(molint, convergence,max_steps);
          total_energy(molint);
        }
    void HF::fock_matrix(const MolecularIntegralsS& molint) { //added const and & here, not speed up, tried to add it above, nothing
        F=molint.H_core;
        auto dim =molint.V_ee.dimensions();
        //Drv V_eepqrv
        for (int p=0;p<dim[0];++p){
            for (int q=0;q<dim[1];++q){
                for (int r=0;r<D.rows();++r){
                    for (int v=0;v<D.cols();++v){
                        F(p,q)+=D(r,v)*molint.V_ee(p,q,r,v) -0.5 * D(r,v)*molint.V_ee(p,r,q,v);

                    }
                }
            }   
        }
        

    }

    void HF::orbital_coefficients(MolecularIntegralsS molint) {
        
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(F,molint.S);
        o_ene=es.eigenvalues();
        C=es.eigenvectors();
        
    }

    void HF::density_matrix(MolecularIntegralsS molint) {
        D.resize(C.rows(), C.rows());
        D.setZero();
        Eigen::MatrixXd C1=C(Eigen::all,Eigen::seq(0,molint.molecule.n_occ-1));

        for (int p=0; p<C1.rows(); ++p){
            for (int q=0; q<C1.rows(); ++q){
                for (int i=0; i<C1.cols(); ++i)
                {
                    
                    D(p,q)+=2*C1(p,i)*C1(q,i);
                }
            }   
        }
    }

    void HF::SCF(MolecularIntegralsS molint, double convergence, int max_steps) {
        // Initial guess
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(molint.H_core,molint.S);
        C=es.eigenvectors();

        density_matrix(molint);
    
        double E0 = compute_electronic_energy(D, molint.H_core, molint.H_core);
        double E;
 
        for (int step = 0; step < max_steps; ++step) 
        {
            
            fock_matrix(molint);
            orbital_coefficients(molint);
            density_matrix(molint);
            
            E = compute_electronic_energy(D, molint.H_core, F);
        
            std::cout << E << std::endl;

            if (abs(E - E0) < convergence) {
                std::cout << "Converged\n";
                E_ee = E;
                return;
            }
            E0 = E;
        }
        std::cout << "Did not converge\n";
    }

    void HF::total_energy(MolecularIntegralsS molint) {
        E_total= E_ee + molint.E_nn;
    }





