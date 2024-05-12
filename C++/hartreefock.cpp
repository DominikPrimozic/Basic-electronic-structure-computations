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
#include <reader.h>
#include <chrono>

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


class HF {
public:
    Eigen::MatrixXd F;
    Eigen::MatrixXd C;
    Eigen::VectorXd o_ene;
    Eigen::MatrixXd D;
    double E_ee;
    double E_total;
    
public:
    HF(const MolecularIntegralsS& molint, double convergence, int max_steps) 
        {   
          SCF(molint, convergence,max_steps);
          total_energy(molint);
        }
    void fock_matrix(const MolecularIntegralsS& molint) { //added const and & here, not speed up, tried to add it above, nothing
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
        /* //moved -0.5 * D(r,v)*molint.V_ee(p,r,q,v) to loop above, saved half a milisecond with water
        for (int p=0;p<dim[0];++p){
            for (int r=0;r<dim[1];++r){
                for (int q=0;q<D.rows();++q){
                    for (int v=0;v<D.cols();++v){
                        F(p,q)+=-0.5 * D(r,v)*molint.V_ee(p,r,q,v);

                    }
                }
            }   
        }*/

    }

    void orbital_coefficients(MolecularIntegralsS molint) {
        
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(F,molint.S);
        o_ene=es.eigenvalues();
        C=es.eigenvectors();
        
    }

    void density_matrix(MolecularIntegralsS molint) {
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

    void SCF(MolecularIntegralsS molint, double convergence, int max_steps) {
        // Initial guess
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(molint.H_core,molint.S);
        C=es.eigenvectors();

        density_matrix(molint);
    
        double E0 = compute_electronic_energy(D, molint.H_core, molint.H_core);
        double E;
 
        for (int step = 0; step < max_steps; ++step) 
        {
            auto start1 = std::chrono::high_resolution_clock::now();
            fock_matrix(molint);
            auto stop1 = std::chrono::high_resolution_clock::now();
            auto duration1 = duration_cast<std::chrono::microseconds>(stop1 - start1);
            std::cout<<duration1.count()<<std::endl;

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

    void total_energy(MolecularIntegralsS molint) {
        E_total= E_ee + molint.E_nn;
    }
};




int main(){ 

    std::string path;
    std::cout << "Path to input: ";
    std::cin >> path;
    

    auto [Z,centers,alphas,coefficients,l]=inputer(path); 
    double convergence=1e-10;
    int maxstep=2000;

    auto start = std::chrono::high_resolution_clock::now();

    MolecularIntegralsS molS(alphas, centers,l, Z, coefficients); 
   
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::seconds>(stop - start);

    
    auto startHF = std::chrono::high_resolution_clock::now();

    HF hartreefock(molS,convergence,maxstep); 
    std::cout<<std::setprecision(17) << hartreefock.E_total << std::endl;

    auto stopHF = std::chrono::high_resolution_clock::now();
    auto durationHF = duration_cast<std::chrono::milliseconds>(stopHF - startHF);

    ofstream myfile;
    myfile.open("HFresult.txt");
    myfile << "Total energy is "<<std::setprecision(17)<<hartreefock.E_total<<" hartree\n";
    myfile << "Integrals took: "<<duration.count()<<" seconds\n";
    myfile << "hartree-fock took: "<<durationHF.count()<<" milliseconds";
    myfile.close();
    return 0;
   
} 