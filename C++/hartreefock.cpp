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
#include <diis.h>

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
    HF(MolecularIntegralsS molint, double convergence, int max_steps) 
        {   
          SCF(molint, convergence,max_steps);
          total_energy(molint);
        }
    void fock_matrix(MolecularIntegralsS molint) {
        F=molint.H_core;
        auto dim =molint.V_ee.dimensions();
        //Drv V_eepqrv
        for (int p=0;p<dim[0];++p){
            for (int q=0;q<dim[1];++q){
                for (int r=0;r<D.rows();++r){
                    for (int v=0;v<D.cols();++v){
                        F(p,q)+=D(r,v)*molint.V_ee(p,q,r,v);

                    }
                }
            }   
        }

        for (int p=0;p<dim[0];++p){
            for (int r=0;r<dim[1];++r){
                for (int q=0;q<D.rows();++q){
                    for (int v=0;v<D.cols();++v){
                        F(p,q)+=-0.5 * D(r,v)*molint.V_ee(p,r,q,v);

                    }
                }
            }   
        }

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
    /*
        for (int i = 0; i < molint.molecule.n_occ; ++i) {
        D += 2 * C.col(i) * C.col(i).transpose();
    }*/
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
        //std::cout << C << std::endl;
        density_matrix(molint);
        //std::cout << "\n" << std::endl;
        //std::cout << "D"<<D << std::endl;    
        double E0 = compute_electronic_energy(D, molint.H_core, molint.H_core);
        double E;
        //diis ds;
        for (int step = 0; step < max_steps; ++step) 
        {
            fock_matrix(molint);
            orbital_coefficients(molint);
            density_matrix(molint);
            //ds.update(F,D,molint.S);
            //std::cout << "ds made" << std::endl;
            //if (step>1){
            //    this->F=ds.DIIS_F();
            //    std::cout << "Fmade?" << std::endl;
            //}
            //std::cout <<"\nC"<< C << std::endl;
            //std::cout <<"\nD"<< D << std::endl;
            //std::cout <<"\nF"<< F << std::endl;
            //std::cout << "energy?" << std::endl;
            E = compute_electronic_energy(D, molint.H_core, F);

            std::cout << E << std::endl;
            //double rd=ds.RMSD_check();
            //std::cout << rd << std::endl;

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
    /*
    std::vector<std::vector<std::vector<double>>> alphas={{{0.8021420155E+01,0.1467821061E+01,0.4077767635E+00,0.1353374420E+00}},{{0.8021420155E+01,0.1467821061E+01,0.4077767635E+00,0.1353374420E+00}}};
     std::vector<std::vector<double>> centers = {{0,0,0},{0,0,1.4}};
      std::vector<std::vector<std::vector<int>>> l= {{{0, 0, 0}}, {{0, 0, 0}}};
      std::vector<int> Z={1,1};
       std::vector<std::vector<std::vector<double>>> coefficients={{{0.5675242080E-01,0.2601413550E+00,0.5328461143E+00,0.2916254405E+00}},{{0.5675242080E-01,0.2601413550E+00,0.5328461143E+00,0.2916254405E+00}}};
    /**/
    std::vector<std::vector<std::vector<double>>> alphas = {
        {{0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00}},
        {{0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00}},
        {
            {0.1307093214E+03, 0.2380886605E+02, 0.6443608313E+01},
            {0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00},
            {0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00},
            {0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00},
            {0.5033151319E+01, 0.1169596125E+01, 0.3803889600E+00}
        }
    };
    std::vector<std::vector<double>> centers = {
        {0, 1.4305227, 1.1092692},
        {0, -1.4305227, 1.1092692},
        {0, 0, 0}
    };
    std::vector<std::vector<std::vector<double>>> coefficients = {
        {{0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00}},
        {{0.1543289673E+00, 0.5353281423E+000, 0.4446345422E+00}},
        {
            {0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00},
            {-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00},
            {0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00},
            {0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00},
            {0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00}
        }
    };
    std::vector<int> Z = {1, 1, 8};
    std::vector<std::vector<std::vector<int>>> l = {
        {{{0, 0, 0}}},
        {{{0, 0, 0}}},
        {
            {{0, 0, 0}},
            {{0, 0, 0}},
            {{1, 0, 0}},
            {{0, 1, 0}},
            {{0, 0, 1}}
        }
    };
    /**/
    MolecularIntegralsS molS(alphas, centers,l, Z, coefficients);
    /*
    std::cout<< std::setprecision(5) << molS.S << std::endl;
    std::cout << "\n" << std::endl;
    std::cout << molS.T << std::endl;
    std::cout << "\n" << std::endl;
    std::cout<< std::setprecision(5) << molS.V_ne << std::endl;
    std::cout << "\n" << std::endl;
    std::cout<< std::setprecision(5) << molS.E_nn << std::endl;
    std::cout << "\n" << std::endl;
    std::cout <<std::setprecision(17) << molS.V_ee.sum() << std::endl;
    std::cout << "\n" << std::endl;
    return 0;
    */
   //oscillates, need DIIS
   HF hartreefock(molS,1e-10,200);
   std::cout<<std::setprecision(17) << hartreefock.E_total << std::endl;
}