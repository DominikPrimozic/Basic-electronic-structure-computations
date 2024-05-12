#include <iostream>
#include <vector>
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib/Eigen/Eigenvalues>
#include <boost/math/special_functions/gamma.hpp>
#include <integrals.h>
#include <diis.h>
#include <exco.h>
#include <grid.h>
#include <reader.h>
#include <chrono>



 Eigen::VectorXd gaussian_grid_eval(Gaussian gaussian,std::vector<double> gridx,std::vector<double> gridy,std::vector<double> gridz){
    Eigen::VectorXd X(gridx.size());
    Eigen::VectorXd Y(gridy.size());
    Eigen::VectorXd Z(gridz.size());
    for (int i=0;i<gridx.size();++i)
    {
        X[i]=std::pow(gridx[i]-gaussian.r[0],gaussian.l[0]) *std::exp(-gaussian.a*std::pow(gridx[i]-gaussian.r[0],2.0));
        Y[i]=std::pow(gridy[i]-gaussian.r[1],gaussian.l[1]) *std::exp(-gaussian.a*std::pow(gridy[i]-gaussian.r[1],2.0));
        Z[i]=std::pow(gridz[i]-gaussian.r[2],gaussian.l[2]) *std::exp(-gaussian.a*std::pow(gridz[i]-gaussian.r[2],2.0));

    }
    Eigen::VectorXd phi(gridx.size());
    for (int j=0;j<gridx.size();++j)
    {
        phi[j]=X[j]*Y[j]*Z[j]*gaussian.N * gaussian.c;
    }
    return phi;
};
Eigen::MatrixXd get_J(Eigen::MatrixXd D, Eigen::Tensor<double,4> V_ee){
    Eigen::MatrixXd J(D.rows(),D.cols());
    J.setZero();
    auto dim =V_ee.dimensions();
    for (int p=0;p<dim[0];++p){
        for (int q=0;q<dim[1];++q){
            for (int r=0;r<dim[2];++r){
                for (int s=0;s<dim[3];++s){
                    J(p,q)+=D(r,s)*V_ee(p,q,r,s);
                }
            }
        }
    }
    return J;
}



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

public:
    DFT(MolecularIntegralsS molint, std::tuple<double, double, int> scf_param, int ty=1, std::string functional="MP", int limit=0) {
        n_occ = molint.molecule.n_occ;
        get_grid(molint,ty);
        grid_atomic_orbitals(molint.molecule.mol, gridx, gridy, gridz);
        SCF(scf_param,molint,functional,limit);
        total_energy(molint);
    }
void get_grid(MolecularIntegralsS molint,int ty){
    auto tup=grid_maker(molint.molecule, ty);
    gridx=std::get<0>(tup);
    gridy=std::get<1>(tup);
    gridz=std::get<2>(tup);
    weights=std::get<3>(tup);
}

void grid_atomic_orbitals(std::vector<std::vector<Gaussian>> basis, std::vector<double> gridx,std::vector<double> gridy,std::vector<double> gridz){
    ao.resize(gridx.size(),basis.size());
    for (int i=0;i<basis.size();++i){
        auto gto=basis[i];
        Eigen::VectorXd psi(gridx.size());
        psi.setZero();
        for (int j=0;j<gto.size();++j){
            auto g=gto[j];
            psi+=gaussian_grid_eval(g,gridx,gridy,gridz);
        ao.col(i)=psi;

        }
    }
}

Eigen::VectorXd evaluate_density(){
    Eigen::VectorXd aoden(gridx.size());
    aoden.setZero();
    
    for (int p=0;p<aoden.size();++p){
        for (int u=0;u<D.rows();++u){
            for (int v=0;v<D.cols();++v){
                    aoden[p]+=ao(p,u)*D(u,v)*ao(p,v);
                    
            }
        }
    }
    
    rho=aoden;
    return aoden;
}

Eigen::MatrixXd matrix_xc_potential(Eigen::VectorXd vxc){
    Eigen::MatrixXd V_temp(ao.cols(),ao.cols());
    V_temp.setZero();
    for (int p=0;p<ao.rows();++p){
        for (int u=0;u<ao.cols();++u){
            for (int v=0;v<ao.cols();++v){
                    V_temp(u,v)+=ao(p,u)*0.5*weights[p]*vxc[p]*ao(p,v);
            }
        }
    }
    V_xc=V_temp+V_temp.transpose();
    return V_temp+V_temp.transpose();
}

double xc_energy(Eigen::VectorXd exc){
    double xc=0;
    for (int p=0;p<exc.size();++p){
        xc+=rho[p]*weights[p]*exc[p];
    }
    E_xc=xc;
    return xc;
}

void density_matrix() {
        D.resize(C.rows(), C.rows());
        D.setZero();
        Eigen::MatrixXd C1=C(Eigen::all,Eigen::seq(0,n_occ-1));

        for (int p=0; p<C1.rows(); ++p){
            for (int q=0; q<C1.rows(); ++q){
                for (int i=0; i<C1.cols(); ++i)
                {
                    
                    D(p,q)+=2*C1(p,i)*C1(q,i);
                }
            }   
        }
    }

void SCF(std::tuple<double, double, int> scf_param,MolecularIntegralsS molint, std::string functional, int limit){
    double convergence=std::get<0>(scf_param);
    double diss_c=std::get<1>(scf_param);
    int max_steps=std::get<2>(scf_param);
    diis ds;
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(molint.H_core,molint.S);
    C=es.eigenvectors();
    density_matrix();
    
    double E0=0;
    for (int step = 0; step < max_steps; ++step) 
    {
        Eigen::MatrixXd   J=get_J(D,molint.V_ee);
        
        Eigen::VectorXd rho=evaluate_density();
        double norm=0;
        for (int i=0;i<rho.size();++i){norm+=rho[i]*weights[i];}

        Eigen::VectorXd vxc=Vexchange_LDA(rho) + V_correlation(wigner_seitz_r(rho),functional,limit); //can be changed to WVN3
        Eigen::VectorXd exc=Eexchange_LDA(rho) + E_correlation(wigner_seitz_r(rho),functional,limit);

        Eigen::MatrixXd  fKS=molint.H_core+J+matrix_xc_potential(vxc);
       
        ds.update(fKS,D,molint.S);
        if (step>1){
            fKS=ds.DIIS_F();
        
        }
        
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(fKS,molint.S);
        C=es.eigenvectors();
        density_matrix();

        Eks=0;
        for (int p=0;p<D.rows();++p){
            for (int q=0;q<D.cols();++q){
                Eks+=D(p,q)*(molint.H_core(p,q)+0.5*J(p,q));
            }
        }
        Eks+=xc_energy(exc);

        std::cout <<std::setprecision(17)<< Eks << std::endl;
        double rd=ds.RMSD_check();
       

        if ((abs(Eks - E0) < convergence) && (rd<diss_c)) {
            std::cout << "Converged\n";
            E_ee = Eks;
            return;
        }
        E0 = Eks;
    }
    std::cout << "Did not converge\n";
}


void total_energy(MolecularIntegralsS molint){
    E_total=E_ee+molint.E_nn;
}

};



int main(){
    
    std::string path;
    std::cout << "Path to input: ";
    std::cin >> path;

    int gty;
    std::cout << "Configuring calcualtions "<<std::endl;
    std::cout << "Available grids: small, nucleus-dense"<<endl;
    std::cout << "Write either 1 or 2 for choice: ";
    std::cin >> gty;
    
    std::string functional;
    std::cout << "Available functionals: MP, WVN3"<<endl;
    std::cout << "Write either MP or WVN for choice: ";
    std::cin >> functional;

    int limit=0;
    if (functional=="WVN"){
        std::cout << "Spin polarized or unpolarized WVN? 0 for unpolarized, 1 for polarized. 0 by default";
        std::cin >>limit;
        if (limit!=1){limit=0;}
    }

    auto [Z,centers,alphas,coefficients,l]=inputer(path); 
    std::tuple<double,double,int> scf_param= {1e-6,1e-6,2000};

    auto start = std::chrono::high_resolution_clock::now();

    MolecularIntegralsS molS(alphas, centers,l, Z, coefficients);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::seconds>(stop - start);


    auto startDFT = std::chrono::high_resolution_clock::now();

    DFT dftmol(molS,scf_param,gty,functional,limit);
    
    std::cout <<std::setprecision(17)<< dftmol.E_total << std::endl;
    
    auto stopDFT = std::chrono::high_resolution_clock::now();
    auto durationDFT = duration_cast<std::chrono::milliseconds>(stopDFT - startDFT);

    ofstream myfile;
    myfile.open("result.txt");
    myfile << "Total energy is "<<std::setprecision(17)<<dftmol.E_total<<" hartree\n";
    myfile << "Integrals took: "<<duration.count()<<" seconds\n";
    myfile << "hartree-fock took: "<<durationDFT.count()<<" milliseconds";
    myfile.close();
    return 0;

};



       