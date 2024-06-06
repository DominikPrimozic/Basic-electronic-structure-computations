#include <cmath>
#include <iostream>
#include <limits>
#include <gaussian.h>
#include <molecule.h>
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib\unsupported\Eigen\CXX11\Tensor>
#include <boost/math/special_functions/gamma.hpp>
#include <diis.h>
#include <eigen_lib\unsupported/Eigen/MatrixFunctions>



void diis::ao_graident(Eigen::MatrixXd F,Eigen::MatrixXd D, Eigen::MatrixXd S){
    Eigen::MatrixXd A=S.pow(-0.5);
    //std::cout <<"A\n"<< A << std::endl;
    Eigen::MatrixXd fds(F.rows(),S.cols());
    Eigen::MatrixXd sdf(S.rows(),F.cols());
    fds.setZero();
    sdf.setZero();
    //r is zero constantly grrrr, check with python what stuff should be
    for (int i=0; i<F.rows();++i)
    {
        for (int m=0; m<D.rows();++m)
    {
            for (int n=0; n<D.cols();++n)
    {
                for (int j=0; j<S.cols();++j)
    {
                    fds(i,j)+=0.5*F(i,m)*D(m,n)*S(n,j);
                    sdf(i,j)+=0.5*S(i,m)*D(m,n)*F(n,j);
    }
    }
    }
    }
    //std::cout <<"fds\n"<< fds << std::endl;
   // std::cout <<"sdf\n"<< sdf << std::endl;
    Eigen::MatrixXd r(A.cols(),A.cols());
    r.setZero();
    for (int m=0; m<A.rows();++m)
    {
        for (int i=0; i<A.cols();++i)
    {
            for (int n=0; n<A.rows();++n)
    {
                for (int j=0; j<A.cols();++j)
    {
                    r(i,j)+=A(m,i) * ( fds(m,n)-sdf(m,n) ) * A(n,j);
                    //std::cout<<"r\n" << r(i,j) << std::endl;


    }
    }
    }
    }
    //std::cout<<"r\n" << r << std::endl;
    r_list.push_back(r);
}

void diis:: update(Eigen::MatrixXd F,Eigen::MatrixXd D, Eigen::MatrixXd S){
    F_list.push_back(F);
    ao_graident(F,D, S);

}

double diis::RMSD_check(){

    Eigen::MatrixXd r_last=r_list.back();
    Eigen::MatrixXd r_sq(r_last.rows(),r_last.cols());
    double rmsd=0;
    for (int i=0; i<r_last.rows();++i){
        for (int j=0; j<r_last.cols();++j){
            r_sq(i,j)=r_last(i,j)*r_last(i,j);
        }
    }
    rmsd=r_sq.mean();
    return std::sqrt(rmsd);
}

void diis:: buildB(){
    int dim=F_list.size()+1;
    Eigen::MatrixXd B(dim,dim);
    B.setZero();

    for (int i=0;i<B.cols();++i){
        B(dim-1,i)=-1;
    }
    for (int i=0;i<B.rows();++i){
        B(i,dim-1)=-1;
    }
    B(dim-1,dim-1)=0;

    for (int i=0; i<F_list.size(); ++i){
        for (int j=0; j<F_list.size(); ++j){
            for (int ii=0; ii<r_list[i].rows(); ++ii){
                for (int jj=0; jj<r_list[i].cols(); ++jj){
                    B(i,j)+=r_list[i](ii,jj)*r_list[j](ii,jj);

                }
            }

        }
    }
    //std::cout <<"B\n"<< B << std::endl;
    this->B= B;
}

void diis::Pulay(){
    buildB();
    int dim=F_list.size()+1;
    Eigen::VectorXd desna(dim);
    desna.setZero();
    desna[dim-1]=-1;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> dec(B);
    Eigen::VectorXd coeff=dec.solve(desna);
    this->pc=coeff;
    //std::cout <<"coeff\n"<< pc << std::endl;
}

Eigen::MatrixXd diis::DIIS_F(){
    Pulay();
    Eigen::MatrixXd F_o(F_list[0].rows(),F_list[0].cols());
    F_o.setZero();
    for (int x=0; x<(pc.size()-1); ++x){
        F_o+=pc[x]*F_list[x];
    }
    //std::cout <<"F_out\n"<< F_o << std::endl;
    return F_o;
}