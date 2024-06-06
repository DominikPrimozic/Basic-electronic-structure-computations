#ifndef PostHF
#define PostHF
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib/Eigen/Eigenvalues>
#include <eigen_lib\unsupported\Eigen\CXX11\Tensor>

double MP2( Eigen::MatrixXd C,  Eigen::VectorXd e, Eigen::Tensor<double,4> eri,int nao,int noc);
Eigen::Tensor<double,4> get_eri_mo(const Eigen::Tensor<double,4> &eri_ao, const Eigen::MatrixXd &coeff, int nao);


#endif //PostHF

