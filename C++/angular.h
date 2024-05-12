#ifndef ANGULAR
#define ANGULAR
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
double factorial(int n);
double boys(double x, int n);
double compute_s(double& A,double& B,double& P,double& p,int& l1, int&l2);
double overlap(Gaussian gi, Gaussian gj);
double compute_k(double& A,double& B,double& P,double& a,double& b,int& l1, int&l2);
double kinetic(Gaussian gi, Gaussian gj);
double nuclearHRR(std::vector<Gaussian> cGTO1,std::vector<Gaussian> cGTO2,std::vector<int> la,std::vector<int> lb,std::vector<double> R);
double HRR(std::vector<Gaussian> cGTO1,std::vector<Gaussian> cGTO2,std::vector<Gaussian> cGTO3,std::vector<Gaussian> cGTO4,std::vector<int> la,std::vector<int> lb,std::vector<int> lc,std::vector<int> ld);

#endif // ANGULAR