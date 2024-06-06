#ifndef DIIS
#define DIIS
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

class diis {
public:
    //variables
    std::vector<Eigen::MatrixXd> F_list;
    std::vector<Eigen::MatrixXd> r_list;
    Eigen::MatrixXd B;
    Eigen::VectorXd pc;
public:
   //constructor
   //diis();

   //functions
   void update(Eigen::MatrixXd F,Eigen::MatrixXd D, Eigen::MatrixXd S);
   void ao_graident(Eigen::MatrixXd F,Eigen::MatrixXd D, Eigen::MatrixXd S);
   double RMSD_check();
   void buildB();
   void Pulay();
   Eigen::MatrixXd DIIS_F();

};



#endif // DIIS