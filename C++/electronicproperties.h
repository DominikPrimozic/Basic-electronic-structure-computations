#ifndef eprop
#define eprop

#include <hartreefock.h>
#include <DFT.h>
#include <eigen_lib/Eigen/Dense>

std::vector<double> mulliken(MolecularIntegralsS molint, Eigen::MatrixXd D);



#endif //eprop