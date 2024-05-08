#ifndef GRID
#define GRID

#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <eigen_lib/Eigen/Dense>
#include <numeric>   
#include <eigen_lib/Eigen/Dense>
#include <molecule.h>
#include <set>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>> grid_maker (Molecule mol, int ty=1);

#endif // GRID
