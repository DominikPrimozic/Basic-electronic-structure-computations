#ifndef GRID
#define GRID

#include <istream>
#include <iostream>
#include <fstream>
#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>

std::tuple<std::vector<double>, std::vector<double>,std::vector<double>,std::vector<double>> load_grid();

#endif // GRID
