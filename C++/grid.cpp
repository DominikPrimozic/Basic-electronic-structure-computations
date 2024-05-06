#include <istream>
#include <iostream>
#include <fstream>
#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <grid.h>

std::tuple<std::vector<double>, std::vector<double>,std::vector<double>,std::vector<double>> load_grid()
{
    // Note: you need to supply the data type you are loading
    //       in this case "double".
    auto data = xt::load_npy<double>("grid_x.npy");
    std::vector<double> x_grid(data.begin(), data.end());

    auto data2 = xt::load_npy<double>("grid_y.npy");
    std::vector<double> y_grid(data2.begin(), data2.end());

    auto data3 = xt::load_npy<double>("grid_z.npy");
    std::vector<double> z_grid(data3.begin(), data3.end());

    auto data4 = xt::load_npy<double>("weights.npy");
    std::vector<double> weights(data4.begin(), data4.end());




  //it does work
    return std::make_tuple(x_grid,y_grid,z_grid,weights);
}
// auto [x,y,z,w]=load_grid