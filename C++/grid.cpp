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
#include <grid.h>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

Eigen::MatrixXd atom_separations(Molecule& molecule){
    int n_atoms = molecule.charges.size();
    Eigen::MatrixXd separations(n_atoms,n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = 0; j < n_atoms; ++j) {
            separations(i,j)=std::sqrt( std::pow(molecule.position[i][0]-molecule.position[j][0],2) +std::pow(molecule.position[i][1]-molecule.position[j][1],2) +std::pow(molecule.position[i][2]-molecule.position[j][2],2) );
            separations(i,i)=1;
        }
    }
    return separations;
}

std::tuple<std::unordered_map<int, int>, std::unordered_map<int, int>, std::unordered_map<int, int>> gridMetric(int ty) {
    std::unordered_map<int, int> radial, angular, lebedev;
    if (ty == 1) {
        radial = {{1, 10}, {2, 15}};
        angular = {{1, 11}, {2, 15}};
        lebedev = {{11, 50}, {15, 86}};
    } else if (ty == 2) {
        radial = {{1, 50}, {2, 75}};
        angular = {{1, 29}, {2, 29}};
        lebedev = {{29, 302}};
    }
    return {radial, angular, lebedev};
}

std::tuple<std::vector<double>, std::vector<double>> MK_radial_grid(int r_grid, int atom, int m = 3) {
    std::vector<int> atom7s={3,4,11,12,19,20};
    double a = 5.0;
    if (std::find(atom7s.begin(), atom7s.end(), atom) != atom7s.end()) {
            a=7.0;
        }
    
    std::vector<double> r(r_grid), dr(r_grid);

    for (int i = 0; i < r_grid; ++i) {

        double x = (i + 0.5) / r_grid;
        r[i] = -a * std::log(1 - std::pow(x, m));
        dr[i] = a * m * std::pow(x, m - 1) / ((1 - std::pow(x, m)) * r_grid);
    }
    return {r, dr};
}

std::vector<int> radial_pruning(int lebedev_grid, int r_grid) {
    std::vector<int> pruned(r_grid);
    
    int third = r_grid / 3;
    int half = r_grid / 2;
    
    std::fill(pruned.begin(), pruned.begin() + third, 14);
    std::fill(pruned.begin() + third, pruned.begin() + half, 50);
    std::fill(pruned.begin() + half, pruned.end(), lebedev_grid);
    
    return pruned;
}

Eigen::VectorXd SB_scheme(Eigen::VectorXd coords) {
    double a = 0.64;
    std::vector<double> m(coords.size());
    Eigen::VectorXd z(coords.size());

    for (int i=0;i<z.size();++i){
        m[i]=coords[i]/a;
        double temp = (1.0 / 16) * (35 * m[i] - 35 * std::pow(m[i], 3) + 21 * std::pow(m[i], 5) - 5 * std::pow(m[i], 7)); //or just coord[i]/a
        if (coords[i]<= -a){z[i]=-1;}
        else if (coords[i]>= a){z[i]=1;}
        else {z[i]=temp;}
    }
    return z;
}

Eigen::MatrixXd treutlerAdjust(Molecule molecule) {
    std::unordered_map<int, double> bragg = {
        {1, 0.35}, {2, 1.40}, {3, 1.45}, {4, 1.05}, {5, 0.85}, {6, 0.70}, {7, 0.65}, {8, 0.60}, {9, 0.50}, {10, 1.50}
    };
    std::vector<int> atoms=molecule.charges;
    std::vector<double> r(atoms.size());
    for (int i = 0; i < atoms.size(); ++i)
        r[i] = std::sqrt(bragg[(atoms[i])]);

    int natoms = atoms.size();
    Eigen::MatrixXd a(natoms,natoms);
    a.setZero();
    for (int i = 0; i < natoms; ++i) {
        for (int j = i + 1; j < natoms; ++j) {
            double temp = 0.25 * (r[i]-r[j])*(r[i]+r[j])/(r[i]*r[j]);
            if (temp<-0.5){a(i,j)=-0.5;}
            else if (temp>0.5){a(i,j)=0.5;}
            else {a(i,j)=temp;}

            a(j,i)=-a(i,j);
        }
    }


    return a;
}

std::pair<int, std::vector<double>> spheric_symmetry(int points, double a, double b, double v) {
    
    int n;
    std::unordered_map<std::string, std::vector<int>> shell;
    double c;
    if (points == 0) {
        n = 6;
        a = 1.0;
        shell = {{"-a", {4, 13, 22}}, {"+a", {0, 9, 18}}, {"-b", {}}, {"+b", {}}, {"v", {3, 7, 11, 15, 19, 23}}};
    } else if (points == 1) {
        n = 12;
        a = std::sqrt(0.5);
        shell = {{"-a", {5, 10, 13, 14, 20, 26, 28, 30, 36, 41, 44, 45}},
                 {"+a", {1, 2, 6, 9, 16, 18, 22, 24, 32, 33, 37, 40}}, {"-b", {}}, {"+b", {}}, {"v", {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}}};
    } else if (points == 2) {
        n = 8;
        a = std::sqrt(1.0 / 3.0);
        shell = {{"-a", {4, 9, 12, 13, 18, 20, 22, 25, 26, 28, 29, 30}},
                 {"+a", {0, 1, 2, 5, 6, 8, 10, 14, 16, 17, 21, 24}}, {"-b", {}}, {"+b", {}}, {"v", {3, 7, 11, 15, 19, 23, 27, 31}}};
    } else if (points == 3) {
        n = 24;
        b = std::sqrt(1.0 - 2.0 * a * a);
        shell = {{"+a", {0, 1, 5, 8, 16, 17, 21, 24, 32, 34, 38, 40, 42, 46, 48, 56, 65, 66, 69, 70, 74, 78, 81, 85}},
                 {"-a", {4, 9, 12, 13, 20, 25, 28, 29, 36, 44, 50, 52, 54, 58, 60, 62, 73, 77, 82, 86, 89, 90, 93, 94}},
                 {"+b", {2, 6, 10, 14, 33, 37, 49, 53, 64, 72, 80, 88}},
                 {"-b", {18, 22, 26, 30, 41, 45, 57, 61, 68, 76, 84, 92}}, {"v", {3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95}}};
    } else if (points == 4) {
        n = 24;
        b = std::sqrt(1.0 - a * a);
        shell = {{"+a", {0, 8, 17, 21, 32, 40, 50, 54, 65, 73, 82, 86}},
                 {"-a", {4, 12, 25, 29, 36, 44, 58, 62, 69, 77, 90, 94}},
                 {"+b", {1, 5, 16, 24, 34, 38, 48, 56, 66, 70, 81, 89}},
                 {"-b", {9, 13, 20, 28, 42, 46, 52, 60, 74, 78, 85, 93}}, {"v", {3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95}}};
    } else if (points == 5) {
        n = 48;
        c = std::sqrt(1.0 - a * a - b * b);
        shell = {{"+a", {0, 8, 16, 24, 32, 40, 48, 56, 65, 69, 81, 85, 98, 102, 106, 110, 129, 133, 145, 149, 162, 166, 170, 174}},
                 {"-a", {4, 12, 20, 28, 36, 44, 52, 60, 73, 77, 89, 93, 114, 118, 122, 126, 137, 141, 153, 157, 178, 182, 186, 190}},
                 {"+b", {1, 5, 17, 21, 34, 38, 42, 46, 64, 72, 80, 88, 96, 104, 112, 120, 130, 134, 138, 142, 161, 165, 177, 181}},
                 {"-b", {9, 13, 25, 29, 50, 54, 58, 62, 68, 76, 84, 92, 100, 108, 116, 124, 146, 150, 154, 158, 169, 173, 185, 189}},
                 {"+c", {2, 6, 10, 14, 33, 37, 49, 53, 66, 70, 74, 78, 97, 101, 113, 117, 128, 136, 144, 152, 160, 168, 176, 184}},
                 {"-c", {18, 22, 26, 30, 41, 45, 57, 61, 82, 86, 90, 94, 105, 109, 121, 125, 132, 140, 148, 156, 164, 172, 180, 188}},
                 {"v", {3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95,99,103,107,111,115,119,123,127,131,135,139,143,147,151,155,159,163,167,171,175,179,183,187,191}}};
    }

    if (points < 5) {
        shell["+c"] = {};
        shell["-c"] = {};
    }
    std::vector<double> pts(n*4, 0);
    for (int i = 0; i < n * 4; ++i) {
        if (std::find(shell["-a"].begin(), shell["-a"].end(), i) != shell["-a"].end()) {
            pts[i]=-a;
        }
        else if (std::find(shell["+a"].begin(), shell["+a"].end(), i) != shell["+a"].end()){
            pts[i]=+a;
        }
        else if (std::find(shell["-b"].begin(), shell["-b"].end(), i) != shell["-b"].end()){
            pts[i]=-b;
        }
        else if (std::find(shell["+b"].begin(), shell["+b"].end(), i) != shell["+b"].end()){
            pts[i]=+b;
        }
        else if (std::find(shell["-c"].begin(), shell["-c"].end(), i) != shell["-c"].end()){
            pts[i]=-c;
        }
        else if (std::find(shell["+c"].begin(), shell["+c"].end(), i) != shell["+c"].end()){
            pts[i]=-c;
        }
        else if (std::find(shell["v"].begin(), shell["v"].end(), i) != shell["v"].end()){
            pts[i]=v;
        }
    }
     

    return {n, pts};
}

std::pair<int, std::vector<std::vector<double>>> lebedev_grid(int order) {
    int suma;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> v;
    std::vector<int> n;

    if (order == 14) {
        suma = 2;
         a.resize(suma,0.0);
         b.resize(suma,0.0);
         v = {0.6666666666666667e-1, 0.7500000000000000e-1};
         n = {0, 2};
    }
     else if (order == 50) {
        suma = 4;
         a = {0.0, 0.0, 0.0, 0.3015113445777636e+0};
         b.resize(suma,0.0);
         v = {0.1269841269841270e-1, 0.2257495590828924e-1, 0.2109375000000000e-1, 0.2017333553791887e-1};
         n = {0, 1, 2, 3};

     }
     else if (order == 86) {
        suma = 5;
         a = {0.0, 0.0, 0.3696028464541502e+0, 0.6943540066026664e+0, 0.3742430390903412e+0};
         b.resize(suma,0.0);
         v = {0.1154401154401154e-1, 0.1194390908585628e-1, 0.1111055571060340e-1, 0.1187650129453714e-1, 0.1181230374690448e-1};
         n = {0, 2, 3, 3, 4};

     }
     else if (order == 302) {
        suma = 12;
         a = {0.0,0.0,0.3515640345570105,0.6566329410219612,0.4729054132581005,0.9618308522614784e-1,0.2219645236294178, 0.7011766416089545, 0.2644152887060663, 0.5718955891878961, 0.2510034751770465, 0.1233548532583327};
         b = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.8000727494073952,0.4127724083168531};
         v = {0.8545911725128148e-3, 0.3599119285025571e-2, 0.3449788424305883e-2, 0.3604822601419882e-2, 0.3576729661743367e-2,
                                  0.2352101413689164e-2, 0.3108953122413675e-2, 0.3650045807677255e-2, 0.2982344963171804e-2, 0.3600820932216460e-2,
                                  0.3571540554273387e-2, 0.3392312205006170e-2};
         n = {0, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5};

    
    }
    std::vector<int> returned_order(suma);
    std::vector<std::vector<double>> g(4);
    for (int i=0;i<suma;++i){
        std::pair<int, std::vector<double>> paire=spheric_symmetry(n[i],a[i],b[i],v[i]);
        returned_order[i]=paire.first;
        std::vector<double> lg=paire.second;
        for (int elg=0;elg<lg.size();elg+=4){
            g[0].push_back(lg[elg]);
            g[1].push_back(lg[elg+1]);
            g[2].push_back(lg[elg+2]);
            g[3].push_back(lg[elg+3]);
        }
        
    }
    auto result = std::reduce(returned_order.begin(), returned_order.end());
    return {result, g};
}

std::unordered_map<int, std::pair<std::vector<std::vector<double>>, std::vector<double>>> build_grids(Molecule molecule, int ty){

    std::unordered_map<int, std::pair<std::vector<std::vector<double>>, std::vector<double>>> atom_weights;

    std::set<int, std::greater<int>> atoms(molecule.charges.begin(), molecule.charges.end()); // unique atoms
    //std::cout<< "set made" <<std::endl;
    
    auto [radial, angular, lebedev] = gridMetric(ty);

     for (int atom : atoms) {

        if (atom_weights.find(atom) == atom_weights.end()) {
            
            int period=1;
            if (atom>2){period=2;}
            
            
            int r_grid = radial[period];
            
            
            auto [r, dr] = MK_radial_grid(r_grid, atom);
            
           // std::cout<< "radial" <<std::endl;
           // std::cout<<"atom is "<< atom <<std::endl;
           // std::cout<< r.size() <<std::endl;
          //  std::cout<< dr.size()<<std::endl;

            std::vector<double> w(r.size());
            for (int ri=0;ri<r.size();++ri){
                w[ri]=4*M_PI*r[ri]*r[ri]*dr[ri];
            }
           // std::cout<<"big w "<< w.size()<<std::endl;

            
            int lebedev_points = lebedev[angular[period]];
            std::cout<<"leb "<< lebedev_points<<std::endl;
            // Prune the radial grid
            std::vector<int> pruned_radial = radial_pruning(lebedev_points, r_grid);
           // std::cout<< "we pruned"<<std::endl;
            
            std::vector<std::vector<double>> coords(3);
            std::vector<double> volume_w;
            
            std::set<int> set_pruned(pruned_radial.begin(), pruned_radial.end());
           // std::cout<< "pruned set"<<std::endl;
            for (int n : set_pruned) {

                auto [garbage, weights] = lebedev_grid(n);
                //std::cout<< weights.size()<<std::endl;
                //std::cout<< weights[0].size()<<std::endl;
                
                std::vector<int> indices;
                for (int i2 = 0; i2 < pruned_radial.size(); ++i2) {
                    if (pruned_radial[i2] == n){indices.push_back(i2);}
                }
                //std::cout<< indices.size()<<std::endl;
                
                std::vector<std::vector<double>> coord_temp(3);
                std::vector<double> volume_w_temp;
                
                std::vector<double> r_ind,w_ind;
                for (int idx : indices) {
                    r_ind.push_back(r[idx]);
                    w_ind.push_back(w[idx]);
                }
                
                /*
                for (int ii=0;ii<r_ind.size();++ii) {
                     std::cout<<"r "<< r_ind[ii]<<std::endl;
                      std::cout<<"w "<< w_ind[ii]<<std::endl;
                }
                //gooooood
                */

                for (int i=0;i<weights[0].size();++i){
                    for (int j=0;j<r_ind.size();++j){
                            coord_temp[0].push_back(r_ind[j]*weights[0][i]);
                            coord_temp[1].push_back(r_ind[j]*weights[1][i]);
                            coord_temp[2].push_back(r_ind[j]*weights[2][i]);

                            volume_w_temp.push_back(w_ind[j]*weights[3][i]);
                }
                }
                /*
                for (int ti=0; ti<coord_temp[0].size();++ti){
                     std::cout<<"x in atom "<< coord_temp[0][ti]<<" "<< atom<<std::endl;
                }
                
                for (int ti=0; ti<volume_w_temp.size();++ti){
                     std::cout<<"w in atom "<< volume_w_temp[ti]<<" "<< atom<<std::endl;
                }
                std::cout<<"coord size "<< coord_temp[0].size()<<std::endl; //size is right
                */ //hoho, if it works for x surely for all
                coords[0].insert(coords[0].end(),coord_temp[0].begin(),coord_temp[0].end());
                coords[1].insert(coords[1].end(),coord_temp[1].begin(),coord_temp[1].end());
                coords[2].insert(coords[2].end(),coord_temp[2].begin(),coord_temp[2].end());
                volume_w.insert(volume_w.end(),volume_w_temp.begin(),volume_w_temp.end());
            }
            
            atom_weights[atom] = std::make_pair(coords, volume_w);
        }
    }

    return atom_weights;
}

Eigen::MatrixXd partition(Molecule molecule, std::vector<std::vector<double>> coords, int n_atoms,int n_grids) {
    auto a =treutlerAdjust(molecule);
    auto adjust=[](int i, int j, Eigen::VectorXd g, Eigen::MatrixXd a){
        Eigen::VectorXd ret(g.size());
        for (int gi=0;gi<g.size();++gi){
            ret[gi]=g[gi] + a(i,j)*(1.0-g[gi]*g[gi]);
        }
        return ret;
    };

    Eigen::MatrixXd meshes(n_atoms, n_grids);
    Eigen::MatrixXd separations = atom_separations(molecule);
    
    //std::cout<<"n grid "<< n_grids<<std::endl;
    for (int i = 0; i < n_atoms; ++i) {
        std::vector<std::vector<double>> c(3,std::vector<double>(coords[0].size()));
        for (int j = 0; j < coords[0].size(); ++j) {
            for (int k=0;k<3;++k){
                c[k][j]=coords[k][j]-molecule.position[i][k];
            }
        //std::cout<<"translated "<<c.size()<<" "<<c[0].size()<<std::endl;
        Eigen::VectorXd cc(c[0].size());
        cc.setZero();
        for (int i=0;i<c[0].size();++i){
            for (int j=0;j<c.size();++j){
                cc[i]+=(c[j][i]*c[j][i]);
            }
            cc[i]=std::sqrt(cc[i]);
        }
        meshes.row(i)=cc;     
        }
    }
    //std::cout<<"meshes\n "<<meshes<<std::endl;
    Eigen::MatrixXd becke_partition(n_atoms, n_grids);
    becke_partition.setOnes();
    
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = 0; j < i; ++j) {
                Eigen::VectorXd u = (1 / separations(i,j)) * (meshes.row(i) - meshes.row(j)); // Becke's transformation to elliptical coordinates, antisymmetric with respect to index switch
               
                Eigen::VectorXd v = adjust(i,j,u,a); // vij = uij + aij*(1 - uij^2)
                
                Eigen::VectorXd g_val = SB_scheme(v); // all this preserves antisymmetry due to index exchange

               // std::cout<<"1\n "<< one<<std::endl;
                for (int k=0;k<becke_partition.cols();++k){
                    becke_partition(i,k) *= 0.5 * (1 - g_val[k]); // cell function s, pk = product of ski over all atoms but i!=k (product over indices up to num of atoms)
                    becke_partition(j,k) *= 0.5 * (1 + g_val[k]);
                }
                 // since aij=-aji and gij=-gji can lessen loops by doing it already with + instead of -
            }
        }
    
   // std::cout<<"becke\n "<< becke_partition<<std::endl;
    return becke_partition;
}

std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>> grid_partition(Molecule molecule, std::unordered_map<int, std::pair<std::vector<std::vector<double>>, std::vector<double>>>& atom_grid_table) {
    int n_atoms = molecule.charges.size();
    std::cout<<n_atoms<<std::endl;
    std::vector<double> coords_x,coords_y,coords_z;
    std::vector<double> weights;
    
    for (int ato = 0; ato < n_atoms; ++ato) {
        auto [a_coords, a_volume] = atom_grid_table.at(molecule.charges[ato]);
        std::vector<std::vector<double>> translated_coords(3, std::vector<double>(a_coords[0].size()));
        for (int i = 0; i < a_coords[0].size(); ++i) {
            for (int j = 0; j < 3; ++j) {
                translated_coords[j][i] = a_coords[j][i] + molecule.position[ato][j]; // translate
            }
        }

        int n_grids = a_coords[0].size();
        Eigen::MatrixXd becke_partition = partition(molecule, translated_coords, n_atoms,n_grids);
        std::vector<double> w(n_grids);

       // std::vector<double> sume(becke_partition.cols());
        Eigen::VectorXd sume=becke_partition.colwise().sum();
        
        for (int i = 0; i < n_grids; ++i) {
            //for (int j = 0; j < becke_partition.rows(); ++j) {
             //   sume[i] += becke_partition(j,i);
            //}
            w[i] = a_volume[i] * becke_partition(ato,i) / sume[i]; // normalized weights entering the quadrature
        }
        //for (int fi=0;fi<coords_x.size();++fi){
         //   std::cout<<"ss "<<sume[fi]<<std::endl;}
        coords_x.insert(coords_x.end(), translated_coords[0].begin(), translated_coords[0].end());
        coords_y.insert(coords_y.end(), translated_coords[1].begin(), translated_coords[1].end());
        coords_z.insert(coords_z.end(), translated_coords[2].begin(), translated_coords[2].end());
        weights.insert(weights.end(), w.begin(), w.end());
    }
    
    return {coords_x,coords_y,coords_z, weights};
}
std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>> grid_maker (Molecule mol, int ty){
    auto atw=build_grids(mol,ty);
    auto [xx,yy,zz,ww]=grid_partition(mol,atw);
    return {xx,yy,zz,ww};
}


