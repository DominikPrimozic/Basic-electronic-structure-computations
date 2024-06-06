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

std::tuple<std::vector<double>, std::vector<double>> MK_radial_grid(int r_grid, int atom, int m = 3.0) {
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
        double temp = (1.0 / 16) * (35.0 * m[i] - 35.0 * m[i]*m[i]*m[i] + 21.0 * m[i]*m[i]*m[i]*m[i]*m[i] - 5.0 * m[i]*m[i]*m[i]*m[i]*m[i]*m[i]*m[i]); //or just coord[i]/a
        if (coords[i]<= -a){z[i]=-1.0;}
        else if (coords[i]>= a){z[i]=1.0;}
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
    double c=0;
    if (points == 0) {
        n = 6;
        a = 1.0;
        std::vector<double> pts={a, 0, 0, v, -a, 0, 0, v, 0, a, 0, v, 0, -a, 0, v, 0, 0, a, v, 0, 0, -a, v};
        return {n, pts};
    } else if (points == 1) {
        n = 12;
        a = std::sqrt(0.5);
        std::vector<double> pts={0, a, a, v, 0, -a, a, v, 0, a, -a, v, 0, -a, -a, v, a, 0, a, v, -a, 0, a, v, a, 0, -a, v, -a, 0, -a, v, a, a, 0, v, -a, a, 0, v, a, -a, 0, v, -a, -a, 0, v};
        return {n, pts};
    } else if (points == 2) {
        n = 8;
        a = std::sqrt(1.0 / 3.0);
        std::vector<double> pts={a, a, a, v, -a, a, a, v, a, -a, a, v, -a, -a, a, v, a, a, -a, v, -a, a, -a, v, a, -a, -a, v, -a, -a, -a, v};
        return {n,pts};
    } else if (points == 3) {
        n = 24;
        b = std::sqrt(1.0 - 2.0 * a * a);
        std::vector<double> pts={a, a, b, v, -a, a, b, v, a, -a, b, v, -a, -a, b, v, a, a, -b, v, -a, a, -b, v, a, -a, -b, v, -a, -a, -b, v, a, b, a, v, -a, b, a, v, a, -b, a, v, -a, -b, a, v, a, b, -a, v, -a, b, -a, v, a, -b, -a, v, -a, -b, -a, v, b, a, a, v, -b, a, a, v, b, -a, a, v, -b, -a, a, v, b, a, -a, v, -b, a, -a, v, b, -a, -a, v, -b, -a, -a, v};
        return {n,pts};
    } else if (points == 4) {
        n = 24;
        b = std::sqrt(1.0 - a * a);
        std::vector<double> pts={a, b, 0, v, -a, b, 0, v, a, -b, 0, v, -a, -b, 0, v, b, a, 0, v, -b, a, 0, v, b, -a, 0, v, -b, -a, 0, v, a, 0, b, v, -a, 0, b, v, a, 0, -b, v, -a, 0, -b, v, b, 0, a, v, -b, 0, a, v, b, 0, -a, v, -b, 0, -a, v, 0, a, b, v, 0, -a, b, v, 0, a, -b, v, 0, -a, -b, v, 0, b, a, v, 0, -b, a, v, 0, b, -a, v, 0, -b, -a, v};
        return {n,pts};
    } else if (points == 5) {
        n = 48;
        c = std::sqrt(1.0 - a * a - b * b);
        std::vector<double> pts={a, b, c, v, -a, b, c, v, a, -b, c, v, -a, -b, c, v, a, b, -c, v, -a, b, -c, v, a, -b, -c, v, -a, -b, -c, v, a, c, b, v, -a, c, b, v, a, -c, b, v, -a, -c, b, v, a, c, -b, v, -a, c, -b, v, a, -c, -b, v, -a, -c, -b, v, b, a, c, v, -b, a, c, v, b, -a, c, v, -b, -a, c, v, b, a, -c, v, -b, a, -c, v, b, -a, -c, v, -b, -a, -c, v, b, c, a, v, -b, c, a, v, b, -c, a, v, -b, -c, a, v, b, c, -a, v, -b, c, -a, v, b, -c, -a, v, -b, -c, -a, v, c, a, b, v, -c, a, b, v, c, -a, b, v, -c, -a, b, v, c, a, -b, v, -c, a, -b, v, c, -a, -b, v, -c, -a, -b, v, c, b, a, v, -c, b, a, v, c, -b, a, v, -c, -b, a, v, c, b, -a, v, -c, b, -a, v, c, -b, -a, v, -c, -b, -a, v};
        return {n,pts};
    }
    return {0,{0.0,0.0}};
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

std::unordered_map<int, std::tuple<std::vector<double>,std::vector<double>,std::vector<double>, std::vector<double>>> build_grids(Molecule molecule, int ty){

    std::unordered_map<int, std::tuple<std::vector<double>,std::vector<double>,std::vector<double>, std::vector<double>>> atom_weights;

    std::set<int, std::greater<int>> atoms(molecule.charges.begin(), molecule.charges.end()); // unique atoms
   
    
    auto [radial, angular, lebedev] = gridMetric(ty);

     for (int atom : atoms) {

        if (atom_weights.find(atom) == atom_weights.end()) {
            
            int period=1;
            if (atom>2){period=2;}
            
            
            int r_grid = radial[period];
            
            
            auto [r, dr] = MK_radial_grid(r_grid, atom);
            
           

            std::vector<double> w(r.size());
            for (int ri=0;ri<r.size();++ri){
                w[ri]=4*M_PI*r[ri]*r[ri]*dr[ri];
            }
           

            
            int lebedev_points = lebedev[angular[period]];
            
            
            std::vector<int> pruned_radial = radial_pruning(lebedev_points, r_grid);
            
            std::vector<double> coordsx,coordsy,coordsz;
            std::vector<double> volume_w;
            
            std::set<int> set_pruned(pruned_radial.begin(), pruned_radial.end());

            for (int n : set_pruned) {

                auto [garbage, weights] = lebedev_grid(n);
                
                
                std::vector<int> indices;
                for (int i2 = 0; i2 < pruned_radial.size(); ++i2) {
                    if (pruned_radial[i2] == n){indices.push_back(i2);}
                }
                
                
                
                std::vector<double> coord_tempx;
                std::vector<double> coord_tempy;
                std::vector<double> coord_tempz;
                std::vector<double> volume_w_temp;
                
                std::vector<double> r_ind,w_ind;
                for (int idx : indices) {
                    r_ind.push_back(r[idx]);
                    w_ind.push_back(w[idx]);
                }
                
                

                for (int i=0;i<weights[0].size();++i){
                    for (int j=0;j<r_ind.size();++j){
                            coord_tempx.push_back(r_ind[j]*weights[0][i]);
                            coord_tempy.push_back(r_ind[j]*weights[1][i]);
                            coord_tempz.push_back(r_ind[j]*weights[2][i]);

                            volume_w_temp.push_back(w_ind[j]*weights[3][i]);
                }
                }
                
                coordsx.insert(coordsx.end(),coord_tempx.begin(),coord_tempx.end());
                coordsy.insert(coordsy.end(),coord_tempy.begin(),coord_tempy.end());
                coordsz.insert(coordsz.end(),coord_tempz.begin(),coord_tempz.end());
                volume_w.insert(volume_w.end(),volume_w_temp.begin(),volume_w_temp.end());
            }
            
            atom_weights[atom] = std::make_tuple(coordsx,coordsy,coordsz, volume_w);
        }
    }
    
    return atom_weights;
}

Eigen::MatrixXd partition(Molecule molecule, std::vector<double> coordsx,std::vector<double> coordsy,std::vector<double> coordsz, int n_atoms,int n_grids) {
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
    
    
    for (int i = 0; i < n_atoms; ++i) {
        std::vector<double> cx(coordsx.size()),cy(coordsy.size()),cz(coordsz.size());
        for (int j = 0; j < coordsx.size(); ++j) {
            
                cx[j]=coordsx[j]-molecule.position[i][0];
                cy[j]=coordsy[j]-molecule.position[i][1];
                cz[j]=coordsz[j]-molecule.position[i][2];
        }
        
        Eigen::VectorXd cc(cx.size());
        cc.setZero();
        for (int ci=0;ci<cx.size();++ci){
            
            cc[ci]+=(cx[ci]*cx[ci])+(cy[ci]*cy[ci])+(cz[ci]*cz[ci]);
            
            cc[ci]=std::sqrt(cc[ci]);
        
        }
        
    meshes.row(i)=cc;     
     
        
    }
    
    Eigen::MatrixXd becke_partition(n_atoms, n_grids);
    becke_partition.setOnes();
    
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = 0; j < i; ++j) {
                Eigen::VectorXd u = (1 / separations(i,j)) * (meshes.row(i) - meshes.row(j)); // Becke's transformation to elliptical coordinates, antisymmetric with respect to index switch
               
                Eigen::VectorXd v = adjust(i,j,u,a); // vij = uij + aij*(1 - uij^2)
                
                Eigen::VectorXd g_val = SB_scheme(v); // all this preserves antisymmetry due to index exchange

               
                for (int k=0;k<becke_partition.cols();++k){
                    becke_partition(i,k) *= 0.5 * (1 - g_val[k]); // cell function s, pk = product of ski over all atoms but i!=k (product over indices up to num of atoms)
                    becke_partition(j,k) *= 0.5 * (1 + g_val[k]);
                }
                 // since aij=-aji and gij=-gji can lessen loops by doing it already with + instead of -
            }
        }
    
   
    return becke_partition;
}

std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>> grid_partition(Molecule molecule, std::unordered_map<int, std::tuple<std::vector<double>,std::vector<double>,std::vector<double>, std::vector<double>>>& atom_grid_table) {
    int n_atoms = molecule.charges.size();
    
    std::vector<double> coords_x,coords_y,coords_z;
    std::vector<double> weights;
    
    for (int ato = 0; ato < n_atoms; ++ato) {
        auto [a_coordsx,a_coordsy,a_coordsz, a_volume] = atom_grid_table.at(molecule.charges[ato]);
        std::vector<double> translated_coordsx(a_coordsx.size()),translated_coordsy(a_coordsy.size()),translated_coordsz(a_coordsz.size());
        for (int i = 0; i < a_coordsx.size(); ++i) {
            
                translated_coordsx[i] = a_coordsx[i] + molecule.position[ato][0]; // translate
                translated_coordsy[i] = a_coordsy[i] + molecule.position[ato][1];
                translated_coordsz[i] = a_coordsz[i] + molecule.position[ato][2];
            
        }

        int n_grids = a_coordsx.size();
        Eigen::MatrixXd becke_partition = partition(molecule, translated_coordsx,translated_coordsy,translated_coordsz, n_atoms,n_grids);
        std::vector<double> w(n_grids);

       
        Eigen::VectorXd sume=becke_partition.colwise().sum();
        
        for (int i2 = 0; i2 < n_grids; ++i2) {
            
            w[i2] = a_volume[i2] * becke_partition(ato,i2) / sume[i2]; // normalized weights entering the quadrature
        }
        
        coords_x.insert(coords_x.end(), translated_coordsx.begin(), translated_coordsx.end());
        coords_y.insert(coords_y.end(), translated_coordsy.begin(), translated_coordsy.end());
        coords_z.insert(coords_z.end(), translated_coordsz.begin(), translated_coordsz.end());
        weights.insert(weights.end(), w.begin(), w.end());
    }
    
    return {coords_x,coords_y,coords_z, weights};
}
std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>> grid_maker (Molecule mol, int ty){
    auto atw=build_grids(mol,ty);
    auto [xx,yy,zz,ww]=grid_partition(mol,atw);
    auto result = std::reduce(ww.begin(), ww.end());
    return {xx,yy,zz,ww};
}



