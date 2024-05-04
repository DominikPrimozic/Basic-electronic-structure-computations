
#include <iostream>
#include <vector>
#include <cmath>
#include <eigen_lib/Eigen/Dense>
#include <eigen_lib/Eigen/Core>
#include <eigen_lib/Eigen/Sparse>
#include <eigen_lib/Eigen/SparseCore>
#include <eigen_lib\unsupported\Eigen\KroneckerProduct>
#include <exco.h>
#include <Spectra\GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

using namespace Eigen;
using namespace std;


typedef Triplet<double> T;

SparseMatrix<double> laplacian(Vector3i n_points, Vector3d dr) {
    int nx = n_points(0), ny = n_points(1), nz = n_points(2);
    double dx = dr(0), dy = dr(1), dz = dr(2);

    VectorXd ex = VectorXd::Ones(nx) / (dx * dx);
    VectorXd ey = VectorXd::Ones(ny) / (dy * dy);
    VectorXd ez = VectorXd::Ones(nz) / (dz * dz);
    
    SparseMatrix<double> Lx(nx, nx), Ly(ny, ny), Lz(nz, nz);

    for (int i = 0; i < nx; ++i) {
        Lx.insert(i, i) = -2 * ex(i);
        if (i > 0) Lx.insert(i, i - 1) = ex(i);
        if (i < nx - 1) Lx.insert(i, i + 1) = ex(i);
    }
    
    for (int i = 0; i < ny; ++i) {
        Ly.insert(i, i) = -2 * ey(i);
        if (i > 0) Ly.insert(i, i - 1) = ey(i);
        if (i < ny - 1) Ly.insert(i, i + 1) = ey(i);
    }

    for (int i = 0; i < nz; ++i) {
        Lz.insert(i, i) = -2 * ez(i);
        if (i > 0) Lz.insert(i, i - 1) = ez(i);
        if (i < nz - 1) Lz.insert(i, i + 1) = ez(i);
    }
    //Lx Ly
    SparseMatrix<double> Ix(nx, nx), Iy(ny, ny), Iz(nz,nz);
    Ix.setIdentity();
    Iy.setIdentity();
    Iz.setIdentity();
    SparseMatrix<double> Lap1 = kroneckerProduct(Ly, Iz) + kroneckerProduct(Iy, Lz);
    SparseMatrix<double> Iyz(nz*ny, nz*ny);
    Iyz.setIdentity();
    SparseMatrix<double> Laplacian = kroneckerProduct(Lx, Iyz) + kroneckerProduct(Ix, Lap1) ;

    return Laplacian;
}

SparseMatrix<double> nuclear_potential(Vector3d centers, VectorXd charge, Vector3i n_points, vector<VectorXd> grid) {
    int natoms = 1; //for He
    
    int npoints = n_points.prod();
    VectorXd x = grid[0], y = grid[1], z = grid[2];
    int nx=n_points(0),ny=n_points(1),nz=n_points(2);
    SparseMatrix<double> v_ne(npoints, npoints);
    //v_ne.reserve(VectorXi::Constant(npoints, natoms));
    for (int a = 0; a < natoms; ++a) {
        
        double xi = centers(0), yi = centers(1), zi = centers(2);
        double zval = charge(a);
        
        int c=0;
        for (int i = 0; i < nx; ++i) {
            for (int j=0; j< ny; ++j){
                for (int k=0; k< nz; ++k) {
                    double dist = sqrt(pow(x(j) - xi, 2) + pow(y(i) - yi, 2) + pow(z(k) - zi, 2));
                    if (dist==0) {dist=1e-100;}
                    v_ne.coeffRef(c, c) -= zval / dist;
                    ++c;
                        
                }
            } 
            
            
        } 
        
    }
    return v_ne;
}

double nuclear_energy(SparseMatrix<double> v_ne, VectorXd density, double dV) {
    int npoints=density.rows();
    double eV=0;
    for (int i=0;i<npoints;++i){
        eV+=v_ne.coeffRef(i,i)*density(i)*dV;
    }
    return eV;
}

pair<SparseMatrix<double>, double> hartree_potential(Vector3i n_points, SparseMatrix<double>& Laplacian, VectorXd& density, double dV) {
    int npoints = n_points.prod();

    ConjugateGradient<SparseMatrix<double> > cg;
    cg.compute(Laplacian);
    VectorXd right=-4*3.14*density;
    VectorXd vh = cg.solve(right);
   
    SparseMatrix<double> vh_diag(npoints, npoints);
    vh_diag.reserve(VectorXi::Constant(npoints, 1));
    for (int i = 0; i < npoints; ++i) {
        vh_diag.insert(i, i) = vh(i);
    }

    double eh = 0.5 * density.dot(vh) * dV;
    return { vh_diag, eh };
}
SparseMatrix<double> kinetic_matrix(SparseMatrix<double>& Laplacian) {
    return -0.5 * Laplacian;
}

double kinetic_energy(SparseMatrix<double>& T, MatrixXd& psi, int nelect, double dV) {
    double eT = 0;
    
    MatrixXd temp = psi.transpose()*(T * psi) * dV*nelect;
    
    eT=temp(0,0);
    
    return eT;
}
pair<SparseMatrix<double>, double> ex_co(VectorXd& density, Vector3i n_points, double dV) {
    int npoints = n_points.prod();
    VectorXd rs = wigner_seitz_r(density);
    double ex=0,ec=0;
    VectorXd vx = Vexchange_LDA(density);
    VectorXd exx = Eexchange_LDA(density);
    VectorXd ecc = Ecorrelation_LDA(rs);
    VectorXd vc = Vcorrelation_LDA(rs);
    for (int i=0;i<npoints;++i){
        ex+=exx(i)*density(i)*dV;
        ec+=ecc(i)*density(i)*dV;
    }

    SparseMatrix<double> vxc(npoints,npoints);
    vxc.reserve(VectorXi::Constant(npoints, 3));
    for (int i = 0; i < npoints; ++i) {
        vxc.insert(i, i) = vx(i) + vc(i);
    }
    return { vxc, ex + ec };
}
MatrixXd normalize(MatrixXd& psiM, double dV) {
    int nrows = psiM.rows(), ncols = psiM.cols();
    MatrixXd psi(nrows, ncols);
    
    for (int i = 0; i < ncols; ++i) {
        psi.col(i) = psiM.col(i) / sqrt(dV);
    }
    return psi;
}

VectorXd get_density(MatrixXd& psi, int nelect) {
    int nrows = psi.rows(), ncols = psi.cols();
    VectorXd rho = VectorXd::Zero(nrows);
    for (int i = 0; i < nelect / 2; ++i) {
        for (int j=0;j<psi.rows();++j){
            rho(j)+=2 * psi(j,i)*psi(j,i);
        }
        
    }
    return rho;
}

int main() {
    
    VectorXd charge(1);
    charge << 2; // for He
    int nelect = charge.sum();
    
    Vector3i dense(50, 50, 50);
    
    Vector3d centers(0., 0., 0.);
    Vector2d corners{-5,5};
  
    VectorXd px = VectorXd::LinSpaced(dense[0], corners[0], corners[1]);
    VectorXd py = VectorXd::LinSpaced(dense[1], corners[0], corners[1]);
    VectorXd pz = VectorXd::LinSpaced(dense[2], corners[0], corners[1]);
    double dx=px[1] - px[0], dy=py[1] - py[0], dz=pz[1] - pz[0];

    Vector3d dr(dx,dy,dz);

    double dV = dr.prod();
    int nx = dense(0), ny = dense(1), nz = dense(2);

    vector<VectorXd> grid(3);
    grid[0] = px;
    grid[1] = py;
    grid[2] = pz;
    
    SparseMatrix<double> Laplacian = laplacian(dense, dr);
   
    
    SparseMatrix<double> T = kinetic_matrix(Laplacian);
   
    SparseMatrix<double> V = nuclear_potential(centers, charge, dense, grid);
    
    SparseMatrix<double> H = T + V;

    int counter = 0;
    double E0 = 0;
    double etol=1e-7;
    double ediff=1e6;
    
    
    while (ediff > etol) {
        Spectra::SparseGenMatProd<double> op(H);
       
        Spectra::GenEigsSolver<Spectra::SparseGenMatProd<double>> eigs(op, 1, 20);
        eigs.init();
        int nconv = eigs.compute(Spectra::SortRule::SmallestReal);
       
        
        VectorXcd Eim=eigs.eigenvalues();
        MatrixXcd psiMim = eigs.eigenvectors();
    
        VectorXd E = Eim.real();
        MatrixXd psiM = psiMim.real();
        
        
        counter++;
        MatrixXd psi = normalize(psiM, dV);
       
        double tota=0;
        for (int i=0; i<psi.rows();++i){
            tota+=psi(i,0)*psi(i,0)*dV;
        }
       
        VectorXd rho = get_density(psi, nelect);
        
 
        
        
        pair<SparseMatrix<double>, double> hartree = hartree_potential(dense, Laplacian, rho, dV);
        cout << "at least hartree works" << endl;
        SparseMatrix<double> vh = hartree.first;
        double eh = hartree.second;
      
       
        
        pair<SparseMatrix<double>, double> exchange_correlation = ex_co(rho, dense,dV);
        SparseMatrix<double> vxc = exchange_correlation.first;
        double exc = exchange_correlation.second;
        
        
        
        double eT = kinetic_energy(T, psi, nelect,dV);
        double eV = nuclear_energy(V, rho, dV);
       
        
        double E_total = eT + eV + eh+exc;
        cout << E_total<< endl;
        
        ediff = abs(E0 - E_total);
        E0 = E_total;

        
        H = T + V + vh + vxc ;

        
        if (counter == 20) {
            cout << "Did not converge" << endl;
            break;
        }
    }

    // Print final energy and convergence status
    cout << "Final Energy: " << E0 << endl;
    cout << "Converged" << endl;
    
    return 0;
}
