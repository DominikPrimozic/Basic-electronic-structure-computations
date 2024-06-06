
#include <postHF.h>


Eigen::Tensor<double,4> get_eri_mo(const Eigen::Tensor<double,4> &eri_ao, const Eigen::MatrixXd &coeff, int nao) {
    
    Eigen::Tensor<double,4> tmp_1 (nao, nao, nao, nao);
    Eigen::Tensor<double,4> tmp_2 (nao, nao, nao, nao);
    tmp_1.setZero();
    tmp_2.setZero();

    // First transformation N5
    for (int u = 0; u < nao; ++u)
        for (int v = 0; v < nao; ++v)
            for (int k = 0; k < nao; ++k)
                for (int l = 0; l < nao; ++l)
                    for (int p = 0; p < nao; ++p)
                        tmp_1(p, v, k, l) += eri_ao(u, v, k, l) * coeff(u, p);

    // Second transformation N5
    for (int p = 0; p < nao; ++p)
        for (int v = 0; v < nao; ++v)
            for (int k = 0; k < nao; ++k)
                for (int l = 0; l < nao; ++l)
                    for (int q = 0; q < nao; ++q)
                        tmp_2(p, q, k, l) += tmp_1(p, v, k, l) * coeff(v, q);

    tmp_1.setZero();

    // Third transformation N5
    for (int p = 0; p < nao; ++p)
        for (int q = 0; q < nao; ++q)
            for (int k = 0; k < nao; ++k)
                for (int l = 0; l < nao; ++l)
                    for (int r = 0; r < nao; ++r)
                        tmp_1(p, q, r, l) += tmp_2(p, q, k, l) * coeff(k, r);
    tmp_2.setZero();

    // Fourth transformation N5
    for (int p = 0; p < nao; ++p)
        for (int q = 0; q < nao; ++q)
            for (int r = 0; r < nao; ++r)
                for (int l = 0; l < nao; ++l)
                    for (int s = 0; s < nao; ++s)
                        tmp_2(p, q, r, s) += tmp_1(p, q, r, l) * coeff(l, s);
    
    return tmp_2;
}
double MP2( Eigen::MatrixXd C,  Eigen::VectorXd e, Eigen::Tensor<double,4> eri,int nao,int noc){
    int nvirt = e.size() - noc;
    Eigen::Tensor<double, 4> eri_mo=get_eri_mo(eri,C,nao);

    Eigen::Tensor<double, 4> eri_iajb(noc, nvirt, noc, nvirt);
        for (int i = 0; i < noc; ++i) {
            for (int a = 0; a < nvirt; ++a) {
                for (int j = 0; j < noc; ++j) {
                    for (int b = 0; b < nvirt; ++b) {
                        eri_iajb(i, a, j, b) = eri_mo(i, noc + a, j, noc + b);
                    }
                }
            }
        }

        
        Eigen::Tensor<double, 4> D_iajb(noc, nvirt, noc, nvirt);
        for (int i = 0; i < noc; ++i) {
            for (int a = 0; a < nvirt; ++a) {
                for (int j = 0; j < noc; ++j) {
                    for (int b = 0; b < nvirt; ++b) {
                        D_iajb(i, a, j, b) = e(i) + e(j) - e(noc + a) - e(noc + b);
                    }
                }
            }
        }

       
        double emp2 = 0.0;
        for (int i = 0; i < noc; ++i) {
            for (int a = 0; a < nvirt; ++a) {
                for (int j = 0; j < noc; ++j) {
                    for (int b = 0; b < nvirt; ++b) {
                        double numerator = eri_iajb(i, a, j, b) * (2.0 * eri_iajb(i, a, j, b) - eri_iajb(i, b, j, a));
                        emp2 += numerator / D_iajb(i, a, j, b);
                    }
                }
            }
        }
        return emp2;
}