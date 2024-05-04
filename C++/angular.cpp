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

double factorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n * factorial(n - 2);
}

double boys(double x, int n) {
    if (x == 0) {
        return 1.0 / (2 * n + 1);
    } else {
        return boost::math::tgamma_lower(n+0.5,x)/(2*std::pow(x,n+0.5));
    }
}
double npdot(std::vector<double> R1,std::vector<double> R2){
    //std::cout<<R1.size()<<std::endl;
    double Rdot=0;
    for (int i=0;i<R1.size();++i){
        Rdot+=R1[i]*R2[i];
    }
    return Rdot;
}

std::vector<double> vector_diff(std::vector<double> R1,std::vector<double> R2){
    std::vector<double> R(R1.size());
    //std::cout<<R1.size()<<std::endl;
    for (int i=0;i<R1.size();++i){
        R[i]=R1[i]-R2[i];
    }
    return R;
}

double compute_s(double& A,double& B,double& P,double& p,int& l1, int& l2){
 if (l1 < 0 || l2 < 0) {
        return 0.0;
    }
if (l1 == 0 && l2 == 0) {
        return 1.0;
    }

if (l2 == 0) {
        int l1m1=l1-1;
        int l1m2=l1-2;
        int zerot=0;
        double s = -(A - P) * compute_s(A, B, P, p, l1m1, zerot) + (l1 - 1) / (2 * p) * compute_s(A, B, P, p, l1m2, zerot);
        return s;
    }
int l1p1=l1+1;
int l2m1=l2-1;
double s = compute_s(A, B, P, p, l1p1, l2m1) + (A - B) * compute_s(A, B, P, p, l1, l2m1);
return s;
}

double overlap(Gaussian gi, Gaussian gj) {
    double ai = gi.a;
    double aj = gj.a;
    std::vector<int> l1 = gi.l;
    std::vector<int> l2 = gj.l;
    std::vector<double> A = gi.r;
    std::vector<double> B = gj.r;

    double Norm=std::pow(M_PI/(ai+aj),1.5);
    double Eab= std::exp(-ai*aj/(ai+aj) * npdot(vector_diff(A,B),vector_diff(A,B)) );

    std::vector<double> P(3);
    for (int i=0;i<3 ;++i)
    {
        P[i]=(ai*A[i]+aj*B[i])/(ai+aj);
    }
    double p =ai+aj;
    //std::cout<<Norm<<std::endl;
    return Eab*Norm*compute_s(A[0],B[0],P[0],p,l1[0],l2[0])*compute_s(A[1],B[1],P[1],p,l1[1],l2[1])*compute_s(A[2],B[2],P[2],p,l1[2],l2[2]);
}

double compute_k(double& A,double& B,double& P,double& a,double& b,int& l1, int&l2){
int l1m1=l1-1;
int l1p1=l1+1;
int l2m1=l2-1;
int l2p1=l2+1;
int one=1;
double p =a+b;

if (l2 <= 0) {
        double k = -l1 * b * compute_s(A, B, P, p, l1m1, one) + 2 * a * b * compute_s(A, B, P, p, l1p1, one);
        return k;
    }
if (l1 <= 0) {
        double k = -a * l2 * compute_s(A, B, P, p, one, l2m1) + 2 * a * b * compute_s(A, B, P, p, one, l2p1);
        return k;
    }
double k = 0.5 * (l1 * l2 * compute_s(A, B, P, p, l1m1, l2m1) - 2 * a * l2 * compute_s(A, B, P, p, l1p1, l2m1) -
                      2 * l1 * b * compute_s(A, B, P, p, l1m1, l2p1) + 4 * a * b * compute_s(A, B, P, p, l1p1, l2p1));
    return k;
}


double kinetic(Gaussian gi, Gaussian gj) {
    double a = gi.a;
    double b = gj.a;
    std::vector<int> l1 = gi.l;
    std::vector<int> l2 = gj.l;
    std::vector<double> A = gi.r;
    std::vector<double> B = gj.r;

    double Norm=std::pow(M_PI/(a+b),1.5);
    double Eab= std::exp(-a*b/(a+b) * npdot(vector_diff(A,B),vector_diff(A,B)) );

    std::vector<double> P(3);
    for (int i=0;i<3 ;++i)
    {
        P[i]=(a*A[i]+b*B[i])/(a+b);
    }
    double p =a+b;

    return Eab*Norm* (compute_k(A[0],B[0],P[0],a,b,l1[0],l2[0])   * compute_s(A[1],B[1],P[1],p,l1[1],l2[1])  * compute_s(A[2],B[2],P[2],p,l1[2],l2[2]) +
                          compute_s(A[0],B[0],P[0],p,l1[0],l2[0]) * compute_k(A[1],B[1],P[1],a,b,l1[1],l2[1])    * compute_s(A[2],B[2],P[2],p,l1[2],l2[2]) +
                          compute_s(A[0],B[0],P[0],p,l1[0],l2[0]) * compute_s(A[1],B[1],P[1],p,l1[1],l2[1])  * compute_k(A[2],B[2],P[2],a,b,l1[2],l2[2]) );   
}

//for s its cool, not yet checked for more
std::vector<int> low(std::vector<int>& l,int crd, int size){
    std::vector<int> llow=l;
    llow[crd]+=size;
    return llow;
}

int find_crd(std::vector<int>& l){
    for (int i=0;i<l.size() ;++i){
        if (l[i]>0){return i;}
    }
    return 9000;
}

bool check_zero(std::vector<int> l){
    for (int i=0;i<l.size() ;++i){
        if (l[i]!=0) {return false;}
    }
    return true;
}

bool check_below(std::vector<int> l){
    for (int i=0;i<l.size() ;++i){
        if (l[i]<0) {return true;}
    }
    return false;
}

bool check_above(std::vector<int> l){
    for (int i=0;i<l.size() ;++i){
        if (l[i]>0) {return true;}
    }
    return false;
}

double nuclear00(double& a,double& b,std::vector<double> A,std::vector<double> B,std::vector<double> R,int n){
    double p =a+b;
    std::vector<double> P(3);
    for (int i=0;i<3 ;++i)
    {
        P[i]=(a*A[i]+b*B[i])/(a+b);
    }
    double T;
    T= p*npdot(vector_diff(P,R),vector_diff(P,R));

    double s00=std::pow(M_PI/p,1.5) * std::exp(-a*b/(a+b) *npdot(vector_diff(A,B),vector_diff(A,B)) );

    return 2*std::pow(p/M_PI,0.5)*s00*boys(T,n);
}

double nuclearVRR(double& a,double& b,std::vector<double> A,std::vector<double> B,std::vector<double> R,std::vector<int> la,int n){
    double p =a+b;
    std::vector<double> P(3);
    for (int i=0;i<3 ;++i)
    {
        P[i]=(a*A[i]+b*B[i])/(a+b);
    }
    //std::cout<<P[0]<< " "<<P[1]<<" "<<P[2] <<std::endl;

    if (check_zero(la)==true){
        return nuclear00(a,b,A,B,R,n);
    }
    
    if (check_below(la)==true){
        return 0;
    }
    int crd=find_crd(la);
    //int np1=n+1; //but cant in previous one ??
    double nu=vector_diff(P,A)[crd] * nuclearVRR(a, b, A, B,R,low(la,crd,-1), n) - vector_diff(P,R)[crd]*nuclearVRR(a, b, A, B,R,low(la,crd,-1), n+1);
    
    //int lowered=la[crd]-1;
    if (la[crd]-1>0){
        nu+= (la[crd]-1)/(2*p) * ( nuclearVRR(a, b, A, B,R,low(la,crd,-2), n) -  nuclearVRR(a, b, A, B,R,low(la,crd,-2), n+1) );
    }
    
    return nu;
}



double contractNU(std::vector<Gaussian> cGTO1,std::vector<Gaussian> cGTO2,std::vector<int> la,std::vector<double> R,int n){
    double contracted_VRR=0;
    
    for (int p=0;p<cGTO1.size();++p){
        for (int q=0;q<cGTO2.size();++q){
            double a = cGTO1[p].a;
            double b = cGTO2[q].a;
            std::vector<double> A = cGTO1[p].r;
            std::vector<double> B = cGTO2[q].r;
            /*
            std::cout<<a<<std::endl;
            std::cout<<b<<std::endl;
            std::cout<<A[0]<<A[1]<<A[2]<<std::endl;
            std::cout<<B[0]<<B[1]<<B[2]<<std::endl;
            std::cout<<R[0]<<R[1]<<R[2]<<std::endl;
            */
            contracted_VRR+=(cGTO1[p].N*cGTO2[q].N)*(cGTO1[p].c*cGTO2[q].c) * nuclearVRR(a, b, A, B, R, la, n);
        }
    }
    
    return contracted_VRR;
}


double nuclearHRR(std::vector<Gaussian> cGTO1,std::vector<Gaussian> cGTO2,std::vector<int> la,std::vector<int> lb,std::vector<double> R){
    std::vector<double> A = cGTO1[0].r;
    std::vector<double> B = cGTO2[0].r;
    double nu=0;
        if (check_above(lb)==true) {
            int crd=find_crd(lb);
            nu+=nuclearHRR(cGTO1,cGTO2,low(la,crd,+1), low(lb,crd,-1),R) + vector_diff(A,B)[crd]*nuclearHRR(cGTO1,cGTO2, la,low(lb,crd,-1),R);
            return nu;
        }
     nu+=contractNU(cGTO1,cGTO2, la,R, 0);
     return nu;
}


double eri0(double& a,double& b,double& c,double& d,std::vector<double> A,std::vector<double> B,std::vector<double> C,std::vector<double> D,int n){
    double pab =a+b;
    double pcd=c+d;
    std::vector<double> P(3);
    std::vector<double> Q(3);
    for (int i=0;i<3 ;++i)
    {
        P[i]=(a*A[i]+b*B[i])/(a+b);
        Q[i]=(c*C[i]+d*D[i])/(c+d);
    }

    double p = pab*pcd/(pab+pcd);
    double T;
    T= p*npdot(vector_diff(P,Q),vector_diff(P,Q));

    double s00ab= std::exp(-a*b/pab *npdot(vector_diff(A,B),vector_diff(A,B)) );
    double s00cd= std::exp(-c*d/pcd *npdot(vector_diff(C,D),vector_diff(C,D)) );


    return 2*std::pow(M_PI,2.5)/(pab*pcd*std::sqrt(pab+pcd))*s00ab*s00cd*boys(T,n);
}

double VRR(double& a,double& b,double& c,double& d,std::vector<double> A,std::vector<double> B,std::vector<double> C,std::vector<double> D,std::vector<int> la,std::vector<int> lc,int n){
    double pab =a+b;
    double pcd=c+d;
    std::vector<double> P(3);
    std::vector<double> Q(3);
    for (int i=0;i<3 ;++i)
    {
        P[i]=(a*A[i] + b*B[i])/(pab);
        Q[i]=(c*C[i] + d*D[i])/(pcd);
    }

    if (check_zero(la)==true && check_zero(lc)==true){
        return eri0(a,b,c,d,A,B,C,D,n);
    }
    
    if (check_below(la)==true || check_below(lc)==true){
        return 0;
    }
    if (check_above(lc)==true){
        int crd=find_crd(lc);
        double eri = vector_diff(Q,C)[crd] * VRR(a, b, c, d, A, B, C, D, la, low(lc,crd,-1), n) + pab/(pab+pcd)*vector_diff(P,Q)[crd] * VRR(a, b, c, d, A, B, C, D, la, low(lc,crd,-1), n+1);

        if ((lc[crd]-1) > 0){
            eri+= (lc[crd]-1)/(2*pcd) * ( VRR(a, b, c, d, A, B, C, D, la, low(lc,crd,-2), n) - pab/(pab+pcd)* VRR(a, b, c, d, A, B, C, D, la, low(lc,crd,-2), n+1) );
        }
        if (la[crd]> 0){
            eri+= la[crd]/(2*(pab+pcd)) * VRR(a, b, c, d, A, B, C, D, low(la,crd,-1), low(lc,crd,-1), n+1) ;
        }
        return eri;
    }

    int crd=find_crd(la);
    double eri=vector_diff(P,A)[crd] * VRR(a, b, c, d, A, B, C, D, low(la,crd,-1), lc, n) - pcd/(pab+pcd)*vector_diff(P,Q)[crd] * VRR(a, b, c, d, A, B, C, D, low(la,crd,-1), lc, n+1);
    if ((la[crd]-1)>0){
        eri+= (la[crd]-1)/(2*pab) * ( VRR(a, b, c, d, A, B, C, D, low(la,crd,-2), lc, n) - pcd/(pab+pcd)* VRR(a, b, c, d, A, B, C, D, low(la,crd,-2), lc, n+1) );
        }
    if (lc[crd]>0){
        eri+= lc[crd]/(2*(pab+pcd)) * VRR(a, b, c, d, A, B, C, D, low(la,crd,-1), low(lc,crd,-1), n+1);}
    return eri;
}

double contract(std::vector<Gaussian> cGTO1,std::vector<Gaussian> cGTO2,std::vector<Gaussian> cGTO3,std::vector<Gaussian> cGTO4,std::vector<int> la,std::vector<int> lc,int n){
    double contracted_VRR=0;
    
    for (int p=0; p<cGTO1.size(); ++p){
        for (int q=0; q<cGTO2.size(); ++q){
            for (int r=0; r<cGTO3.size(); ++r){
                for (int v=0; v<cGTO4.size(); ++v){
                    double a = cGTO1[p].a;
                    double b = cGTO2[q].a;
                    double c = cGTO3[r].a;
                    double d = cGTO4[v].a;
                    std::vector<double> A = cGTO1[p].r;
                    std::vector<double> B = cGTO2[q].r;
                    std::vector<double> C = cGTO3[r].r;
                    std::vector<double> D = cGTO4[v].r;
                    contracted_VRR+=(cGTO1[p].N*cGTO2[q].N*cGTO3[r].N*cGTO4[v].N)*(cGTO1[p].c*cGTO2[q].c*cGTO3[r].c*cGTO4[v].c) * VRR(a, b, c, d, A, B, C, D, la, lc, n);

                }}}}
    return contracted_VRR;

}

double HRR(std::vector<Gaussian> cGTO1,std::vector<Gaussian> cGTO2,std::vector<Gaussian> cGTO3,std::vector<Gaussian> cGTO4,std::vector<int> la,std::vector<int> lb,std::vector<int> lc,std::vector<int> ld){
    std::vector<double> A = cGTO1[0].r;
    std::vector<double> B = cGTO2[0].r;
    std::vector<double> C = cGTO3[0].r;
    std::vector<double> D = cGTO4[0].r;
    

    if (check_above(lb)==true){
        int crd=find_crd(lb);
        double e=HRR(cGTO1,cGTO2,cGTO3,cGTO4,low(la,crd,+1), low(lb,crd,-1),lc,ld) + vector_diff(A,B)[crd]*HRR(cGTO1,cGTO2,cGTO3,cGTO4, la,low(lb,crd,-1), lc,ld); 
        return e;
    }
    if (check_above(ld)==true){
        int crd=find_crd(ld);
        double e=HRR(cGTO1,cGTO2,cGTO3,cGTO4, la,lb, low(lc,crd,+1), low(ld,crd,-1)) + vector_diff(C,D)[crd]*HRR(cGTO1,cGTO2,cGTO3,cGTO4, la, lb,lc, low(ld,crd,-1)) ; 
        return e;
    }
    double e=contract(cGTO1,cGTO2,cGTO3,cGTO4, la, lc, 0);
    return e;


}