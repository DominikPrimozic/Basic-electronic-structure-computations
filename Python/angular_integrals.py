# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:38:14 2024

@author: domin
"""
import numpy as np
from scipy import special
def boys(x,n):
    
    if x==0:
        return 1/(2*n+1)
    else:
        return special.gammainc(n+0.5,x) * special.gamma(n+0.5) * (1/(2*x**(n+0.5)))
    
def low(arr,idx,num):
    arr=arr.copy()
    arr[idx]+=num
    return arr

"""
electron repulsion integrals are evaluated by Obara-Saika scheme
[1] S. Obara, A. Saika; Efficient recursive computation of molecular integrals over Cartesian Gaussian functions. J. Chem. Phys. 1 April 1986; 84 (7): 3963–3974. https://doi.org/10.1063/1.450106
[2] Zhang, J. LIBRETA: Computerized Optimization and Code Synthesis for Electron Repulsion Integral Evaluation. J. Chem. Theory Comput. 2018, 14, 572-587.
"""
def eri0(a,b,c,d,A,B,C,D,n):
    pab=a+b
    pcd=c+d
    
    s_0_0_ab=np.exp(-a*b/(pab) * np.dot(A-B,A-B))
    s_0_0_cd=np.exp(-c*d/(pcd) * np.dot(C-D,C-D))   
    
    P=(a*A+b*B)/(pab)
    Q=(c*C+d*D)/(pcd)
    
    p=pab*pcd/(pab+pcd)
    T=p*np.dot(P-Q,P-Q)
    e=2*np.pi**(5/2) / (pab*pcd*np.sqrt(pab+pcd)) * s_0_0_ab*s_0_0_cd*boys(T,n)
    return e

def VRR(a,b,c,d,A,B,C,D,la,lc,n): 
    pab=a+b
    pcd=c+d
    P=(a*A+b*B)/(pab)
    Q=(c*C+d*D)/(pcd)
    if not np.array([la,lc]).any():
        eri=eri0(a,b,c,d,A,B,C,D,n)
        
        return eri
    #if it call below 0 return 0, to avoid computing brezveze
    if np.any(np.array([la,lc])<0):
        return 0

    if np.any(lc>0):
        crd=np.where(lc!=0)[0][0] #always take the first one out, it will build down with recurison, when all are 0 it wont get to here   
        
        eri=(Q-C)[crd] * VRR(a, b, c, d, A, B, C, D, la, low(lc,crd,-1), n) + pab/(pab+pcd)*(P-Q)[crd] * VRR(a, b, c, d, A, B, C, D, la, low(lc,crd,-1), n+1)
        
        if (lc[crd]-1)>0:
            eri+= (lc[crd]-1)/(2*pcd) * ( VRR(a, b, c, d, A, B, C, D, la, low(lc,crd,-2), n) - pab/(pab+pcd)* VRR(a, b, c, d, A, B, C, D, la, low(lc,crd,-2), n+1) )
            
        if (la[crd])>0:
            eri+= la[crd]/(2*(pab+pcd)) * VRR(a, b, c, d, A, B, C, D, low(la,crd,-1), low(lc,crd,-1), n+1) 
            
        return eri
    
    crd=np.where(la!=0)[0][0] 
    eri=(P-A)[crd] * VRR(a, b, c, d, A, B, C, D, low(la,crd,-1), lc, n) - pcd/(pab+pcd)*(P-Q)[crd] * VRR(a, b, c, d, A, B, C, D, low(la,crd,-1), lc, n+1)
    if (la[crd]-1)>0:
        eri+= (la[crd]-1)/(2*pab) * ( VRR(a, b, c, d, A, B, C, D, low(la,crd,-2), lc, n) - pcd/(pab+pcd)* VRR(a, b, c, d, A, B, C, D, low(la,crd,-2), lc, n+1) )
    if (lc[crd])>0:
        eri+= lc[crd]/(2*(pab+pcd)) * VRR(a, b, c, d, A, B, C, D, low(la,crd,-1), low(lc,crd,-1), n+1) 
    return eri


def contract(cGTO1,cGTO2,cGTO3,cGTO4,la,lc,n): #these are contracted gaussians so [g1,g2,g3] is cGTO
    contracted_VRR=0
    for p in range(len(cGTO1)):
        for q in range(len(cGTO2)):
            for r in range(len(cGTO3)):
                for v in range(len(cGTO4)):   
                    #l should be constant so could call it cGTO[0].l 
                    a,b,c,d=    cGTO1[p].a, cGTO2[q].a, cGTO3[r].a, cGTO4[v].a
                    A,B,C,D=    cGTO1[p].r, cGTO2[q].r, cGTO3[r].r, cGTO4[v].r
                    
                   
                    contracted_VRR+=(cGTO1[p].N*cGTO2[q].N*cGTO3[r].N*cGTO4[v].N)*(cGTO1[p].c*cGTO2[q].c*cGTO3[r].c*cGTO4[v].c) * VRR(a, b, c, d, A, B, C, D, la, lc, n)
                    
    return contracted_VRR

def HRR(cGTO1,cGTO2,cGTO3,cGTO4,la,lb,lc,ld):  
    """
    takes the four contracted gaussian object from molecule class and their angular momentums
    """
    A,B,C,D=    cGTO1[0].r, cGTO2[0].r, cGTO3[0].r, cGTO4[0].r
    
    if np.any(lb>0): 
        crd=np.where(lb!=0)[0][0] 
       
        #e=contract(cGTO1,cGTO2,cGTO3,cGTO4,low(la,crd,+1), lc, 0) + (A-B)[crd]*contract(cGTO1,cGTO2,cGTO3,cGTO4, la, lc, 0) 
        e=HRR(cGTO1,cGTO2,cGTO3,cGTO4,low(la,crd,+1), low(lb,crd,-1),lc,ld) + (A-B)[crd]*HRR(cGTO1,cGTO2,cGTO3,cGTO4, la,low(lb,crd,-1), lc,ld) 
       
        return e
    if np.any(ld>0):
        crd=np.where(ld!=0)[0][0]
        
        #e=contract(cGTO1,cGTO2,cGTO3,cGTO4, la, low(lc,crd,+1), 0) + (C-D)[crd]*contract(cGTO1,cGTO2,cGTO3,cGTO4, la, lc, 0) 
        e=HRR(cGTO1,cGTO2,cGTO3,cGTO4, la,lb, low(lc,crd,+1), low(ld,crd,-1)) + (C-D)[crd]*HRR(cGTO1,cGTO2,cGTO3,cGTO4, la, lb,lc, low(ld,crd,-1)) 
        
        return e
    
    e=contract(cGTO1,cGTO2,cGTO3,cGTO4, la, lc, 0)
    
    return e

"""
computed via recurssion given at:
Hô, M., & Hernández-Pérez, J.-M. (2012). Evaluation of Gaussian Molecular Integrals. The Mathematica Journal, 14. https://doi.org/10.3888/tmj.14-3
"""

def compute_s_2(A,B,P,p,l1,l2):
    if l1<0 or l2<0:
        s=0
        return s
    
    if l1==0 and l2==0:
        s=1
        return s
    
    if l2==0: #could be ==0 cause it shouldnt call less
        s=s=-(A-P)*compute_s_2(A,B,P,p,l1-1,0)+ (l1-1)/(2*p) * compute_s_2(A, B, P, p, l1-2, 0)
        return s
    
    s=compute_s_2(A,B,P,p,l1+1,l2-1) + (A-B)*compute_s_2(A,B,P,p,l1,l2-1)
    
    return s

def overlap(g1,g2):
    """
    takes two gaussian objects
    """
    Norm=(np.pi/(g1.a+g2.a))**(3/2)
    Eab=np.exp(-g1.a*g2.a/(g1.a+g2.a) * np.dot(g1.r-g2.r, g1.r-g2.r))
    P=(g1.a*g1.r+g2.a*g2.r)/(g1.a+g2.a)
    p=g1.a+g2.a
    A,B=g1.r,g2.r
    
    l1,l2=g1.l,g2.l
    
    return Eab*Norm*compute_s_2(A[0],B[0],P[0],p,l1[0],l2[0])*compute_s_2(A[1],B[1],P[1],p,l1[1],l2[1])*compute_s_2(A[2],B[2],P[2],p,l1[2],l2[2])           

"""
computed via recurssion given at:
Hô, M., & Hernández-Pérez, J.-M. (2013). Evaluation of Gaussian Molecular Integrals II. The Mathematica Journal, 15. https://doi.org/10.3888/tmj.15-1   
"""
def compute_k(A,B,P,a,b,l1,l2):
    if l2<=0:
        k=-l1*b*compute_s_2(A,B,P,a+b,l1-1,1) + 2*a*b*compute_s_2(A,B,P,a+b,l1+1,1)
        return k
    if l1<=0:
        k=-a*l2*compute_s_2(A,B,P,a+b,1,l2-1) + 2*a*b*compute_s_2(A,B,P,a+b,1,l2+1)
        return k
    k=1/2 * (l1*l2*compute_s_2(A, B, P, a+b, l1-1, l2-1) - 2*a*l2*compute_s_2(A, B, P, a+b, l1+1, l2-1) -\
             2*l1*b*compute_s_2(A, B, P, a+b, l1-1, l2+1) + 4*a*b*compute_s_2(A, B, P, a+b, l1+1, l2+1)  )
    return k

def kinetic(g1,g2):
    """
    takes two gaussian objects
    """
    Norm=(np.pi/(g1.a+g2.a))**(3/2)
    Eab=np.exp(-g1.a*g2.a/(g1.a+g2.a) * np.dot(g1.r-g2.r, g1.r-g2.r))
    P=(g1.a*g1.r+g2.a*g2.r)/(g1.a+g2.a)
    a,b=g1.a,g2.a
    A,B=g1.r,g2.r
    
    l1,l2=g1.l,g2.l

    return Eab*Norm* (    compute_k(A[0],B[0],P[0],a,b,l1[0],l2[0])   * compute_s_2(A[1],B[1],P[1],a+b,l1[1],l2[1])  * compute_s_2(A[2],B[2],P[2],a+b,l1[2],l2[2]) +\
                          compute_s_2(A[0],B[0],P[0],a+b,l1[0],l2[0]) * compute_k(A[1],B[1],P[1],a,b,l1[1],l2[1])    * compute_s_2(A[2],B[2],P[2],a+b,l1[2],l2[2]) +\
                          compute_s_2(A[0],B[0],P[0],a+b,l1[0],l2[0]) * compute_s_2(A[1],B[1],P[1],a+b,l1[1],l2[1])  * compute_k(A[2],B[2],P[2],a,b,l1[2],l2[2]) )


"""
computed via recurssion given at:
Hô, M., & Hernándes-Pérez, J. M. (2014). Evaluation of Gaussian Molecular Integrals - III. Nuclear-Electron Attraction Integrals. Math. J., 16(2).    
"""
def compute_n(A,B,P,a,b,l1,l2,R,t):
    if l1<0: #maybe and and l2==0 or l2<0
        n=0
        return n
    if l1==0 and l2==0: #maybe l2<=0
        n=1
        return n
    if l2==0: #dont think less can be called here            #without sympy it has to be less general with n returning a list [(0,value),(1,value),(2,value),...]
            n=-(A-P + t**2 *(P-R))*compute_n(A,B,P,a,b,l1-1,0,R,t) + (l1-1)/(2*(a+b)) * (1-t**2) * compute_n(A, B, P, a, b, l1-2, 0, R, t)
            return n
    n=compute_n(A, B, P, a, b, l1+1, l2-1, R, t) + (A-B)*compute_n(A, B, P, a, b, l1, l2-1, R, t)
    return n

def nuclear(g1,g2,R): 
    """
    takes two gaussian objects and coordinates of nucleus
    """
    Norm=(np.pi/(g1.a+g2.a))**(3/2) 
    Eab=np.exp(-g1.a*g2.a/(g1.a+g2.a) * np.dot(g1.r-g2.r, g1.r-g2.r))
    P=(g1.a*g1.r+g2.a*g2.r)/(g1.a+g2.a)
    a,b=g1.a,g2.a
    A,B=g1.r,g2.r
    l1,l2=g1.l,g2.l
    T=(g1.a+g2.a)*np.dot(P-R,P-R)
    
    import sympy as sm
    t=sm.Symbol("t")
    #get full =nx*ny*nz
    n= compute_n(A[0],B[0],P[0],a,b,l1[0],l2[0],R[0],t) * compute_n(A[1],B[1],P[1],a,b,l1[1],l2[1],R[1],t) * compute_n(A[2],B[2],P[2],a,b,l1[2],l2[2],R[2],t)
    
    #t_max=sm.degree(n, gen=t) #cause they all have to be in powers of 2
    if np.sum(l1+l2)==0:
        n_dec={0:n}
    else:
        if n==0:
            n_dec={0:0}
        else:
            n_dec={term.as_poly(t).degree(): term.as_poly(t).coeffs()[0] for term in sm.Add.make_args(n)} #now just take out t_max-2 steps
   
    ne_int=0
    for key in n_dec.keys():
        ne_int+=n_dec[key] * boys(T,key/2)
    
    NE=Eab*2*np.pi/(a+b) * ne_int
    return NE

"""
faster nuclear-electron repulsion integrals
computed from formula given in:
[1] S. Obara, A. Saika; Efficient recursive computation of molecular integrals over Cartesian Gaussian functions. J. Chem. Phys. 1 April 1986; 84 (7): 3963–3974. https://doi.org/10.1063/1.450106
"""

def nuclear00(a,b,A,B,R,n):          
    p=(a+b) 
    P=(a*A+b*B)/(a+b)
    T=p* np.dot(P-R,P-R)
    s_0_0_ab=(np.pi/p)**(3/2)*np.exp(-a*b/(a+b) * np.dot(A-B,A-B))
    
    n_00=2*(p/np.pi)**(1/2) * s_0_0_ab * boys(T,n)
    return n_00

def nuclearVRR(a,b,A,B,R,la,n): 
    p=a+b
    P=(a*A+b*B)/(a+b)
    if not la.any():
        nu=nuclear00(a, b, A, B, R, n)
        return nu
    if np.any(la<0):
        return 0
    #if np.any(la>0):
    crd=np.where(la!=0)[0][0] #always take the first one out, it will boild down with recurison, when all are 0 it wont get to here   
    nu=(P-A)[crd] * nuclearVRR(a, b, A, B,R,low(la,crd,-1), n) - (P-R)[crd]*nuclearVRR(a, b, A, B,R,low(la,crd,-1), n+1)
    
    if (la[crd]-1)>0:
        nu+= (la[crd]-1)/(2*p) * ( nuclearVRR(a, b, A, B,R,low(la,crd,-2), n) -  nuclearVRR(a, b, A, B,R,low(la,crd,-2), n+1) )
        
    return nu
    
def nuclearHRR(cGTO1,cGTO2,la,lb,R):   
    """
    takes two contracted gaussian objects from molecule, their angular momenta and coordinates of nuclues
    """
    A,B=cGTO1[0].r, cGTO2[0].r
    
    if np.any(lb>0):
        crd=np.where(lb!=0)[0][0] 
        
        nu=nuclearHRR(cGTO1,cGTO2,low(la,crd,+1), low(lb,crd,-1),R) + (A-B)[crd]*nuclearHRR(cGTO1,cGTO2, la,low(lb,crd,-1),R) 
        return nu
    nu=contractNU(cGTO1,cGTO2, la,R, 0)
    return nu
 

def contractNU(cGTO1,cGTO2,la,R,n): #these are contracted gaussians so [g1,g2,g3] is cGTO
    contracted_VRR=0
    for p in range(len(cGTO1)):
        for q in range(len(cGTO2)):
                   
                    a,b=    cGTO1[p].a, cGTO2[q].a,
                    A,B=    cGTO1[p].r, cGTO2[q].r
                   
                    contracted_VRR+=(cGTO1[p].N*cGTO2[q].N)*(cGTO1[p].c*cGTO2[q].c) * nuclearVRR(a, b, A, B, R, la, n)
                    
    return contracted_VRR
