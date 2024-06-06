# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:55:25 2024

@author: domin
"""

import scipy.linalg as sp
import numpy as np

class DIIS():
    """
    This is direct inversion in the iterative subspace implementation
    It was written with the help of tutorial at psi4numpy repository 
    Psi4NumPy: An Interactive Quantum Chemistry Programming Environment for Reference Implementations and Rapid Development Daniel G. A. Smith, Lori A. Burns, Dominic A. Sirianni, Daniel R. Nascimento, Ashutosh Kumar, Andrew M. James, Jeffrey B. Schriber, Tianyuan Zhang, Boyi Zhang, Adam S. Abbott, Eric J. Berquist, Marvin H. Lechner, Leonardo A. Cunha, Alexander G. Heide, Jonathan M. Waldrop, Tyler Y. Takeshita, Asem Alenaizan, Daniel Neuhauser, Rollin A. King, Andrew C. Simmonett, Justin M. Turney, Henry F. Schaefer, Francesco A. Evangelista, A. Eugene DePrince III, T. Daniel Crawford, Konrad Patkowski, and C. David Sherrill Journal of Chemical Theory and Computation, 2018, 14 (7), 3504-3511 DOI: 10.1021/acs.jctc.8b00286
    
    and
    
    P. Pulay. Chem. Phys. Lett. 73, 393-398 (1980)
    """
    def __init__(self):
        self.F=[]
        self.r=[]
        
    def update(self,F,D,S):
        self.F.append(F)
        self.ao_gradient(F, D, S)
        
    
    def ao_gradient(self,F,D,S):
        A=sp.fractional_matrix_power(S, -1/2)
        fds=np.einsum("im,mn,nj->ij",F,D,S,optimize=True) *1/2
        sdf=np.einsum("im,mn,nj->ij",S,D,F,optimize=True) *1/2
        r= np.einsum("mi,mn,nj->ij",A,fds-sdf,A,optimize=True)
       # print("fds",fds)
        #print("sdf",sdf)
        
        self.r.append(r)
        
    def RMSD_check(self):
        return np.mean(self.r[-1]**2)**0.5
    
    def buildB(self):
        dim=len(self.F)+1
        B=np.empty((dim,dim))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for i in range(len(self.F)):
            for j in range(len(self.F)):
                B[i,j] = np.einsum("ij,ij->",self.r[i],self.r[j], optimize=True)
        
        self.B=B
    
    def Pulay(self):
        self.buildB()
        dim=len(self.F)+1
        desna=np.zeros((dim))
        desna[-1]=-1
        coeff = np.linalg.solve(self.B, desna)
        
        self.pc=coeff
    
    def DISS_F(self):
        self.Pulay()
        F=np.zeros_like(self.F[-1])
        for x in range(self.pc.shape[0]-1):
            F+=self.pc[x]*self.F[x]
        
        return F