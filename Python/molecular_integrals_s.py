# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:33:27 2024

@author: domin
"""

import numpy as np
from scipy import special
import scipy.linalg as sp

from moleculeS import molecule

def overlap(gi,gj):
    
    ai=gi.a
    aj=gj.a 
    return gi.N * gj.N * gi.c * gj.c * np.pi**(3/2)/ (ai+aj)**(3/2) * np.exp(-ai*aj/(ai+aj) * np.dot(gi.r-gj.r, gi.r-gj.r))


def boys(x,n):
    
    if x==0:
        return 1/(2*n+1)
    else:
        return special.gammainc(n+0.5,x) * special.gamma(n+0.5) * (1/(2*x**(n+0.5)))

class molecular_integrals_s(molecule):
    """
    integrals were evaluated as per:
    Piela, L. (2014). Molecular Integrals with Gaussian Type Orbitals 1s. Ideas of Quantum Chemistry, e131–e132. https://doi.org/10.1016/B978-0-444-59436-5.00036-2
    """
    def __init__(self,alphas,centers, Z,coefficients, *args):
        molecule.__init__(self,alphas,centers,Z,coefficients, *args)
        self.overlap_matrix()
        self.kinetic_matrix()
        self.electron_nuclear_matrix()
        self.electron_electron_repulsion_matrix()
        self.H_core=self.T+self.V_ne
        self.nuclear_nuclear_repulsion()
    
    
    def overlap_matrix(self): #optimize the loops (less looping)
        n_mol_basis=len(self.mol) #AO basis
        S=np.zeros((n_mol_basis,n_mol_basis)) 
        
        for i in range(n_mol_basis):
            for j in range(n_mol_basis):
                
                n_exp_basis_i=len(self.mol[i]) #gaussian expansion of each AO
                n_exp_basis_j=len(self.mol[j])
                
                for p in range(n_exp_basis_i):
                    for q in range(n_exp_basis_j):
                        
                        S[i,j]+= overlap(self.mol[i][p],self.mol[j][q])
                        
        self.S=S
        #print(S)
      
    def kinetic_matrix(self):
        n_mol_basis=len(self.mol) #AO basis
        T=np.zeros((n_mol_basis,n_mol_basis)) 
        
        for i in range(n_mol_basis):
            for j in range(n_mol_basis):
                
                n_exp_basis_i=len(self.mol[i]) #gaussian expansion of each AO
                n_exp_basis_j=len(self.mol[j])
                
                for p in range(n_exp_basis_i):
                    for q in range(n_exp_basis_j):
                        Spq=overlap(self.mol[i][p],self.mol[j][q])
                        
                        ap=self.mol[i][p].a
                        aq=self.mol[j][q].a
                        Rp=self.mol[i][p].r
                        Rq=self.mol[j][q].r
                        
                        T[i,j]+= ap*aq/(ap+aq) * (3- 2*ap*aq/(ap+aq) * np.dot(Rp-Rq,Rp-Rq)) * Spq
                            
        self.T=T
        #print(T)
        
    def electron_nuclear_matrix(self):
        n_mol_basis=len(self.mol) #AO basis
        n_atoms=len(self.charges) # or len(alphas)
        V_ne=np.zeros((n_mol_basis,n_mol_basis)) 
        
        #get atomic coordiantes
        atom_coords=self.position  
        
        for atom in range(n_atoms):
            
            for i in range(n_mol_basis):
                for j in range(n_mol_basis):
                    
                    n_exp_basis_i=len(self.mol[i]) #gaussian expansion of each AO
                    n_exp_basis_j=len(self.mol[j])
                    
                    for p in range(n_exp_basis_i):
                        for q in range(n_exp_basis_j):
                            
                            Spq=overlap(self.mol[i][p],self.mol[j][q])
                            
                            ap=self.mol[i][p].a
                            aq=self.mol[j][q].a
                            Rp=self.mol[i][p].r
                            Rq=self.mol[j][q].r
                            
                            Rpq=(ap*Rp + aq*Rq)/(ap+aq)
                            Ra=atom_coords[atom]
                            
                            V_ne[i,j]+=-self.charges[atom] *2* np.sqrt( (ap+aq)/np.pi ) * boys( (ap+aq)*np.dot(Ra-Rpq,Ra-Rpq) , 0 )* Spq
                        
                        
        self.V_ne=V_ne 
        #print(V_ne)

    def electron_electron_repulsion_matrix(self):
        n_mol_basis=len(self.mol) #AO basis
        
        V_ee=np.zeros((n_mol_basis,n_mol_basis,n_mol_basis,n_mol_basis)) #two electron term 
        
        for i in range(n_mol_basis):
            for j in range(n_mol_basis):
                for k in range(n_mol_basis):
                    for l in range(n_mol_basis):
                
                        n_exp_basis_i=len(self.mol[i]) #gaussian expansion of each AO
                        n_exp_basis_j=len(self.mol[j])
                        n_exp_basis_k=len(self.mol[k]) 
                        n_exp_basis_l=len(self.mol[l])
                
                        for p in range(n_exp_basis_i):
                            for q in range(n_exp_basis_j):
                                for r in range(n_exp_basis_k):
                                    for v in range(n_exp_basis_l): #try and do it with less loops (maybe some matrix summations)
                                       
                                        Spq=overlap(self.mol[i][p],self.mol[j][q])
                                        Srv=overlap(self.mol[k][r],self.mol[l][v])
                                        
                                        ap=self.mol[i][p].a
                                        aq=self.mol[j][q].a
                                        ar=self.mol[k][r].a
                                        av=self.mol[l][v].a
                                        
                                        Rp=self.mol[i][p].r
                                        Rq=self.mol[j][q].r
                                        Rr=self.mol[k][r].r
                                        Rv=self.mol[l][v].r
                                        
                                        Rpq=(ap*Rp + aq*Rq)/(ap+aq)
                                        Rrv=(ar*Rr + av*Rv)/(ar+av)
                                        
                                        V_ee[i,j,k,l]+= 2/np.sqrt(np.pi) * np.sqrt(ap+aq)*np.sqrt(ar+av)/np.sqrt(ap+aq+ar+av) * \
                                                        boys( (ap+aq)*(ar+av)/(ap+aq+ar+av) * np.dot(Rpq-Rrv,Rpq-Rrv), 0) * Spq * Srv                      
        self.V_ee=V_ee
        #print(V_ee)

    def nuclear_nuclear_repulsion(self):
        E_nn=0
        for atom1 in range(len(self.position)):
            #print(atom1)
            for atom2 in range(atom1+1,len(self.position)):
                #print(atom2, "a")
                E_nn+=self.charges[atom1]*self.charges[atom2]/ (np.sqrt(np.dot(self.position[atom1]-self.position[atom2],self.position[atom1]-self.position[atom2])))
        self.E_nn=E_nn
