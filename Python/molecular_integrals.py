# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:35:12 2024

@author: domin
"""
import numpy as np
np.set_printoptions(precision=3)
from scipy import special
import scipy.linalg as sp

from angular_integrals import overlap,kinetic,nuclear,HRR,nuclearHRR

from molecule import molecule

class molecular_integrals(molecule):
    def __init__(self,alphas,centers,l, Z,coefficients, *args):
        molecule.__init__(self,alphas,centers,l,Z,coefficients, *args)
        self.overlap_matrix()
        self.kinetic_matrix()
        self.electron_nuclear_matrix()
        self.electron_electron_repulsion_matrix()
        self.H_core=self.T+self.V_ne
        self.nuclear_nuclear_repulsion()
        
    def overlap_matrix(self):
        n_mol_basis=len(self.mol) #AO basis
        S=np.zeros((n_mol_basis,n_mol_basis)) 

        for i in range(n_mol_basis):
            for j in range(n_mol_basis):
                
                n_exp_basis_i=len(self.mol[i]) #gaussian expansion of each AO
                n_exp_basis_j=len(self.mol[j])
                
                for p in range(n_exp_basis_i):
                    for q in range(n_exp_basis_j):
                        S[i,j]+= self.mol[i][p].N * self.mol[j][q].N * self.mol[i][p].c* self.mol[j][q].c * overlap(self.mol[i][p], self.mol[j][q])
                        
        self.S=S
        
    def kinetic_matrix(self):
        n_mol_basis=len(self.mol)
        K=np.zeros((n_mol_basis,n_mol_basis))    
        for i in range(n_mol_basis):
            for j in range(n_mol_basis):
                
                n_exp_basis_i=len(self.mol[i]) #gaussian expansion of each AO
                n_exp_basis_j=len(self.mol[j])
                
                for p in range(n_exp_basis_i):
                    for q in range(n_exp_basis_j):

                        K[i,j]+= self.mol[i][p].N * self.mol[j][q].N * self.mol[i][p].c* self.mol[j][q].c * kinetic(self.mol[i][p], self.mol[j][q])
                        
        self.T=K
        
    def electron_nuclear_matrix(self):
        n_mol_basis=len(self.mol)
        n_atoms=len(self.charges)  
        V=np.zeros((n_mol_basis,n_mol_basis))   
        """
        for atom in range(n_atoms):
            for i in range(n_mol_basis):
                for j in range(n_mol_basis):
                    
                    n_exp_basis_i=len(self.mol[i]) #gaussian expansion of each AO
                    n_exp_basis_j=len(self.mol[j])
                    
                    for p in range(n_exp_basis_i):
                        for q in range(n_exp_basis_j):
                            #print( molecule[i][p].c*molecule[j][q].c)
                            V[i,j]+= -self.charges[atom]* self.mol[i][p].N * self.mol[j][q].N * self.mol[i][p].c* self.mol[j][q].c * nuclear(self.mol[i][p], self.mol[j][q],self.position[atom])
                           #print(gaussian_overlap(molecule[i][p], molecule[j][q]))    
        self.V_ne=V
        """
        #faster
        for atom in range(n_atoms):
            for i in range(n_mol_basis):
                for j in range(n_mol_basis):
                    gto1=self.mol[i] #gaussian expansion of each AO
                    gto2=self.mol[j]
                    
                    la,lb= gto1[0].l,  gto2[0].l
                    
                    V[i,j]+=-self.charges[atom]*nuclearHRR(gto1,gto2, la,lb,self.position[atom])
        self.V_ne=V
        #"""
    
    def electron_electron_repulsion_matrix(self):
        n_mol_basis=len(self.mol)
        V_ee=np.zeros((n_mol_basis,n_mol_basis,n_mol_basis,n_mol_basis)) #two electron term 

        for i in range(n_mol_basis):
            for j in range(n_mol_basis):
                for k in range(n_mol_basis):
                    for l in range(n_mol_basis):
                
                        gto1=self.mol[i] #gaussian expansion of each AO
                        gto2=self.mol[j]
                        gto3=self.mol[k]
                        gto4=self.mol[l]
                        #assuming pure orbitals
                        #print(HRR(gto1,gto2,gto3,gto4))
                        la,lb,lc,ld= gto1[0].l,  gto2[0].l,  gto3[0].l,  gto4[0].l
                        V_ee[i,j,k,l]=HRR(gto1,gto2,gto3,gto4, la,lb,lc,ld) 
        self.V_ee=V_ee
        
    def nuclear_nuclear_repulsion(self):
        E_nn=0
        for atom1 in range(len(self.position)):
            for atom2 in range(atom1+1,len(self.position)):
                E_nn+=self.charges[atom1]*self.charges[atom2]/ (np.sqrt(np.dot(self.position[atom1]-self.position[atom2],self.position[atom1]-self.position[atom2])))
        self.E_nn=E_nn
