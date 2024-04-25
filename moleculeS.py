# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:32:41 2024

@author: domin
"""

import numpy as np

class gaussian_s():
    
    def __init__(self, alpha, coords, coeff=1, normalized=True):
        self.a=alpha
        self.r=np.array(coords)
        self.c=coeff
        self.l=np.array([0,0,0])
        if normalized==True:
            self.N= ((2.0*alpha)/np.pi)**(3/4) #normalization of a gaussian
        else: 
            self.N=1
            

class molecule():
    
    def __init__(self, alphas,centers, Z,coefficients, norm=True ):
        self.position=centers 
        self.charges=Z
        self.nocc()
        self.build_molecule(alphas,coefficients, norm)
        
    def build_molecule(self,alphas, coefficients, norm): 
        self.mol=[]
        for atom in range(len(alphas)):
            for orbital in range(len(alphas[atom])):
                orbital_expansion=alphas[atom][orbital]
                temp_basis=[]
                for i in range(len(orbital_expansion)):
                    temp_basis.append(gaussian_s(orbital_expansion[i], self.position[atom],coeff=coefficients[atom][orbital][i], normalized=norm))
                self.mol.append(temp_basis)

    def nocc(self): 
        n_occ=np.array(self.charges).sum()//2
        self.n_occ=n_occ 