# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:30:21 2024

@author: domin
"""

import numpy as np
from scipy import special
import scipy.linalg as sp


def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-2)
    
class gaussian():
    """
    class that contains the properties of a gaussian
    """
    def __init__(self, alpha, coords, x, y, z, coeff=1):
        """

        Parameters
        ----------
        alpha : float
            exponent of gaussian
        coords : array
            center of gaussian
        x : int
            x angular momentum
        y : int
            y angular momentum
        z : TYPE
            z angular momentum
        coeff : float, optional
            coefficient of gaussian.The default is 1.

        Returns
        -------
        initilazes gaussian object

        """
        self.a=alpha
        self.r=np.array(coords)
        self.c=coeff
        self.l=np.array([x,y,z])
        N=(2*self.a/np.pi)**(3/4)
        up=(4*self.a)**((x+y+z)/2)
        down=(factorial(2*x-1)*factorial(2*y-1)*factorial(2*z-1))**(1/2)
        self.N=N*up/down
        
class molecule():
    """
    class that builds the molecular strucutre
    """
    def __init__(self, alphas,centers,l, Z,coefficients):
        """
        

        Parameters
        ----------
        alphas : list
            a list of exponents in the form [ atom1=[contractedGTO=[gaussian1,gaussian2,gaussian3]]]
            example for H2 [ [ [0.802,0.146,0.407,0.1350]], [[0.802,0.146,0.407,0.1350] ] ] 
            can be run for atoms by providing uncontracted gaussians as contracted ones: 
            eg. for He alphas = [ [[38.474970],[5.782948],[1.242567],[0.298073]] ]
        centers : list
            list of np.arrays with centers of each atom
        l : list
            similar to a but with [x,y,z] for each contractedGTO
            example for H2 l=[ [[0,0,0]],[[0,0,0]] ]
        Z : list
            charges of each atom
        coefficients : list
            exactly like a but with coefficients
        
        detailed input can be found in test2
        """
        self.position=centers 
        self.charges=Z
        self.nocc()
        self.build_molecule(alphas,l,coefficients)
        
    def build_molecule(self,alphas,l, coefficients): #also try with solving for all Cs, (just use all gaussians as basis intead of fixed orbitals)
        self.mol=[]
        for atom in range(len(alphas)):
            #print("atoms",atom)
            for orbital in range(len(alphas[atom])):
                #print("orbital",alphas[atom][orbital])
                orbital_expansion=alphas[atom][orbital]
                temp_basis=[]
                for i in range(len(orbital_expansion)):
                    #(i)
                    temp_basis.append(gaussian(orbital_expansion[i], self.position[atom],l[atom][orbital][0],l[atom][orbital][1],l[atom][orbital][2],coeff=coefficients[atom][orbital][i]))
                self.mol.append(temp_basis)

    def nocc(self): #bad for now!!
        n_occ=np.array(self.charges).sum()//2
        self.n_occ=n_occ #more complex for molecule case
