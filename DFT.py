# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:52:40 2024

@author: domin
"""
import numpy as np
from scipy import special
import scipy.linalg as sp
import matplotlib.pyplot as plt

from diis import DIIS
from molecular_integrals_s import molecule



class DFT():#molecule is actually integrals  
    """
    This is a class implementing KS-DFT code, it builds on molecular_integrals objects
    
    Density evaluation was written with the help of tutorials found in psi4numpy repository
    Psi4NumPy: An Interactive Quantum Chemistry Programming Environment for Reference Implementations and Rapid Development Daniel G. A. Smith, Lori A. Burns, Dominic A. Sirianni, Daniel R. Nascimento, Ashutosh Kumar, Andrew M. James, Jeffrey B. Schriber, Tianyuan Zhang, Boyi Zhang, Adam S. Abbott, Eric J. Berquist, Marvin H. Lechner, Leonardo A. Cunha, Alexander G. Heide, Jonathan M. Waldrop, Tyler Y. Takeshita, Asem Alenaizan, Daniel Neuhauser, Rollin A. King, Andrew C. Simmonett, Justin M. Turney, Henry F. Schaefer, Francesco A. Evangelista, A. Eugene DePrince III, T. Daniel Crawford, Konrad Patkowski, and C. David Sherrill Journal of Chemical Theory and Computation, 2018, 14 (7), 3504-3511 DOI: 10.1021/acs.jctc.8b00286
    """   
    def __init__(self,molecule,scf_param,functional="MP",limit=0):
        self.n_occ=molecule.n_occ
        self.get_grid(molecule)
        self.grid_atomic_orbitals(molecule.mol,self.grid)
        self.total_energy(scf_param,molecule,functional,limit)
    
    def get_grid(self,molecule):
        from grid import GRID 
        #there is also a grid2 script in which you can set coarse or close grid
        self.grid,self.weights=GRID(molecule)
    
    def grid_atomic_orbitals(self,basis,grid_point):
        ao=[]
        for gto in basis:
            psi=np.zeros(len(grid_point))
            for g in gto:
                psi+=gaussian_grid_eval(g,grid_point)
            ao.append(psi)
            
        self.ao=np.column_stack(ao)
        
        
    def evaluate_density(self):
        
        ao_density=np.einsum("pr,rq->pq", self.ao, self.D)
        ao_density=np.einsum("pi,pi->p", self.ao, ao_density)
        
        ao_density[abs(ao_density) < 1.0e-15] = 0
        
        self.rho=ao_density
        return ao_density
        
    def matrix_xc_potential(self,vxc):
        weighted_ao=np.einsum("pi,p->pi", self.ao, 0.5*self.weights*vxc)
        xc=np.einsum("rp,rq->pq", self.ao, weighted_ao)
        
        self.V_xc=xc+xc.T #to ensure symetry
        return xc+xc.T
    
    def xc_energy(self,exc):
        E_xc=np.einsum('p,p->', self.rho*self.weights, exc)
        self.E_xc=E_xc
        return E_xc
    
    def density_matrix(self):
        self.D=2*np.einsum("pi,qi->pq", self.C[:,:self.n_occ], self.C[:,:self.n_occ])
        
    def SCF(self,scf_param,molecule,functional,limit):
        import exco as ex
        convergence,diss_c, max_steps=scf_param
        ds=DIIS()
        
        e,self.C=sp.eigh(molecule.H_core,molecule.S)
        self.density_matrix()
        
        E0=0
        for step in range(max_steps):
            J=np.einsum("rs,pqrs->pq", self.D, molecule.V_ee)
            rho=self.evaluate_density()
            #exc,vxc=ex.Eexchange_LDA(rho)+ ex.Ecorrelation_LDA(ex.wigner_seitz_r(rho)), ex.Vexchange_LDA(rho)+ ex.Vcorrelation_LDA(ex.wigner_seitz_r(rho))
            exc,vxc=ex.Eexchange_LDA(rho)+ ex.e_correlation(ex.wigner_seitz_r(rho),functional,limit), ex.Vexchange_LDA(rho)+ ex.V_correlation(ex.wigner_seitz_r(rho),functional,limit)
           
            #V_xc=self.matrix_xc_potential(vxc) 
            #print(V_xc)
            fKS=molecule.H_core+J+self.matrix_xc_potential(vxc) 
            ds.update(fKS,self.D,molecule.S)
            
            if step>1:
                fKS=ds.DISS_F()
            
            e,self.C=sp.eigh(fKS,molecule.S)
            self.density_matrix()
            
            eKS=np.einsum('pq,pq->', self.D, (molecule.H_core + 1/2*J)) + self.xc_energy(exc)
            if (abs(eKS - E0) < convergence) and (ds.RMSD_check()<diss_c):
                #print("Converged to:", eKS)
                print("Converged")
                self.E_ee=eKS
                return
            E0=eKS
        print("Did not converge")
        
    def total_energy(self,scf_param,molecule,functional,limit):
        self.SCF(scf_param,molecule,functional,limit)
        self.E_total=self.E_ee+molecule.E_nn
        #print("Total energy is:", self.E_total)
    
def gaussian_grid_eval(gaussian,grid_point): 
    A=grid_point-gaussian.r
    psi=np.prod(A**gaussian.l, axis=1) * gaussian.N * gaussian.c * np.exp(-gaussian.a*np.sum(A*A, axis=1))
    return psi    
