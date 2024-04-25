# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:35:53 2024

@author: domin
"""
import numpy as np
import scipy.linalg as spl
import scipy.sparse as sp



class finite_differences():
    
    def __init__(self,grid,charge):
        grid.laplacian()
        self.laplacian=grid.Lap
        self.nelect=np.sum(charge)
    
    def nuclear_electron(self,grid, centers, charge):
        natoms=len(centers)
        npoints=np.prod(grid.n_points)
        
        xn,yn,zn=grid.ravel_grid()
        
        v_ne=0
        for i in range(natoms):
            xi,yi,zi=centers[i]
            z=charge[i]
    
            v_ne+=-z/np.sqrt((xn-xi)**2+(yn-yi)**2+(zn-zi)**2)
    
        v_ne=sp.spdiags(v_ne,0,npoints,npoints)
        self.vne=v_ne
    
    def v_neE(self,grid, density):
        vne=self.vne
        
        dV=grid.dV
        
        e_ne=np.sum(density*vne)*dV
        
        self.ene=e_ne
    
    def hartree(self,grid, density): #put in grid.laplacian()
        npoints=np.prod(grid.n_points)
        
        vh=sp.linalg.cgs(self.laplacian, -4*np.pi*density)[0]
        vh=sp.spdiags(vh,0, npoints, npoints)
        
        dV=grid.dV
        
        eh=0.5*np.sum(density*vh)*dV
        
        self.vh=vh
        self.eh=eh
    
    def kinetic(self):
        T=-0.5*self.laplacian
        self.T=T
    
    def kineticE(self,grid):
        
        dV=grid.dV
        
        kin=[]
        for eig in self.psi.T[:self.nelect//2]:
            inner=eig*self.T.dot(eig)
            kin.append(inner*dV)
        
        self.eT= np.sum(kin)
    
    def exchange(self,grid, density):
        dV=grid.dV
        npoints=np.prod(grid.n_points)
        
        import exco as exc
        vx=exc.Vexchange_LDA(density)
        vc=exc.Vcorrelation_MP(exc.wigner_seitz_r(density))
        
        vx=sp.spdiags(vx, 0, npoints, npoints)
        vc=sp.spdiags(vc, 0, npoints, npoints)
        
        ex=exc.Eexchange_LDA(density) * dV
        ec=exc.Ecorrelation_MP(exc.wigner_seitz_r(density)) * dV
    
        self.vxc= vx+vc
        self.exc= ex+ec
    
    def wavefunction(self,grid,psiM): #normalization
        dV=grid.dV
        self.psi=np.zeros_like(psiM)
        for i in range(psiM.shape[1]):
            self.psi[:,i]=(psiM[:,i])/np.sqrt(dV) #this normalizes to 125000, without psi**2 it goes to 1
    
    def get_density(self):
        rho=np.zeros_like(self.psi[:,0])
        for i in range (self.nelect//2):
            rho+= 2*self.psi[:,i]**2
        return rho

    def SCF(self,scf_param,grid,centers,charge):
        counter = 0
        E0 = 0
        ediff ,etol= scf_param 
        psi = None
        
        self.kinetic()
        self.nuclear_electron(grid,centers,charge)
        H=self.T+self.vne
        
        dV=grid.dV
        #print(ediff, etol)
        while ediff>etol:
            
            #E, psiM=sp.linalg.eigsh(H, k=int(np.ceil(self.nelect*1.0/2)), which='SA')
            E, psiM = sp.linalg.eigsh(H, k=self.nelect//2, which='SA')
            
            counter+=1
            
            self.wavefunction(grid, psiM)
            rho=self.get_density()
    
            self.hartree(grid, rho)
            self.exchange(grid, rho)
            
            self.kineticE(grid)
            self.v_neE(grid, rho)
            self.exc=np.sum(self.exc*dV)
    
            E=self.eT+self.ene+self.exc+self.eh
            #print(E)
            ediff=abs(E0-E)
            E0=E
            
            H=self.T+self.vne+self.vh+self.vxc
            if counter==50: 
                print("Did not converge")
                return None
            
        print("Converged")
        self.E=E
        
"""
#How to run it, 20 is a slice taken, can be changed. Energy is bad but potentials have correct shapes

import matplotlib.pyplot as plt
import exco as exc

from cubic_grid import cubic_grid
grid=cubic_grid(pos, [50,50,50], 3)
grid.laplacian()

pos=np.array([[0,0,0], [0,0,1.434]])
charge=np.array([1,1])
H2=finite_differences(grid,charge)
H2.SCF([10**6,1e-9],grid,pos,charge)
import exco as exc
vc=exc.Vcorrelation_MP(exc.wigner_seitz_r(rho))
vx=exc.Vexchange_LDA(rho)
x,y,z=grid.raveled_grid
plt.plot(z[20,20,:],(rho.reshape((50,50,50)))[20,20,:])
plt.plot(z[20,20,:],(vx.reshape((50,50,50)))[20,20,:])
plt.plot(z[20,20,:],(vc.reshape((50,50,50)))[20,20,:])
vh=H2.vh.diagonal()
plt.plot(z[20,20,:],(vh.reshape((50,50,50)))[20,20,:])
vn=H2.vne.diagonal()
plt.plot(z[20,20,:],(vn.reshape((50,50,50)))[20,20,:])
"""
