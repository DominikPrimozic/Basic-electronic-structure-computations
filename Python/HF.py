# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:42:11 2024

@author: domin
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp
import time

def compute_electronic_energy(D,H,F):
    return 1/2 * np.einsum("pq,pq->", D, H+F) 

class HF(): #molecule is actually integrals
    """
    general reference for the whole procedure
    Szabo, A.; Ostlund, N. S. Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory Dover Publication Inc., 1996.
    """
    
    def __init__(self,scf_parameters,molecule):
        #self.SCF(scf_parameters,molecule)
        self.total_energy(scf_parameters, molecule)
    
    def fock_matrix(self,molecule):
        self.F=molecule.H_core + np.einsum("rv,pqrv->pq", self.D,molecule.V_ee) -1/2 * np.einsum("rv,prqv->pq", self.D,molecule.V_ee) #there is a factor 2 in density, thats why 1/2
        
    def orbital_coefficients(self,molecule):
        self.e, self.C=sp.eigh(self.F,molecule.S)
        
    def density_matrix(self,molecule):
        self.D=2*np.einsum("pi,qi->pq", self.C[:,:molecule.n_occ], self.C[:,:molecule.n_occ])
    
        
        
    def SCF(self, scf_parameters,molecule):
        convergence, max_steps=scf_parameters
        #initial guess
        self.e,self.C = sp.eigh(molecule.H_core,molecule.S)
        self.density_matrix(molecule)
        E0=compute_electronic_energy(self.D,molecule.H_core,molecule.H_core)
        for step in range(max_steps):
            self.fock_matrix(molecule)
            self.orbital_coefficients(molecule)
            self.density_matrix(molecule)
            E=compute_electronic_energy(self.D,molecule.H_core,self.F)
            if abs(E-E0)<convergence:
                #print("converged to: ", E, "hartrees")
                print("converged")
                self.E_ee=E
                return 
            E0=E
        print("Did not converge")    
        
    def total_energy(self,scf_parameters,molecule):
        self.SCF(scf_parameters,molecule)
        self.E_total=self.E_ee + molecule.E_nn
        #print("total energy: ", self.E_total, "hartrees")
    
    
    def update_coefficents(self,state,molecule): #add to other ones
        assert isinstance(state, int)
        #assert isinstance(object, molecule)
        C_vector=[]
        for i in range(len(molecule.mol)):
            C_vector.append(self.C[i,state])
        return C_vector
    
    def plot_state(self, state,molecule, x=[-2,2,50], y=[-2,2,50], z=[-2,2.25,50],mode="XY",slider=0):
         import matplotlib.pyplot as plt
         C_state=self.update_coefficents(state,molecule)
         #print(C_state)
         X = np.linspace(x[0], x[1], x[2])
         Y = np.linspace(y[0], y[1], y[2])
         Z = np.linspace(z[0], z[1], z[2])
         if mode=="XY":
             X, Y = np.meshgrid(X, Y)
             Z=slider
         elif mode=="XZ":
             X, Z = np.meshgrid(X, Z)
             Y=slider
         else:
             Y, Z = np.meshgrid(Y, Z)
             X=slider
             
         fun=[]
         for i in range(len(molecule.mol)):
             fi=0
             for function in molecule.mol[i]:
                 fi+=function.c * function.N * (X-function.r[0])**function.l[0] * (Y-function.r[1])**function.l[1] * (Z-function.r[2])**function.l[2] * \
                     np.exp(-function.a *( (X-function.r[0])**2 + (Y-function.r[1])**2 + (Z-function.r[2])**2))
             fun.append(fi)
         assert len(C_state)==len(fun)
         f=0
         for state in range(len(fun)):
             f+=C_state[state]*fun[state]
         if mode=="XY":
             plt.contourf(X, Y, f,30, cmap='RdGy')
             plt.plot(np.array(molecule.position)[:,0],np.array(molecule.position)[:,1], "o")
         elif mode=="XZ":
             plt.contourf(X, Z, f,30, cmap='RdGy')
             plt.plot(np.array(molecule.position)[:,0],np.array(molecule.position)[:,2], "o")
         else:
             plt.contourf(Y, Z, f,30, cmap='RdGy')
             plt.plot(np.array(molecule.position)[:,1],np.array(molecule.position)[:,2], "o")
    
    def plot3D(self,molecule,state,x=[-5,5,50], y=[-5,5,50], z=[-5,5,50],cuttoff=0.1,imin=0.02,imax=0.3,opac=0.3):
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.renderers.default='browser'
        X = np.linspace(x[0], x[1], x[2])
        Y = np.linspace(y[0], y[1], y[2])
        Z = np.linspace(z[0], z[1], z[2])
        X, Y,Z = np.meshgrid(X, Y,Z)
        C=self.update_coefficents(state,molecule)
        fun=[]
        for i in range(len(molecule.mol)):
            fi=0
            for function in molecule.mol[i]:
                fi+=function.c * function.N * (X-function.r[0])**function.l[0] * (Y-function.r[1])**function.l[1] * (Z-function.r[2])**function.l[2] * \
                    np.exp(-function.a *( (X-function.r[0])**2 + (Y-function.r[1])**2 + (Z-function.r[2])**2))
            fun.append(fi)
        f=0
        for state in range(len(fun)):
            f+=C[state]*fun[state]
        g=np.where(abs(f)>cuttoff,f,0)
        pos=np.array(molecule.position)
        fig= go.Figure(data=[go.Isosurface(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),value=g.flatten(),isomin=imin,isomax=imax,opacity=opac,colorscale='RdBU'),go.Isosurface(x=X.flatten(),y=Y.flatten(),z=Z.flatten(),value=-1*g.flatten(),isomin=imin,isomax=imax,opacity=opac,colorscale='ice')])
        fig.add_scatter3d(x=pos[:,0],y=pos[:,1],z=pos[:,2], mode="markers")
        fig.show()        
        
        
class MP2(HF):
    
    def __init__(self,scf_parameters,molecule):
        HF.__init__(self,scf_parameters,molecule)
        self.n_occ=molecule.n_occ
        self.MP2_energy(molecule)
        self.energy()
        
        
    def transform_eri(self,molecule):
        self.V_ee_mo= np.einsum("up,vq,uvkl,kr,ls->pqrs",self.C,self.C,molecule.V_ee,self.C,self.C,optimize=True)
        #return self.V_ee_mo
    
    def MP2_energy(self,molecule):
        self.transform_eri(molecule)
        iajb=self.V_ee_mo[:self.n_occ,self.n_occ:,:self.n_occ,self.n_occ:]
        up=iajb * (2*iajb -iajb.swapaxes(1, 3) )
        #ei-ea+ej-eb
        down= self.e[:self.n_occ, None, None, None] - self.e[None, self.n_occ:, None, None] +self.e[None, None, :self.n_occ, None] - self.e[None, None, None, self.n_occ:]
        
        #np.einsum("iajb,iajb,iajb->", iajb,2*iajb-iajb.swapaxes(1, 3),1/down, optimize=True)
        self.E_MP2=np.sum(up/down)
    
    def energy(self):
        self.E_total_MP2=self.E_total+self.E_MP2
        #print(self.E_total_MP2)
