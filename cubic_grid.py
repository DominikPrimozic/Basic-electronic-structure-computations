# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:32:12 2024

@author: domin
"""

import numpy as np
import scipy.sparse as sp


class cubic_grid(): #dense is how many points if spacing mode is false
    #make it 5 bohr more than min and max
    def __init__(self,centers,dense,offset, spacing_mode=False,equal=True):
        self.get_corners(centers,offset,equal)
        self.make_grid(dense, spacing_mode)
        
    def get_corners(self,centers, offset, equal):
        mx,my,mz=np.max(centers,axis=0)
        nx,ny,nz=np.min(centers,axis=0)
        #equal cube
        if equal==True:
            self.corners=[[min(nx,ny,nz)-offset,max(mx,my,mz)+offset],[min(nx,ny,nz)-offset,max(mx,my,mz)+offset],[min(nx,ny,nz)-offset,max(mx,my,mz)+offset]]
        else:
            self.corners=[[nx-offset,mx+offset],[ny-offset,my+offset],[nz-offset,mz+offset]]

    def make_grid(self,dense,spacing_mode): #dense=[x,y,z]
        if spacing_mode==False:
            px,py,pz=np.linspace(self.corners[0][0],self.corners[0][1],dense[0]),np.linspace(self.corners[1][0],self.corners[1][1],dense[1]),np.linspace(self.corners[2][0],self.corners[2][1],dense[2])
            dV=(px[1]-px[0])*(py[1]-py[0])*(pz[1]-pz[0])
            dr=(px[1]-px[0],py[1]-py[0],pz[1]-pz[0])
            nx,ny,nz=dense[0],dense[1],dense[2]
        else:
            px,py,pz=np.arrange(self.corners[0][0],self.corners[0][1],dense[0]),np.arrange(self.corners[1][0],self.corners[1][1],dense[1]),np.arrange(self.corners[2][0],self.corners[2][1],dense[2])
            dV=dense[0]*dense[1]*dense[2]
            dr=dense[0],dense[1],dense[2]
            nx,ny,nz=len(px),len(py),len(pz)
        
        self.dV=dV
        self.dr=dr
        self.grid=np.meshgrid(px,py,pz)
        self.n_points=nx,ny,nz
        
    def ravel_grid(self):
        x,y,z=self.grid
        self.raveled_grid=x,y,z
        return np.ravel(x) ,  np.ravel(y),  np.ravel(z)
    
    def laplacian(self):
        nx,ny,nz=self.n_points
        dx,dy,dz=self.dr
        
        ex=np.ones(nx) / dx**2
        ey=np.ones(ny) / dy**2
        ez=np.ones(nz) / dz**2
        
        Lx=sp.spdiags(np.array([ex, -2*ex, ex]), np.array([-1,0,1]), nx, nx)
        Ly=sp.spdiags(np.array([ey, -2*ey, ey]), np.array([-1,0,1]), ny, ny)
        Lz=sp.spdiags(np.array([ez, -2*ez, ez]), np.array([-1,0,1]), nz, nz)
        
        Laplacian=sp.kronsum(Lz, sp.kronsum(Ly,Lx))
        
        self.Lap=Laplacian