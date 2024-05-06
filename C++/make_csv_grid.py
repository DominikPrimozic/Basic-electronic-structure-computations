# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:43:20 2024

@author: domin
"""
import numpy as np
from grid2 import GRID


class molecule():
    def __init__(self,charge,position):
        self.charges=np.array(charge)
        self.position=position
        
charge=[1,1,8]
centers=[ np.array([0, 1.4305227, 1.1092692]), 
          np.array([0, -1.4305227, 1.1092692]),
          np.array([0, 0, 0])] 

mole=molecule(charge,centers)

grid,w=GRID(mole)

np.save("grid_x",grid[:,0])
np.save("grid_y",grid[:,1])
np.save("grid_z",grid[:,2])
np.save("weights",w)
