
import numpy as np
from grid2 import GRID


class molecule():
    def __init__(self,charge,position):
        self.charges=np.array(charge)
        self.position=position

choice=input("He,H2 or H2O :")
if choice=="He":
    #He
    charge=[2]
    centers=[ np.array([0, 0,0])]
elif choice=="H2":
    #H2
    charge=[1,1]
    centers=[ np.array([0, 0,0]), 
              np.array([0,0,1.4])] 
elif choice=="H2O":
    #H2O
    charge=[1,1,8]
    centers=[ np.array([0, 1.4305227, 1.1092692]), 
              np.array([0, -1.4305227, 1.1092692]),
              np.array([0, 0, 0])] 
else: raise ValueError

mole=molecule(charge,centers)

grid,w=GRID(mole)

np.save("grid_x",grid[:,0])
np.save("grid_y",grid[:,1])
np.save("grid_z",grid[:,2])
np.save("weights",w)
