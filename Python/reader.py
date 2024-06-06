# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:26:40 2024

@author: domin
"""
import numpy as np
class reader():
    def __init__(self,path):
        self.Z=[]
        self.centers = []
        self.alphas = []
        self.coefficients = []
        self.l = []
        self.orbital_alpha=[]
        self.orbital_c=[]
        self.orbital_l=[]
        
        self.mode=None
        self.read(path)
        
    def read(self,path):
        f=open(path,"r")
        for line in f:
            line=line.strip()
            if line=="atoms": self.mode="atomic";continue
            elif line == "centers":self.mode="central";continue
            elif line == "alpha":    self.mode="exponents";continue
            elif line == "coefficients":self.mode="cs";continue
            elif line == "momenta":self.mode="ang";continue
            elif line=="*":break
     
            if self.mode=="atomic":
                if line.strip():
                    self.Z.append(int(line.split()[0]))
            elif self.mode=="central":
                if line.strip():
                    self.centers.append(np.array(list(map(float,line.split(","))))*1.8897259886)
            elif self.mode=="exponents":
                if not line.strip():
                    self.alphas.append(self.orbital_alpha)
                    self.orbital_alpha=[]
                else:
                    self.orbital_alpha.append(list(map(float,line.split(","))))
            elif self.mode=="cs":
                if not line.strip():
                    self.coefficients.append(self.orbital_c)
                    self.orbital_c=[]
                else:
                    self.orbital_c.append(list(map(float,line.split(","))))
            elif self.mode=="ang":
                if not line.strip():
                    self.l.append(self.orbital_l)
                    self.orbital_l=[]
                else:
                    self.orbital_l.append(list(map(int,line.split(","))))
                    
    def output(self):
        return (self.Z,self.centers,self.alphas,self.l,self.coefficients)