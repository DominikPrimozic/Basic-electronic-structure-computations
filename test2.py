# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:47:21 2024

@author: domin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:23:44 2024

@author: domin
"""
import numpy as np
from molecular_integrals import molecular_integrals
from HF import HF
from HF import MP2
from DFT import DFT

choice=input("Choose He, H2, H2O by writing it: ")

if choice=="He":
    #He
    alphas = [ 
                [
                 [38.474970],
                 [5.782948],
                 [1.242567],
                 [0.298073]
                ] 
             ]
    
    coefficients=[ 
                  [
                      [1],
                      [1],
                      [1],
                      [1]
                  ] 
                 ]
    centers=[np.array([0,0,0])]
    Z=np.array([2])
    l=[ 
       [ 
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [0,0,0]
       ] 
      ]
    
    scf_param=[1e-10,2000]
    scf_param_DFT=[1e-6,1e-4,2000]
    #"""




if choice=="H2":
    #H2
    alphas= [ [[0.8021420155E+01,0.1467821061E+01,0.4077767635E+00,0.1353374420E+00]],
              [[0.8021420155E+01,0.1467821061E+01,0.4077767635E+00,0.1353374420E+00]]  ]
    
    coefficients=[ [[0.5675242080E-01,0.2601413550E+00,0.5328461143E+00,0.2916254405E+00]],
                   [[0.5675242080E-01,0.2601413550E+00,0.5328461143E+00,0.2916254405E+00]]  ]
    
    centers=[ np.array([0 ,0, 0]), 
              np.array([0, 0, 1.4]) ] 
    
    Z=np.array([1,1])
    
    l=[ [[0,0,0]],
        [[0,0,0]]  ]
    
    scf_param=[1e-10,2000]
    scf_param_DFT=[1e-6,1e-4,2000]

if choice=="H2O":
    #H2O
    alphas=[ [ [0.3425250914E+01,0.6239137298E+00,0.1688554040E+00] ],
            
             [ [0.3425250914E+01,0.6239137298E+00,0.1688554040E+00] ], 
             
             [ [0.1307093214E+03,0.2380886605E+02,0.6443608313E+01],
               [0.5033151319E+01,0.1169596125E+01,0.3803889600E+00], 
               [0.5033151319E+01,0.1169596125E+01,0.3803889600E+00], 
               [0.5033151319E+01,0.1169596125E+01,0.3803889600E+00],
               [0.5033151319E+01,0.1169596125E+01,0.3803889600E+00] 
             ] 
            ]
    
    coefficients=[ [ [0.1543289673E+00,0.5353281423E+00,0.4446345422E+00] ],
                  
                   [ [0.1543289673E+00,0.5353281423E+000,0.4446345422E+00] ], 
                   
                   [
                     [0.1543289673E+00,0.5353281423E+00,0.4446345422E+00], 
                     [-0.9996722919E-01,0.3995128261E+00,0.7001154689E+00],
                     [0.1559162750E+00,0.6076837186E+00,0.3919573931E+00],
                     [0.1559162750E+00,0.6076837186E+00,0.3919573931E+00],
                     [0.1559162750E+00,0.6076837186E+00,0.3919573931E+00] 
                   ] 
                 ]
    
    centers=[
             np.array([0, 1.4305227, 1.1092692]), 
             np.array([0, -1.4305227, 1.1092692]),
             np.array([0, 0, 0])
             ]
    
    Z=np.array([1,1,8])
    
    l=[ 
       [[0,0,0]],
       
       [[0,0,0]],
       
       [ 
        [0,0,0],
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1] 
       ] 
      ]
    
    
    scf_param=[1e-10,2000]
    scf_param_DFT=[1e-6,1e-4,2000]


molecule=molecular_integrals(alphas,centers,l,Z, coefficients) 

molHF=HF(scf_param,molecule)
print("HF energy:", molHF.E_total)

molMP2=MP2(scf_param,molecule)
print("HF-MP2 energy:", molMP2.E_total_MP2)


#"""
molDFTMP=DFT(molecule,scf_param_DFT, functional="MP")
print("DFT-MP energy:", molDFTMP.E_total)

molDFTWVN0=DFT(molecule,scf_param_DFT, functional="WVN",limit=0)
print("DFT-WVN unpolarized energy:", molDFTWVN0.E_total)

molDFTWVN1=DFT(molecule,scf_param_DFT, functional="WVN",limit=1)
print("DFT-WVN polarized energy:", molDFTWVN1.E_total)

molDFTGL=DFT(molecule,scf_param_DFT, functional="GL")
print("DFT-GL energy:", molDFTGL.E_total)

molDFTPZ=DFT(molecule,scf_param_DFT, functional="PZ")
print("DFT-PZ energy:", molDFTPZ.E_total)
#"""