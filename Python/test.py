# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:23:44 2024

@author: domin
"""
import numpy as np
from molecular_integrals_s import molecular_integrals_s
from HF import HF
from HF import MP2
from DFT import DFT
#Currently not working cause diis was added to DFT

alphas= [ [[0.8021420155E+01,0.1467821061E+01,0.4077767635E+00,0.1353374420E+00]],
          [[0.8021420155E+01,0.1467821061E+01,0.4077767635E+00,0.1353374420E+00]]  ]

coefficients=[ [[0.5675242080E-01,0.2601413550E+00,0.5328461143E+00,0.2916254405E+00]],
               [[0.5675242080E-01,0.2601413550E+00,0.5328461143E+00,0.2916254405E+00]]  ]

centers=[ np.array([0 ,0, 0]), 
          np.array([0, 0, 1.4]) ] 

Z=np.array([1,1])

scf_param=[1e-10,2000]

molecule=molecular_integrals_s(alphas,centers,Z, coefficients)

molHF=HF(scf_param,molecule)
print("HF energy:", molHF.E_total)

molMP2=MP2(scf_param,molecule)
print("HF-MP2 energy:", molMP2.E_total_MP2)

molDFTMP=DFT(molecule,scf_param, functional="MP")
print("DFT-MP energy:", molDFTMP.E_total)

molDFTWVN=DFT(molecule,scf_param, functional="WVN")
print("DFT-WVN energy:", molDFTWVN.E_total)

molDFTGL=DFT(molecule,scf_param, functional="GL")
print("DFT-GL energy:", molDFTGL.E_total)

molDFTPZ=DFT(molecule,scf_param, functional="PZ")
print("DFT-PZ energy:", molDFTPZ.E_total)
