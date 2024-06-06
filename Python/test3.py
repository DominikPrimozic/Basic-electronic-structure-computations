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
import reader as rd
import time


path=input("File input: ")
limit=0
gty=2
#gty=int(input("Small or dense grid? Type 1 for small, 2 for dense: "))
mode=input("HF, HFMP or DFT: ")
if mode=="DFT":
    gty=int(input("Sparse(1) or dense(2): "))
    funct=input("Choose functional. Available are MP, VWN, GL, PZ. Type one of those for choice: ")
    if funct=="WVN":
        choose=int(input("Unpolarized or polarized? Type 0 or 1 "))
        if choose==1:
            limit=1
        else:
            limit=0
            
if gty!=1 or gty!=2: gty=2

ready=rd.reader((path))

Z,centers,alphas,l,coefficients=ready.output()    
scf_param=[1e-10,2000]
scf_param_DFT=[1e-6,1e-4,2000]

start = time.time()

molecule=molecular_integrals(alphas,centers,l,Z, coefficients) 

end = time.time()
length = end - start
print("molecular integrals took: ", length, "seconds")

if mode=="HF":
    start2 = time.time()
    
    molHF=HF(scf_param,molecule)
    print("HF energy:", molHF.E_total)
    
    end2 = time.time()
    length2 = end2 - start2
    print("HF took: ", length2*1e3, "milliseconds")

elif mode=="HFMP":
    molMP2=MP2(scf_param,molecule)
    print("HF-MP2 energy:", molMP2.E_total_MP2)

elif mode=="DFT":
    start = time.time()
    
    
    molDFT=DFT(molecule,scf_param_DFT,gty, funct, limit)
    print("DFT-MP energy:", molDFT.E_total)
    
    end = time.time()
    length = end - start
    print("DFT took: ", length*1e3, "milliseconds")
    
else:
    raise TypeError("non recognized mode")
    
        
    
