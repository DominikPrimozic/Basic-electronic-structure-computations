# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:50:27 2024

@author: domin
"""
import numpy as np

def wigner_seitz_r(n):
    """
	r_s=(3/4 *1/(pi*n) )^(1/3)
	"""
    n[n==0]=1e-120
    
    return np.power(3/(4 *np.pi*n),1/3)

def Vexchange_LDA(n):
	return -np.power(3*n/np.pi,1/3) #4/3 * ex

def Eexchange_LDA(n):
	e_x=-3/4 * np.power(3*n/np.pi,1/3) 
	return e_x

def Vcorrelation_MP(rs):
    e_c=Ecorrelation_MP(rs)
    #de_c=derivative(Ecorrelation_LDA,rs) 
    a=(np.log(2)-1) / (2*np.pi**2)
    b=20.4562557
    de_c=-a*b*(rs+2)/(rs*(rs**2+b*rs+b))
    return e_c -1/3 * rs *de_c 

def Ecorrelation_MP(rs):
    
    a=(np.log(2)-1) / (2*np.pi**2)
    b=20.4562557
    e_c=a*np.log(1 + b/rs + b/rs**2)
    return e_c

def Vcorrelation_WVN(rs, limit=0):
    #from the original paper
    #paramagentic limit 
    if limit==0:
        a=0.0621814 #0.0310907 
        b=13.0720 #7.06042
        c=42.7198 #18.0578
        x0=-0.409286 #-0.32500
        b1=(b*x0-c)/(c*x0) #3.46791
        b2=(x0-b)/(c*x0) #1.25842
        b3=-1/(c*x0) #0.170393

    #ferromagentic limit
    elif limit==1:
        a=0.0310907 
        x0=-0.743294
        b=20.1231
        c=101.578
        b1=(b*x0-c)/(c*x0)
        b2=(x0-b)/(c*x0) 
        b3=-1/(c*x0)
        
    else: raise ValueError("limit must be 0 or 1") 
    
    Xr=rs + b*np.sqrt(rs) + c
    Xx=x0**2 + b*np.sqrt(x0**2) + c
    Q=np.sqrt(4*c-b**2)
    
    t2=Q/(2*np.sqrt(rs)+b)
    t1=np.log( (np.sqrt(rs)-x0)**2/Xr) + 2*(b+2*x0)/Q * np.arctan(t2)
    t=np.log(rs/Xr) + 2*b/Q * np.arctan(t2) - b*x0*t1/Xx
    
    return Ecorrelation_WVN(rs,limit) - a/3 * (1+b1*np.sqrt(rs))/(1+b1*np.sqrt(rs) + b2*rs + b3*np.power(rs,3/2))

def Ecorrelation_WVN(rs,limit=0):
    #paramagentic limit 
    if limit==0:
        a=0.0621814 #0.0310907 
        b=13.0720 #7.06042
        c=42.7198 #18.0578
        x0=-0.409286 #-0.32500
        b1=(b*x0-c)/(c*x0) #3.46791
        b2=(x0-b)/(c*x0) #1.25842
        b3=-1/(c*x0) #0.170393
    #ferromagentic limit
    elif limit==1:
        a=0.0310907 
        x0=-0.743294
        b=20.1231
        c=101.578
        b1=(b*x0-c)/(c*x0)
        b2=(x0-b)/(c*x0) 
        b3=-1/(c*x0)
    else: raise ValueError("limit must be 0 or 1") 
    
    Xr=rs + b*np.sqrt(rs) + c
    Xx=x0**2 + b*np.sqrt(x0**2) + c
    Q=np.sqrt(4*c-b**2)
    
    t2=Q/(2*np.sqrt(rs)+b)
    t1=np.log( (np.sqrt(rs)-x0)**2/Xr) + 2*(b+2*x0)/Q * np.arctan(t2)
    t=np.log(rs/Xr) + 2*b/Q * np.arctan(t2) - b*x0*t1/Xx
    
    return a*t    

def Vcorrelation_GL(rs,limit=0):
    #paramagnetic limit 
    if limit==0:
        A=11.4 #15.9
        C=0.0666 #0.0406
    #ferromagentic limit
    elif limit==1:
        A=15.9 
        C=0.0406
    else: raise ValueError("limit must be 0 or 1") 
    
    x=rs/A
    
    return -C*np.log(1+1/x)

def Ecorrelation_GL(rs,limit=0):
    #paramagnetic limit
    if limit==0:
        A=11.4 #15.9
        C=0.0666 #0.0406
    #ferromagentic limit
    elif limit==1:
        A=15.9 
        C=0.0406
    else: raise ValueError("limit must be 0 or 1") 
    
    x=rs/A
    
    return -C*( (1+x**3)*np.log(1+1/x) + 1/2 *x - x**2 -1/3)

def Vcorrelation_PZ(rs,limit=0):
   #paramagentic limit
   if limit==0:
        g=-0.1324  #-0.0843 
        b1=1.0529 #1.3981
        b2=0.3334 #0.2611
        A=0.0311 #0.01555
        B=-0.048 #-0.0269
        C=0.0020 #0.0007
        D=-0.0116 #-0.0048
     #ferromagentic limit
   elif limit==1:
        g=-0.0843 
        b1=1.3981
        b2=0.2611
        A=0.01555
        B=-0.0269
        C=0.0007
        D=-0.0048
   else: raise ValueError("limit must be 0 or 1") 
    #vc=np.zeros_like(rs)
   vc=np.where(rs>=1,Ecorrelation_PZ(rs,limit)*(1+7/6*b1*np.sqrt(rs)+4/3*b2*rs)/(1+b1*np.sqrt(rs)+b2*rs),A*np.log(rs)+B-A/3+2/3*C*rs*np.log(rs)+1/3*(2*D-C)*rs)
    
   return vc
    
def Ecorrelation_PZ(rs,limit=0):    
   #paramagentic limit
   if limit==0:
        g=-0.1324  #-0.0843 
        b1=1.0529 #1.3981
        b2=0.3334 #0.2611
        A=0.0311 #0.01555
        B=-0.048 #-0.0269
        C=0.0020 #0.0007
        D=-0.0116 #-0.0048
     #ferromagentic limit
   elif limit==1:
        g=-0.0843 
        b1=1.3981
        b2=0.2611
        A=0.01555
        B=-0.0269
        C=0.0007
        D=-0.0048
   else: raise ValueError("limit must be 0 or 1") 
    #ec=np.zeros_like(rs)
   ec=np.where(rs>=1, g/(1+b1*np.sqrt(rs)+b2*rs), A*np.log(rs)+B+C*rs*np.log(rs)+D*rs)

   return ec







def V_correlation(rs,tip="MP", limit=0):
    if tip=="MP":
        return Vcorrelation_MP(rs)
    elif tip=="WVN":
        return Vcorrelation_WVN(rs,limit)
    elif tip=="GL":
        return Vcorrelation_GL(rs,limit)
    elif tip=="PZ":
        return Vcorrelation_PZ(rs,limit)
    else: raise ValueError("type not recognised")
    
def e_correlation(rs,tip="MP", limit=0):
    if tip=="MP":
        return Ecorrelation_MP(rs)
    elif tip=="WVN":
        return Ecorrelation_WVN(rs,limit)
    elif tip=="GL":
        return Ecorrelation_GL(rs,limit)
    elif tip=="PZ":
        return Ecorrelation_PZ(rs,limit)
    else: raise ValueError("type not recognised")
    
    
def Ecorrelation_WVN3(rs,limit=0):
    if limit==0:
        A=0.0621814
        x0=-0.409286
        b=13.0720
        c=42.7198
      #ferromagentic limit
    elif limit==1:
        A=0.0310907
        x0=-0.743294
        b=20.1231
        c=101.578
    else: raise ValueError("limit must be 0 or 1") 
    
    x=np.sqrt(rs)
    X=x**2 + b*x + c
    X0=x0**2 + b*x0 + c
    Q=np.sqrt(4*c-b**2)
    
    ec=A* ( np.log(x**2 / X) + 2*b/Q * np.arctan(Q/(2*x+b)) - b*x0/X0 * ( np.log( (x-x0)**2 / X) + 2*(b+2*x0)/Q * np.arctan(Q/(2*x+b)) ) )
    return ec

def Vcorrelation_WVN3(rs,limit=0):
    if limit==0:
        A=0.0621814
        x0=-0.409286
        b=13.0720
        c=42.7198
      #ferromagentic limit
    elif limit==1:
        A=0.0310907
        x0=-0.743294
        b=20.1231
        c=101.578
    else: raise ValueError("limit must be 0 or 1") 
    
    x=np.sqrt(rs)
    X=x**2 + b*x + c
    X0=x0**2 + b*x0 + c
    Q=np.sqrt(4*c-b**2)
    
    vc = Ecorrelation_WVN3(rs,limit) - (A/3) * (c * (x-x0) - b*x*x0)/((x-x0)*(x*x + b*x + c))
    return vc
    
    
    
#Accurate spin-dependent electron liquid correlation energies for local spin density calculations: a critical analysi    
    
    
    
    
    
    
    
    
    
