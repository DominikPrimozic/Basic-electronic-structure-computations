# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:11:00 2024

@author: domin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:13:53 2024

@author: domin
"""
#for c++

import numpy as np

def atom_separations(molecule):
    #distance array

    n_atoms = len(molecule.charges)
    separations = np.zeros((n_atoms, n_atoms))
    for i, a in enumerate(molecule.position):
        for j, b in enumerate(molecule.position):
            separations[i,j] = np.linalg.norm(a - b)

    return separations

def gridMetric(ty=2):
    if ty==1:
        radial  = {1:10, 2:15}
        angular = {1:11, 2:15}
        lebedev = {11:50, 15:86} #13:74,17:110,19:146,21:170,23:194,25:230
    if ty==2:
        radial  = {1:50, 2:75}
        angular = {1:29, 2:29}
        lebedev = {29: 302} #31:350, 35:434, 41:590
    
    return radial, angular, lebedev

def MK_radial_grid(r_grid,atom,m=3):
    # Mura, M. E., & Knowles, P. J. (1996). Improved radial grids for quadrature in molecular density-functional calculations. The Journal of Chemical Physics, 104(24), 9848–9858. https://doi.org/10.1063/1.471749
    
    #choose f
    if atom in [3,4,11,12,19,20]: a=7.0
    else: a=5.0
    
    #r_grid is the size of grid 
    r = np.empty(r_grid) 
    dr = np.empty(r_grid)
    
    for i in range(r_grid):
        x= (i+0.5)/r_grid #initilization of points on interval 0-1
        #LOG3 grid (m=3)
        r[i]=-a*np.log(1-x**m)
        dr[i]=a*m*x**(m-1)/((1-x**m) * r_grid) #why *r_grid though
    
    return r, dr

def radial_pruning(lebedev_grid,r_grid):
    #Treutler, O., & Ahlrichs, R. (1995). Efficient molecular numerical integration schemes. The Journal of Chemical Physics, 102(1), 346–354. https://doi.org/10.1063/1.469408
    pruned=np.empty(r_grid,dtype=int)
    
    pruned[:r_grid//3]=14
    pruned[r_grid//3:r_grid//2]=50
    pruned[r_grid//2:]=lebedev_grid
    
    return pruned #number of angular parts for each radial

def SB_scheme(coords): #elipitcal coordinates uij
    #Stratmann, R. E., Scuseria, G. E., & Frisch, M. J. (1996). Achieving linear scaling in exchange-correlation density functional quadratures. Chemical Physics Letters, 257(3–4), 213–223. https://doi.org/10.1016/0009-2614(96)00600-8
    a=0.64
    m=coords/a
    g=np.array( (1/16) * (35*m - 35*np.power(m,3) + 21*np.power(m,5)-5*np.power(m,7) ) ) #z polynomial in article

    g[coords <= -a] = -1
    g[coords >=  a] =  1

    return g  

def treutlerAdjust(mol): #atomic size adjustments
    #Treutler, O., & Ahlrichs, R. (1995). Efficient molecular numerical integration schemes. The Journal of Chemical Physics, 102(1), 346–354. https://doi.org/10.1063/1.469408
    #Becke, A. D. (1988). A multicenter numerical integration scheme for polyatomic molecules. The Journal of Chemical Physics, 88(4), 2547–2553. https://doi.org/10.1063/1.454033
    atoms=mol.charges
    bragg = {'1':0.35,'2':1.40,'3':1.45,'4':1.05,'5':0.85,'6':0.70,'7':0.65,'8':0.60,'9':0.50, '10':1.50} 
    r= np.sqrt([bragg[str(s)] for s in atoms])
    
    atoms=len(mol.charges)
    a = np.zeros((atoms, atoms))
    for i in range(atoms):
        for j in range(i+1, atoms):
            """
            X=Ri/Rj
            uij=(X-1)/(X+1)=(Ri-Rj)/(Ri+Rj)
            aij=uij/(uij**2-1)
            """
            a[i,j] = 1/4 * (r[i]-r[j])*(r[i]+r[j])/(r[i]*r[j])
            a[j,i] = -a[i,j]

    a[a < -0.5] = -0.5 #a has to be in range -0.5,0.5 for monotonicity (see Becke appendix)
    a[a > 0.5] = 0.5
 
    return a

def spheric_symmetry(points, a, b, v):
    if points == 0:
        n = 6 
        a = 1.0
        shell = {'-a':[4,13,22],'+a':[0,9,18],'-b':[],'+b':[],'v':[3,7,11,15,19,23]}

    elif points == 1:
        n = 12
        a = np.sqrt(0.5)
        shell = {'-a':[5,10,13,14,20,26,28,30,36,41,44,45],'+a':[1,2,6,9,16,18,22,24,32,33,37,40],'-b':[],'+b':[], \
                  'v':list(range(3, 48, 4))}

    elif points == 2:
        n = 8
        a = np.sqrt(1.0/3.0)
        shell = {'-a':[4,9,12,13,18,20,22,25,26,28,29,30],'+a':[0,1,2,5,6,8,10,14,16,17,21,24], '-b':[],'+b':[], \
                  'v':list(range(3, 32, 4))}

    elif points == 3:
        n = 24
        b = np.sqrt(1.0 - 2.0*a*a)
        shell = {'+a':[0,1,5,8,16,17,21,24,32,34,38,40,42,46,48,56,65,66,69,70,74,78,81,85], \
                 '-a':[4,9,12,13,20,25,28,29,36,44,50,52,54,58,60,62,73,77,82,86,89,90,93,94], \
                 '+b':[2,6,10,14,33,37,49,53,64,72,80,88], \
                 '-b':[18,22,26,30,41,45,57,61,68,76,84,92], \
                  'v':list(range(3, 96, 4))} 

    elif points == 4:
        n = 24
        b = np.sqrt(1.0 - a*a)
        shell = {'+a':[0,8,17,21,32,40,50,54,65,73,82,86], \
                 '-a':[4,12,25,29,36,44,58,62,69,77,90,94], \
                 '+b':[1,5,16,24,34,38,48,56,66,70,81,89], \
                 '-b':[9,13,20,28,42,46,52,60,74,78,85,93], \
                  'v':list(range(3, 96, 4))}

    elif points == 5:
        n = 48
        c = np.sqrt(1.0 - a*a - b*b)
        shell = {'+a':[0,8,16,24,32,40,48,56,65,69,81,85,98,102,106,110,129,133,145,149,162,166,170,174], \
                 '-a':[4,12,20,28,36,44,52,60,73,77,89,93,114,118,122,126,137,141,153,157,178,182,186,190], \
                 '+b':[1,5,17,21,34,38,42,46,64,72,80,88,96,104,112,120,130,134,138,142,161,165,177,181], \
                 '-b':[9,13,25,29,50,54,58,62,68,76,84,92,100,108,116,124,146,150,154,158,169,173,185,189], \
                 '+c':[2,6,10,14,33,37,49,53,66,70,74,78,97,101,113,117,128,136,144,152,160,168,176,184], \
                 '-c':[18,22,26,30,41,45,57,61,82,86,90,94,105,109,121,125,132,140,148,156,164,172,180,188], \
                 'v' :list(range(3, 192,4))}

    if points in range(0,5): shell['+c'] = [] ; shell['-c'] = []
    pts = np.zeros(n*4)

    for i in range(n*4):
        if i in shell['-a']   : pts[i] = -a
        elif i in shell['+a'] : pts[i] = a
        elif i in shell['-b'] : pts[i] = -b
        elif i in shell['+b'] : pts[i] = b
        elif i in shell['-c'] : pts[i] = -c
        elif i in shell['+c'] : pts[i] = c
        elif i in shell['v']  : pts[i] = v

    return n, pts

def lebedev_grid(order):
   if order == 14:

       sum = 2
       a = [0.0] * sum ; b = [0.0] * sum
       v = [0.6666666666666667e-1, 0.7500000000000000e-1]
       n = [0, 2]
       
   elif order == 50:

       sum = 4
       b = [0.0] * sum
       v = [0.1269841269841270e-1, 0.2257495590828924e-1, 0.2109375000000000e-1, 0.2017333553791887e-1]
       a = [0.0, 0.0, 0.0, 0.3015113445777636e+0]
       n = [0, 1, 2, 3]

   elif order == 86:

       sum = 5
       b = [0.0] * sum
       v = [0.1154401154401154e-1, 0.1194390908585628e-1, 0.1111055571060340e-1, 0.1187650129453714e-1, 0.1181230374690448e-1]
       a = [0.0, 0.0, 0.3696028464541502e+0, 0.6943540066026664e+0, 0.3742430390903412e+0]
       n = [0, 2, 3, 3, 4]

   elif order == 302:

       sum = 12
       v = [0.8545911725128148e-3, 0.3599119285025571e-2, 0.3449788424305883e-2, 0.3604822601419882e-2, 0.3576729661743367e-2, \
            0.2352101413689164e-2, 0.3108953122413675e-2, 0.3650045807677255e-2, 0.2982344963171804e-2, 0.3600820932216460e-2, \
            0.3571540554273387e-2, 0.3392312205006170e-2]
       a = [0.0,0.0,0.3515640345570105,0.6566329410219612,0.4729054132581005,0.9618308522614784e-1,0.2219645236294178, \
            0.7011766416089545, 0.2644152887060663, 0.5718955891878961, 0.2510034751770465, 0.1233548532583327]
       b = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.8000727494073952,0.4127724083168531]
       n = [0,2,3,3,3,3,3,3,4,4,5,5]
       
   returned_order = [0]*sum
   g = np.empty((0, 4))
   for i in range(sum):
        returned_order[i], lg = spheric_symmetry(n[i], a[i], b[i], v[i])
        g = np.vstack((g, lg.reshape(returned_order[i],4)))

   return np.sum(returned_order), g


def build_grids(mol,ty):
    atoms=sorted(set(mol.charges))[::-1] #unique atoms
    atom_weights={}
    radial, angular, lebedev = gridMetric(ty)
    
    for atom in atoms:
        
        if atom not in atom_weights:
            #only for two periods for now
            if atom>2:
                period=2
            elif atom<=2:
                period=1
            
            r_grid=radial[period] #get numeber of radial points
            r, dr = MK_radial_grid(r_grid,atom) #atom is number, which is the charge
            #radial weights
            w=4*np.pi*r*r*dr #4np.pi*r**2 from jacobian, dr is Mura weights
            
            
            lebedev_points=lebedev[angular[period]] #get lebedev order
            
            pruned_radial=radial_pruning(lebedev_points, r_grid) #sets angular grid
            
            coords=[]
            volume_w=[]
            
            for n in sorted(set(pruned_radial)): #unique 
                
               
                points, weights = lebedev_grid(n)
                
                
                #assert points, n
                
                indices=np.where(pruned_radial==n)[0] #find radial indicies of this angular

                coords.append(np.einsum("i,jk->jik", r[indices], weights[:,:3]).reshape(-1,3))
                volume_w.append(np.einsum("i,j->ji", w[indices], weights[:,3]).ravel())
                
            atom_weights[atom] = np.vstack(coords), np.hstack(volume_w)
            
    return atom_weights

def partition(mol,coords, n_atoms, n_grids):
    
    a=treutlerAdjust(mol)
    adjust = lambda i, j, g : g + a[i,j]*(1.0 - g**2) #vij=uij+aij*(1-uij)
    
    meshes = np.empty((n_atoms, n_grids))
    separations = atom_separations(mol)
    #print(separations)
    
    #this get rig,rjg - distances from grid points for each atom 
    for i in range(n_atoms):
        c=coords-mol.position[i] #translation to atomic center
        meshes[i]=np.sqrt(np.einsum("ij,ij->i", c,c)) #root(x**2+y**2+z**2) for each triplet
        
    becke_partition = np.ones((n_atoms, n_grids))
    
    for i in range(n_atoms):
        for j in range(i):
            separations[separations==0]=1 #for atom case
            
            u=(1/separations[i,j]) * (meshes[i] - meshes[j]) #Becke's transformation to eliptical coordiantes, antisymetric with respect to index swithc
            
            v=adjust(i,j,u)
            
            g=SB_scheme(v) #all this perserves antisymmetry due to index exchange
            
            becke_partition[i] *= 0.5 * (1.0 - g) #cell function s, pk=product of ski over all atoms but i!=k (product over indices up to num of atoms)
            becke_partition[j] *= 0.5 * (1.0 + g) # since aij=-aji and gij=-gji can lessen loops by doing it already with + instead of -
            
    return becke_partition

def grid_partition(mol, atom_grid_table):
    mol_coords=mol.position
    n_atoms=len(mol.charges)
    
    coords=[]
    weights=[]
    
    for ato in range(n_atoms):
        a_coords,a_volume=atom_grid_table[mol.charges[ato]]
        
        a_coords=a_coords + mol.position[ato] #translate
        
        n_grids=a_coords.shape[0]
        becke_partition=partition(mol, a_coords, n_atoms, n_grids)
        
        w=a_volume * becke_partition[ato] /becke_partition.sum(axis=0) #normalized weights enetering the quadrature
        
        coords.append(a_coords)
        weights.append(w)
        
    coords = np.vstack(coords)
    weights = np.hstack(weights)
        
    return coords, weights

class molecule():
    def __init__(self,charge,position):
        self.charges=np.array(charge)
        self.position=position

def GRID(mol,ty):
    
    
    atom_grid_table= build_grids(mol,ty)        
    
    coords, w = grid_partition(mol, atom_grid_table)
    
    return coords, w
