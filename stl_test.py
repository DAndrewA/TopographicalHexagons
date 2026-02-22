# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:17:51 2022

@author: Andrew

Testing the stl creation
"""

import numpy as np
import os
from stl import mesh
import matplotlib.pyplot as plt

from topohex import Hexagon, HexGrid


x = np.array(range(500))
y = np.array(range(500))

X,Y = np.meshgrid(x,y)

D = 20*np.sin((X+Y)*np.pi /180  +  ((Y-200)/100)**2)

plt.contourf(D)

centre = np.array([200,200]).reshape((2,1))

targetH = Hexagon.Hexagon(60,0,centre)


NUMHEX = 250
v,f = HexGrid.layerAlgorithm(targetH,NUMHEX)

HexD = HexGrid.interpolateGrids(v,x,y,D)

baseVal = -22
v,HexD,f = HexGrid.generateHexBase(NUMHEX, v, HexD, f, baseVal)


STL_tile = mesh.Mesh(np.zeros(f.shape[1], dtype=mesh.Mesh.dtype))
                     
for i, face in enumerate(f.T):
    for j,pos in enumerate(face):
        pos = int(pos)
        vec3 = [ *v[:,pos] , HexD[pos] ]
        STL_tile.vectors[i][j] = vec3

save_dir = os.path.dirname(
    os.path.abspath(__file__)
)      
STL_tile.save(
    os.path.join(save_dir,'test.stl')
)