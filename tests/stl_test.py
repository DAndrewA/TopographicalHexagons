# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:17:51 2022

@author: Andrew

Testing the stl creation
"""

import numpy as np
from stl import mesh
from matplotlib import pyplot as py

import loadData as LD
import CoordinateTransform as CT
import Hexagon as Hexagon
import HexGrid as HexGrid


x = np.array(range(500))
y = np.array(range(500))

X,Y = np.meshgrid(x,y)

D = 20*np.sin((X+Y)*np.pi /180  +  ((Y-200)/100)**2)

py.contourf(D)

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
        
STL_tile.save('data\\_stl\\test.stl')