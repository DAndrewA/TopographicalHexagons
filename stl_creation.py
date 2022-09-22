# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:31:08 2022

@author: Andrew

Main script to control the taking of the data. This is based on the jupyter notebook
"""


import numpy as np
from stl import mesh


import loadData as LD
import CoordinateTransform as CT
import Hexagon as Hexagon
import HexGrid as HexGrid


#------------------------------------------------

seaVal = -10

D1,m1 = LD.load_asc_format('data\\srtm_35_04\\','srtm_35_04.asc',seaVal)
D2,m2 = LD.load_asc_format('data\\srtm_36_04\\','srtm_36_04.asc',seaVal)

D_combined = np.hstack((D1,D2))

#------------------------------------------------

newD,lcd = LD.downsample_minimum(D_combined)
newM = m1.copy()
newM[2] = newM[2] * lcd
Transform = CT.CoordinateTransform(metadata=newM)


coords = np.array([-1.974412, 43.215345]).reshape((2,1)) # roughly the first point I want
img = Transform.coords2Img(coords)

# create the hexagon centered on coords
targetH = Hexagon.Hexagon(550,3,img)

# create all the points in image space
x = range(0,np.size(newD,1))
y = range(0,np.size(newD,0))
#X,Y = np.meshgrid(x,y)
#print(np.shape(X))


####
NUMHEX = 550

v,f = HexGrid.layerAlgorithm(targetH,NUMHEX)

# showing the hexagons work themselves
#fig = plt.figure(figsize=(40,20))
#ax = fig.add_subplot(121)
#ax.scatter(v[0,:],v[1,:])


HexD = HexGrid.interpolateGrids(v,x,y,newD)

#ax = fig.add_subplot(111)
#ax.contourf(v[0,:],v[1,:],HexD)

#------------------------------------------------

v,HexD,f = HexGrid.generateHexBase(NUMHEX, v, HexD, f, baseVal=-50)

# SCALING of HexD

HexD[HexD > seaVal] = HexD[HexD > seaVal] * 100000

HexD[HexD == seaVal] = -20 


STL_tile = mesh.Mesh(np.zeros(f.shape[1], dtype=mesh.Mesh.dtype))
                     
for i, face in enumerate(f.T):
    for j,pos in enumerate(face):
        pos = int(pos)
        vec3 = [ *v[:,pos] , HexD[pos] ]
        STL_tile.vectors[i][j] = vec3
        
STL_tile.save('data\\_stl\\TILE1.stl')
print('Done :)')