# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:31:08 2022

@author: Andrew

Main script to control the taking of the data. This is based on the jupyter notebook
"""


import numpy as np
from stl import mesh
import imageio as imageio


import loadData as LD
import CoordinateTransform as CT
import Hexagon as Hexagon
import HexGrid as HexGrid


#------------------------------------------------

seaVal = -1000

D1,m1 = LD.load_asc_format('data\\srtm_35_04\\','srtm_35_04.asc',seaVal)
D2,m2 = LD.load_asc_format('data\\srtm_36_04\\','srtm_36_04.asc',seaVal)

D_combined = np.hstack((D1,D2))

#------------------------------------------------

del D1,D2 # save space

lcd = 1
newD = D_combined.copy()
#newD,lcd = LD.downsample_minimum(D_combined)

newM = m1.copy()
newM[2] = newM[2] * lcd
Transform = CT.CoordinateTransform(metadata=newM)


coords = np.array([-1.974412, 43.215345]).reshape((2,1)) # roughly the first point I want
img = Transform.coords2Img(coords)

HEXSIZE = int(1100/lcd)

# create the hexagon centered on coords
targetH = Hexagon.Hexagon(HEXSIZE,3,img)

# create all the points in image space
x = range(0,np.size(newD,1))
y = range(0,np.size(newD,0))
#X,Y = np.meshgrid(x,y)
#print(np.shape(X))


####
NUMHEX = 900

v,f = HexGrid.layerAlgorithm(targetH,NUMHEX)

# showing the hexagons work themselves
#fig = plt.figure(figsize=(40,20))
#ax = fig.add_subplot(121)
#ax.scatter(v[0,:],v[1,:])


#newWFn = lambda x,y: np.cos(np.pi/2 * x)*np.cos(np.pi/2 * y) + 0.5

HexD = HexGrid.interpolateGrids(v,x,y,newD)#,newWFn)

newSeaVal = -10

HexD[HexD < newSeaVal] = newSeaVal


################### Add in the journey coords
# extract the path coordinates from the journeyCoords.txt file, and convert them into image space coordinates.
pathCoords,pathNames = LD.load_coordinate_list('data\\','journeyCoords.txt')
pathImgCoords = Transform.coords2Img(pathCoords)

JCoords_radius = 25 # in pixel units
JCoords_height = 300
JC_rounding = 2
radiusFn = lambda x: np.sqrt(np.sum(x*x ,axis = 0))

for JC in pathImgCoords.T:
    JC = JC.reshape((2,1))
    JC_displacement = radiusFn(v - JC)
    truth = JC_displacement <= JCoords_radius
    HexD[truth] = np.sum(HexD[truth]) / np.sum(truth)
    HexD[truth] = JCoords_height - JC_displacement[truth] * JC_rounding





#ax = fig.add_subplot(111)
#ax.contourf(v[0,:],v[1,:],HexD)

#------------------------------------------------
baseVal = -400
'''
v,HexD,f = HexGrid.generateHexBase(NUMHEX, v, HexD, f, baseVal)
'''
baseHeightScale = 50
# I will load in the smoothed shell image, create a target hexagon, create a grid for the smoothed shell, and use it as the base to the stl model
smoothShell = imageio.imread('data\\shell\\smoothShell.png')
smoothShell = -smoothShell / np.max(smoothShell) * baseHeightScale

x = range(smoothShell.shape[1])
y = range(smoothShell.shape[0])
baseScale = 540 * 2 / np.sqrt(3)
baseCentre = np.array([520,508]).reshape(2,1)
baseRotation = -90
baseShift = np.array([200,100]).reshape((2,1))

baseTargetH = Hexagon.Hexagon(baseScale,baseRotation,baseCentre+baseShift)
vbase,__ = HexGrid.layerAlgorithm(baseTargetH,NUMHEX)
HexD_base = HexGrid.interpolateGrids(vbase, x, y, smoothShell)

del vbase, __

v,HexD,f = HexGrid.generateTexturedBase(NUMHEX, HexD, HexD_base, baseVal, v, f)





# SCALING of HexD

scaleVal = 0.1

HexD[HexD > newSeaVal] = HexD[HexD > newSeaVal] * scaleVal



STL_tile = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))
                     
for i, face in enumerate(f):
    for j,pos in enumerate(face):
        pos = int(pos)
        vec3 = [ *v[:,pos] , HexD[pos] ]
        STL_tile.vectors[i][j] = vec3#[ *v[:,pos] , HexD[pos] ]
 
        
# create a relevant STL filename so I can keep track of everything I've done
tileNum = '1JC'
folderName = 'data\\_stl\\' 
fileName = 'TILE{}_n{}_b{:d}_c{:d}_s{}.stl'.format(tileNum,NUMHEX,-baseVal,-newSeaVal,scaleVal)

STL_tile.save('data\\_stl\\' + fileName)
print('File saved as {}'.format(fileName))
print('Done :)')
