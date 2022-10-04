# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 02:28:52 2022

@author: Andrew

The actual script to do the thing that I want to do!

We will start by loading in the smoothed shell image, and extracting that to use as the base for the models

We will then load in the SRTM data.

We will then create the first hexagon. We will extract the grid coordinates, and thus the grid values.

We will then discard the vertices in favour of a target hexagon centered on the origin with no rotation, that we can scale so that it works with the printer.
The HexD values can also be scaled by a set amount.
We will save the first stl file.

We will then create the second target hexagon, extract the data, etc.
"""
from time import time
t0 = time()

import numpy as np
from stl import mesh
import imageio as imageio


import loadData as LD
import CoordinateTransform as CT
import Hexagon as Hexagon
import HexGrid as HexGrid

# start by defining all the constants for the program
NUMHEX = 900

SAVE_folder = 'data\\models\\'
SAVE_filename = 'tile{}_r{}_d{}_h{}.stl'



# The hexagon vertices and faces for 3d printing
PRINTER_units_per_mm = 1
PRINTER_desired_radius = 40 # in mm
PRINTER_desired_depth = 7#15 # im mm
PRINTER_desired_height = 10 # in mm
SHELL_depression_mm = 1
PRINTER_sea_depression = -0.5 # in mm
LOAD_seaVal = -10000
SRTM_cullValue = LOAD_seaVal / 25
JC_radius = 15 # in pixel units
JC_height = 3 # in mm
JC_roundingHeight = 0.75 # in mm

PRINTER_Hexagon = Hexagon.Hexagon(PRINTER_desired_radius*PRINTER_units_per_mm)


print('Creating print coordinates: ',end='')
v0, f0 = HexGrid.layerAlgorithm(PRINTER_Hexagon,NUMHEX)
nv = v0.shape[1]

HexD = np.zeros((nv))
# this pre-creates the vertices and faces objects that can simply be reused
print('stacking layers: ',end='')
v0,HexD,f0 = HexGrid.generateHexBase(NUMHEX, v0, HexD, f0, -PRINTER_desired_depth)
print('success')

# create the STL object in memory. We can then reuse it to speed up the savig process
print('Initialising STL object: ',end='')
STL_tile = mesh.Mesh(np.zeros(f0.shape[0], dtype=mesh.Mesh.dtype))
                     
for i, face in enumerate(f0):
    for j,pos in enumerate(face):
        STL_tile.vectors[i][j][[0,1]] = [*v0[:,pos]]

del v0 # save space
print('success')


''' # NOT LOADING IN THE SHELL
# Now we will load in the shell and extract the heightmap for it.
print('Loading in the shell data: ',end='')

# load in the smooth shell heightmap
SHELL_heightmap = imageio.imread('data\\shell\\smoothShell.png')
SHELL_heightmap = -SHELL_heightmap / np.max(SHELL_heightmap) * SHELL_depression_mm * PRINTER_units_per_mm

x = range(SHELL_heightmap.shape[1])
y = range(SHELL_heightmap.shape[0])
SHELL_scale = 540 * 2 / np.sqrt(3)
SHELL_centre = np.array([520,508]).reshape(2,1)
SHELL_rotation = -90
SHELL_shift = np.array([200,100]).reshape((2,1))
SHELL_args = [SHELL_scale,SHELL_rotation, SHELL_centre + SHELL_shift ]

SHELL_Hexagon = Hexagon.Hexagon(*SHELL_args)
vbase,__ = HexGrid.layerAlgorithm(SHELL_Hexagon,NUMHEX)

print('hexagon created; ',end='')
# SHELL_heightmap is the equivelent of HexD_bottom, so needs to be appended onto HexD_top equiv.
SHELL_heightmap = HexGrid.interpolateGrids(vbase, x, y, SHELL_heightmap)
SHELL_heightmap = SHELL_heightmap - (PRINTER_desired_depth * PRINTER_units_per_mm)
del vbase, __, x, y
print('success')
'''

# Now we can start on loading in the SRTM data, getting the hexagons and saving the individual .stl files
# the minimum and maximum occuring values in the SRTM dataset -70.0 : 3258.0
PRINTER_mm_per_heightunit = PRINTER_desired_height / 3258

# load in the SRTM data, as standard. We won't downsample (yet)
print('Loading SRTM data: ',end='')

D1,m1 = LD.load_asc_format('data\\srtm_35_04\\','srtm_35_04.asc',LOAD_seaVal)
D2,m2 = LD.load_asc_format('data\\srtm_36_04\\','srtm_36_04.asc',LOAD_seaVal)

SRTM_D = np.hstack((D1,D2))
SRTM_x = range(SRTM_D.shape[1])
SRTM_y = range(SRTM_D.shape[0])

del D1, D2
print('success')

# create the coordinate transform. Uncomment line (+2) to downsample
lcd = 1
#SRTM_D,lcd = LD.downsample_minimum(SRTM_D)
m1[2] = m1[2] * lcd
Transform = CT.CoordinateTransform(metadata=m1)


###################################################################
###################################################################
######### NEED SECTION FOR LOADING IN JOURNEY COORDINATES #########
###################################################################

# extract the path coordinates from the journeyCoords.txt file, and convert them into image space coordinates.
pathCoords,pathNames = LD.load_coordinate_list('data\\','journeyCoords.txt')
pathImgCoords = Transform.coords2Img(pathCoords)

radiusFn = lambda x: np.sqrt(np.sum(x*x ,axis = 0))


###################################################################
########### AND PLACING THEM INTO THE SRTM_D MATRIX... ############
###################################################################
###################################################################
print('NEED TO LOAD IN JOURNEY COORDINATES')

# example config from Jupyter notebook
SRTM_scale = 480 * 2
SRTM_rotation = 2
SRTM_centre_coords = np.array([-1.86, 43.25]).reshape((2,1))
SRTM_centre = Transform.coords2Img(SRTM_centre_coords)
SRTM_args = [SRTM_scale,SRTM_rotation,SRTM_centre]

SRTM_Hexagon = Hexagon.Hexagon(*SRTM_args)

for tile in np.array([1,2,3,4,5,6]): # for each tile,
    # start by creating the vertices of the specific hexagon, and extracting the heightmap
    print('Creating Hexagon {}: '.format(tile),end='')
    SRTM_v, __ = HexGrid.layerAlgorithm(SRTM_Hexagon,NUMHEX)
    del __
    print('vertices created; ',end='')
    # extract the SRTM data into the hexagonal grid
    SRTM_HexD = HexGrid.interpolateGrids(SRTM_v,SRTM_x,SRTM_y,SRTM_D)
    
    # Now perform the appropriate scaling
    SRTM_HexD[SRTM_HexD > SRTM_cullValue] = SRTM_HexD[SRTM_HexD > SRTM_cullValue] * PRINTER_mm_per_heightunit * PRINTER_units_per_mm
    SRTM_HexD[SRTM_HexD <= SRTM_cullValue] = PRINTER_sea_depression*PRINTER_units_per_mm    
    print('data extracted; ',end='')
    
    print('Inserting Journey Coords')
    # add in the journey coordinates
    for JC in pathImgCoords.T:
        JC = JC.reshape((2,1))
        JC_displacement = radiusFn(SRTM_v - JC)
        truth = JC_displacement <= JC_radius
        if np.sum(truth): # only if there are points within the hexagon that are valid 
            averageValue = np.sum(SRTM_HexD[truth]) / np.sum(truth)
            print('averageValue={}'.format(averageValue))
            #print('averageValue.shape={}'.format(averageValue.shape))
            SRTM_HexD[truth] = averageValue + (JC_height + np.sqrt( JC_radius**2 - JC_displacement[truth]*JC_displacement[truth] )/JC_radius * JC_roundingHeight)*PRINTER_units_per_mm
    print('Inserting JourneyCoords: success')
    
    # now append the shell data to the SRTM data
    #SRTM_HexD = np.hstack((SRTM_HexD,SHELL_heightmap))
    HexD[:nv] = SRTM_HexD
    for i, face in enumerate(f0):
        for j,pos in enumerate(face):
            STL_tile.vectors[i][j][2] = HexD[pos]

    print('success')
    fname = SAVE_filename.format(tile,PRINTER_desired_radius,PRINTER_desired_depth,PRINTER_desired_height)
    STL_tile.save(SAVE_folder + fname)
    print('File saved as {}'.format(fname))

    # create the next hexagon to be turned into a tile
    SRTM_Hexagon = SRTM_Hexagon.createAdjacentHexagon(3)
    
    
print('Mission success!')

tf=time()
print('t0: {}'.format(t0))
print('tf: {}'.format(tf))
print('elapsed time: {}'.format(tf-t0))