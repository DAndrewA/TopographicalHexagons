'''Author: Andrew Martin
Creation date: 12/7/24

Script for using topohex to create topographical hexagons for a LEJOG cycling route.

Topography data is from the SRTM data, stored in ./data/srtm in the .asc format
'''

import numpy as np
import imageio
import os

from stl import mesh

import topohex
import topohex.HexGrid
import topohex.Hexagon

NUMHEX = 900

DIR_out = 'data/stl'
FNAME_out_fmt = 'tile_r{}_d{}_h{}.stl'

PRINTER_units_per_mm = 1
PRINTER_desired_radius = 50 # (mm), edge length and longest radius of hexagon
PRINTER_desired_depth = 7 # (mm), thickness of the tile before any topographical displacement is considered
PRINTER_desired_height = 10 # (mm), the largest desired thickness of the tile, once maximum topographical displacement is considered.
PRINTER_sea_depression = -0.5 # (mm), depression from PRINTER_desired_depth that the sea-designated points in the data will be set to.

LOAD_seaval = -10000
SRTM_cullvalue = LOAD_seaval / 25 # emperical cutoff value used in the HexGrid data that will designate if a spot is ocean or not.

JC_radius = 20 # in pixel units of the SRTM image
JC_height = 3 # (mm) height of the Journey Coordinates pin
JC_rounding_height = 0.75 # (mm)

# initialisation of the master hexagon onto which height values will be projected.
PRINTER_hexagon = topohex.Hexagon.Hexagon(
    scale = PRINTER_desired_radius * PRINTER_units_per_mm
)

# create the coordinates for the PRINTER_hexagon
# This accounts for the horizontal coordinates and face connections, vertical data will be inserted for each hexagon.
v0,f0 = topohex.HexGrid.layerAlgorithm(PRINTER_hexagon, NUMHEX)
nv = v0.shape[1] # number of vertices in the hexgrid

HexD = np.zeros((nv))
v0, HexD, f0 = topohex.HexGrid.generateHexBase(NUMHEX, v0, HexD, f0, -PRINTER_desired_depth)

# initialise the STL object in memory, with horizontal coordinates used. Vertical coordinates will be inserted throughout the script

STL_tile = mesh.Mesh(
    np.zeros(f0.shape[0], dtype=mesh.Mesh.dtype)
)
for i, face in enumerate(f0): # for each face
    for j, pos in enumerate(face): # for each vertex of the triangular face
        STL_tile.vectors[i][j][ [0,1] ] = [*v0[:,pos]] # update the horizontal location of the vertex to correspond to the HexBase grid

# after this, v0 is not required
del v0

'''
TODO
 - [ ] load in the SRTM data
 - [ ] change SRTM grid interpolation such that we don't need to load in an additional 2 tiles (non-convex geometry of domain)
 - [ ] Implement code to traverse a path described by hexagon faces, that can include non-straight line
    IDEA: "." implements a root hexagon (location and rotation provided)
        ".n" describes a hexagon generated off the nth face of the root (n in [0,5])
        ".nmj" would describe the hexagon off the jth face of the hexagon off the mth face of the hexagon off the nth face of the root hexagon. New hexagon generation is efficient so this is reasonable...   
'''