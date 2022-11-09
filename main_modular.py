'''
Created on Sun 30/10/22

Author: Andrew Martin

This will be a redesign of the original main.py script to generate the .stl files from the SRTM data.

This file will aim to be more modular than the original file. It will allow for boolean or string flags to be set at the top of the script defining how the different faces will be considered.

All of the same scaling variables will be available.

### STEPS:

We will start by initialising all of the required scaling variables, along with the flags required to designate how the model will be put together.

The next step will be to create the STL_tile object that will hold the stl data. This will be created with the correct number of vertices depending on how the bottom face is being handled.

After that, we will designate how the bottom face is to be done. This will either be loading in another image, or a flat face. This will be constant throuhgout the cerated hexagons, so can be set in advance.

Next, we will load in the SRTM data, and combine the relevant files. This will also be where we create the CoordinateTransform for the data.

We will also load in the JourneyCoordinates at this point if they are desired.

For each of the desired hexagons, we will sample the SRTM data, do any culling/ scaling, input the journey coordinate pins, and then save the HexGrid_data into the STL_file object. We will then save the file.
'''

##### INITIAL IMPORTS
from time import time
t0 = time()

import numpy as np
from stl import mesh
import imageio as imageio


import loadData as LD
import coordinateTransform as CT
import Hexagon as Hexagon
import HexGrid as HexGrid

# need a specific folder separation character, as differs between mac and windows
folderSep = '/'# '\\'

##### VARIABLE SETUP

# DESIGN VARIABLES

# the first set of variables are the flags for different features to be included in the tiles.
FLAG_textured_bottom = False # flag to load in an image to the bottom of the tile. Set to True to allow. False creates a flat bottom.
SHELL_loadFile = 'data' + folderSep + 'shell' + folderSep + 'smoothShell.png'
FLAG_journey_coordinates = True # flag to allow the inclusion of the journey coordinate pins in the tile. True places raised pins on the tile, False doesn't do anything.
JC_loadFile = 'data' + folderSep + '','journeyCoords.txt'
FLAG_raised_corners = False # flag for raising the top-side corners of the tile, for use in coaster creation. True to have raised corners, False does nothing to the tile. 


# SCALING VARIABLES

NUMHEX = 900

SAVE_folder = 'data' + folderSep + 'models' + folderSep
SAVE_filename = 'tile{}_r{}_d{}_h{}.stl'

# The PRINTER variables define the scalings for the hexagons dimensions when they are printed.
PRINTER_units_per_mm = 1
PRINTER_desired_radius = 50#40 # in mm, this is how long a side of the hexagon is. Largest diameter, corner to corner is 2x.
PRINTER_desired_depth = 7#15 # im mm, this is the depth before any topographical displacement is considered.
PRINTER_desired_height = 10 # in mm, the desired height ASL that the highest value of all the hexagons will be scaled to.
PRINTER_sea_depression = -0.5 # in mm, the depression from PRINTER_desired_depth that the sea-designated points (NaN in data) will be dset to.


# the SHELL variables determine the properties of the textured bottom of the tile if they are desired to be included.
SHELL_depression_mm = 1 # in mm, the depth of the shell when used in the model.

# the JC variables determine how the pins for the Journey Coordinates are implemented onto the tile.
JC_radius = 20#15 # in pixel units of the SRTM image.
JC_height = 3 # in mm
JC_roundingHeight = 0.75 # in mm

# the CORNER variables determine the properties of the raised corners that can be generated for the tiles.
CORNER_desired_height = PRINTER_desired_height + 10 # in mm
CORNER_scale = 0.1 # proportional width of the corner pieces.
CORNER_depth = int(CORNER_scale * NUMHEX / 2)
CORNER_length = int(CORNER_scale * NUMHEX)
CORNER_glassDepth=6 # in mm

# Variables used in loading the SRTM data and in culling the large negative values produced in the final tile.
LOAD_seaVal = -10000
SRTM_cullValue = LOAD_seaVal / 25 # cutoff value in the HexGrid data that will designate if a spot is the sea or not

# defining the hexagon onto which the final printed data will be mapped. This allows us to generate the HexData and then place it onto our arbitrarily picked printing coordinates. 
PRINTER_Hexagon = Hexagon.Hexagon(PRINTER_desired_radius*PRINTER_units_per_mm)


##### STEP 1: creating the STL_tile object
# The STL_tile object will be created in the stl package, adn will be created to have the corect size depending on the type of bottom face being used.
print('STEP 1: creating the STL_tile object: ')
# using the print hexagon, generate a list of vertice coordinates and faces.
v0, f0 = HexGrid.layerAlgorithm(PRINTER_Hexagon,NUMHEX)
nv = v0.shape[1] # the number of vertices on the top face of the hexagon
nf = f0.shape[0] # the number of faces for the top face of the hexagon.

HexD = np.zeros((nv)) # initialise the HexD variable for storing the height data of the vertices.
print('Top-side vertices and faces loaded.')

# At this point, we need to decide if we want a textured bottom or a flat bottom:

if FLAG_textured_bottom:
    # if the textured bottom is desired, we need to load in the data and put it into the heightmap data.
    print('Loading textured bottom face from {}'.format(SHELL_loadFile))
    # load in the smooth shell heightmap
    SHELL_heightmap = imageio.imread(SHELL_loadFile)
    SHELL_heightmap = -SHELL_heightmap / np.max(SHELL_heightmap) * SHELL_depression_mm * PRINTER_units_per_mm

    # The following code is to place the hexagon over the image in an apropriate location and orientation.
    x = range(SHELL_heightmap.shape[1])
    y = range(SHELL_heightmap.shape[0])
    SHELL_scale = 540 * 2 / np.sqrt(3)
    SHELL_centre = np.array([520,508]).reshape(2,1)
    SHELL_rotation = -90
    SHELL_shift = np.array([200,100]).reshape((2,1))
    SHELL_args = [SHELL_scale,SHELL_rotation, SHELL_centre + SHELL_shift ]
    SHELL_Hexagon = Hexagon.Hexagon(*SHELL_args)
    vbase,__ = HexGrid.layerAlgorithm(SHELL_Hexagon,NUMHEX)

    # SHELL_heightmap is the equivelent of HexD_bottom, so needs to be appended onto HexD_top equiv.
    SHELL_heightmap = HexGrid.interpolateGrids(vbase, x, y, SHELL_heightmap)
    SHELL_heightmap = SHELL_heightmap - (PRINTER_desired_depth * PRINTER_units_per_mm)
    del vbase, __, x, y # clear the memory space for more variables

    v0,HexD,f0 = HexGrid.generateTexturedBase(NUMHEX, HexD,SHELL_heightmap,-PRINTER_desired_depth,v0,f0)
    del SHELL_heightmap
    print('SHELL loading successful.')

else: # in this case, no textured bottom is desired, we just require the flat bottom
    print('Creating flat bottom to tile.')
    v0,HexD,f0 = HexGrid.generateHexBase(NUMHEX, v0, HexD, f0, -PRINTER_desired_depth)


# create the STL object in memory. We can then reuse it to speed up the saving process
print('Initialising STL object: ',end='')
STL_tile = mesh.Mesh(np.zeros(f0.shape[0], dtype=mesh.Mesh.dtype))
                     
for i, face in enumerate(f0):
    for j,pos in enumerate(face):
        STL_tile.vectors[i][j] = [*v0[:,pos], HexD[pos]]

#del v0, f0, HexD # save space, these will be reinstantiated later
print('success')


##### STEP 2: Load in the SRTM data and the JourneyCoordinates if desired.

# Now we can start on loading in the SRTM data, getting the hexagons and saving the individual .stl files
# the minimum and maximum occuring values in the SRTM dataset -70.0 : 3258.0
PRINTER_mm_per_heightunit = PRINTER_desired_height / 2650#3258

# load in the SRTM data, as standard. We won't downsample (yet)
print('Loading SRTM data: ',end='')

D1,m1 = LD.load_asc_format('data' + folderSep + 'srtm_35_04' + folderSep + '','srtm_35_04.asc',LOAD_seaVal)
D2,m2 = LD.load_asc_format('data' + folderSep + 'srtm_36_04' + folderSep + '','srtm_36_04.asc',LOAD_seaVal)

SRTM_D = np.hstack((D1,D2))
SRTM_x = range(SRTM_D.shape[1])
SRTM_y = range(SRTM_D.shape[0])

del D1, D2
print('success')

# Also want to scale SRTM_D here to avoid artifacts in the hexagonalisation later.
SRTM_D[SRTM_D > SRTM_cullValue] = SRTM_D[SRTM_D > SRTM_cullValue] * PRINTER_mm_per_heightunit * PRINTER_units_per_mm
SRTM_D[SRTM_D <= SRTM_cullValue] = PRINTER_sea_depression*PRINTER_units_per_mm   


# create the coordinate transform. Uncomment line (+2) to downsample
lcd = 1
#SRTM_D,lcd = LD.downsample_minimum(SRTM_D)
m1[2] = m1[2] * lcd
Transform = CT.CoordinateTransform(metadata=m1)

##### STEP 3: Load in the Journey Coordinates if desired, and the initial hexagon. Will also get the indices for the raised corners if desired.

if FLAG_journey_coordinates:
    # extract the path coordinates from the journeyCoords.txt file, and convert them into image space coordinates.
    pathCoords,pathNames = LD.load_coordinate_list('data' + folderSep + '','journeyCoords.txt')
    pathImgCoords = Transform.coords2Img(pathCoords)

    radiusFn = lambda x: np.sqrt(np.sum(x*x ,axis = 0))


if FLAG_raised_corners:
    ##### SECTION FOR OBTAINING THE CORNER INDICES FOR RAISED CORNERS (use in coasters)
    CORNER_indices = HexGrid.getCornerIndices(NUMHEX,CORNER_depth,CORNER_length)
    ld = CORNER_length-CORNER_depth
    CORNER_glassIndices = HexGrid.getCornerIndices(NUMHEX-CORNER_depth,ld,ld)


# load in the initial hexagon from which the tiles will be itteratively generated.
# example config from Jupyter notebook
SRTM_scale = 480 * 2
SRTM_rotation = 2
SRTM_centre_coords = np.array([-1.86, 43.25]).reshape((2,1))
SRTM_centre = Transform.coords2Img(SRTM_centre_coords)
SRTM_args = [SRTM_scale,SRTM_rotation,SRTM_centre]

SRTM_Hexagon = Hexagon.Hexagon(*SRTM_args)

nextHexagons = [3] * 6 # this array designates the sides off of which the next hexagon will be created. It must be of the length of the desired number of hexagons, but the last entry can be arbitray.
# i.e. [3]*6 will generate 6 hexagons, each one to the left of the previous one in the local reference frame of the hexagons. However, nextHexagons[-1] can be any value.



##### STEP 4: itterate throuhg the hexagons, generating the tiles and saving them.

for tile, directionNext in enumerate(nextHexagons): # for each tile,
    # start by creating the vertices of the specific hexagon, and extracting the heightmap
    print('Creating Hexagon {}: '.format(tile),end='')
    SRTM_v = HexGrid.layerAlgorithm(SRTM_Hexagon,NUMHEX)[0] # don't bother extracting the faces object, as f0 already created
    print('vertices created; ',end='')
    # extract the SRTM data into the hexagonal grid
    SRTM_HexD = HexGrid.interpolateGrids(SRTM_v,SRTM_x,SRTM_y,SRTM_D)
 
    print('data extracted; ',end='')
    
    if FLAG_journey_coordinates:
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
    
    if FLAG_raised_corners:
        # INSERT the corner indices additional height
        SRTM_HexD[CORNER_indices] = CORNER_desired_height * PRINTER_units_per_mm
        SRTM_HexD[CORNER_glassIndices] = (CORNER_desired_height - CORNER_glassDepth) * PRINTER_units_per_mm

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
    SRTM_Hexagon = SRTM_Hexagon.createAdjacentHexagon(directionNext)
    
    
print('Mission success!')

tf=time()
print('t0: {}'.format(t0))
print('tf: {}'.format(tf))
print('elapsed time: {}'.format(tf-t0))
    