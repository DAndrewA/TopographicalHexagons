'''Author: Andrew Martin
Creation date: 12/7/24

Script for using topohex to create topographical hexagons for a LEJOG cycling route.

Topography data is from the SRTM data, stored in ./data/srtm in the .asc format
'''

import numpy as np
import imageio
import os
from copy import deepcopy

from stl import mesh

import topohex as tph
from topohex.Hexagon import Hexagon

NUMHEX = 900

DIR_out = 'data/stl'
FNAME_out_fmt = '{}_r{}_d{}_h{}.stl' # tilename, radius, depth, height

PRINTER_units_per_mm = 1
PRINTER_desired_radius = 60 # (mm), edge length and longest radius of hexagon
PRINTER_desired_depth = 7 # (mm), thickness of the tile before any topographical displacement is considered
PRINTER_desired_height = 15 # (mm), the largest desired thickness of the tile, once maximum topographical displacement is considered.
PRINTER_sea_depression = -0.5 # (mm), depression from PRINTER_desired_depth that the sea-designated points in the data will be set to.

LOAD_seaval = -10000
SRTM_cullvalue = LOAD_seaval / 25 # emperical cutoff value used in the HexGrid data that will designate if a spot is ocean or not.

JC_radius = 0.3#0.5#20 # in units of km within the of the SRTM image
JC_height = 3.2 # (mm) height of the Journey Coordinates pin
JC_rounding_height = 0.75 # (mm)

# Extra Scaling derived from the f-plane projection of the angular coordinates
es = np.cos(np.deg2rad(55))

# initialisation of the master hexagon onto which height values will be projected.
PRINTER_hexagon = Hexagon(
    scale = PRINTER_desired_radius * PRINTER_units_per_mm
)

# create the coordinates for the PRINTER_hexagon
# This accounts for the horizontal coordinates and face connections, vertical data will be inserted for each hexagon.
v0,f0 = tph.hexgrid.layerAlgorithm(PRINTER_hexagon, NUMHEX)
nv = v0.shape[1] # number of vertices in the hexgrid

HexD = np.zeros((nv))
HexD_bottom = np.zeros((nv))

# create a 12mm (diameter)x2mm(depth) depression in the base of the tile :: .2 mm is used as tolerance
rf = lambda x: np.sqrt( np.sum(x*x, axis=0) )
magnet_hole = rf(v0) <= 6.1 * PRINTER_units_per_mm # 3.2
HexD_bottom[magnet_hole] = 2.2*PRINTER_units_per_mm
# generate a textured base
v0, HexD, f0 = tph.hexgrid.generateTexturedBase(NUMHEX, HexD, HexD_bottom, -PRINTER_desired_depth, v0, f0)

#v0, HexD, f0 = tph.hexgrid.generateHexBase(NUMHEX, v0, HexD, f0, -PRINTER_desired_depth)
print(f'{np.max(v0)=}')
# initialise the STL object in memory, with horizontal coordinates used. Vertical coordinates will be inserted throughout the script

STL_tile = mesh.Mesh(
    np.zeros(f0.shape[0], dtype=mesh.Mesh.dtype)
)
for i, face in enumerate(f0): # for each face
    for j, pos in enumerate(face): # for each vertex of the triangular face
        STL_tile.vectors[i][j][ [0,1] ] = [*v0[:,pos]] # update the horizontal location of the vertex to correspond to the HexBase grid

# after this, v0 is not required
del v0


# the minimum and maximum occuring values in the SRTM dataset sea_value : 1335.0
PRINTER_mm_per_heightunit = PRINTER_desired_height / 2650#3258

def load_SRTM_lejog(sea_val: int = LOAD_seaval):
    '''Function to load in the SRTM lejog data, remove Ireland and France, pad the southern edge with water, and return the 2d array
    Also returns XX,YY in reprojected coordinates to account for latitude'''
    n_rows, n_cols = 6000,6000 # default hard coded values obtained from .asc files

    relative_path = ['.', 'data', 'srtm']
    load_dir = os.path.join(*relative_path)
    # values returned from tph.loadData.load_asc_format are the data, and the metadata for the grid
    print('LOADING DATA:')
    def load_asc(names):
        loaded_data = {}
        for name in names:
            fname = f'srtm_{name}.asc'
            print(f'{name}...', end='')
            try:
                D, m = tph.load.load_asc_format(load_dir, fname, seaVal=sea_val)
                print('success')
            except Exception as err: # assume exceptions arise from non-existant files, replace with sea val grid
                print(f' => {err}')
                D = np.full( (n_rows, n_cols), sea_val )
            loaded_data[name] = D
        
        return loaded_data

    names = (
        '35_01', '35_02',
        '36_01', '36_02',
        '37_01', '37_02',
    )

    loaded_data = load_asc(names)
    print('Removing undesirable lands...', end='')
    # remove Ireland
    loaded_data['35_01'][:501, :4841] = sea_val
    loaded_data['35_02'][1700:, :5481] = sea_val
    # remove France
    loaded_data['37_02'][ :1201, 1450: ] = sea_val
    loaded_data['37_02'][ :, 2200: ] = sea_val
    print('success')

    print('Extending the sea...', end='')
    # stack the data
    loaded_data = np.vstack((
        np.hstack(( loaded_data['35_02'], loaded_data['36_02'], loaded_data['37_02'] )),
        np.hstack(( loaded_data['35_01'], loaded_data['36_01'], loaded_data['37_01'] ))
    ))
    print('success')

    OCEAN_PAD_HEIGHT = 2000
    ocean_padding = np.full( (OCEAN_PAD_HEIGHT,18000), sea_val )
    loaded_data = np.vstack((ocean_padding, loaded_data))

    ## 55deg North for the XX - YY coords...
    es = np.cos(np.deg2rad(55)) # accounting for lines of constant latitude being small-circles, not great-circles like lines of constant longitude
    CT = tph.coordtrans.CoordinateTransform(
        ref1Img=[12000, 6000 + OCEAN_PAD_HEIGHT],
        ref1Coords=[0, 55],
        scale=[5/6000 * es, 5/6000], # angle per pixel
    )

    x_img = np.arange(18000)
    y_img = np.arange(12000 + OCEAN_PAD_HEIGHT)

    x_geo, _ = CT.img2Coords(
        [ x_img, x_img ]
    )#x_geo.shape, y_geo.shape

    _, y_geo = CT.img2Coords(
        [ y_img, y_img ]
    )
    del _
    #XXg,YYg = np.meshgrid(x_geo, y_geo) # actually un-needed in the process

    return {
        'data': loaded_data,
        'CT': CT,
        'xg': x_geo,
        'yg': y_geo,
    }

LOADED = load_SRTM_lejog()

# vertically scale the loaded data
scale_factor=1000
scale_func = lambda x: scale_factor*np.log( x/scale_factor + 1 )
ld = LOADED['data']
LOADED['data'][ld >= 0] = scale_func(ld[ld >= 0])

###########################
##### INITIAL HEXAGON ON LOADED DATA
###########################
# This is taken from the notebook determine_hexagons.ipynb
SRTM_scale = 1.75 # transformed degrees
SRTM_rotation = 27 # degrees
# loaction determined for roughly middle of Scotland
coord_start_geo = np.array([-3.555, 55.5]).reshape((2,1))
coord_start_geo[0] *= es # extra scaling for longitude conversion
SRTM_centre_coords = np.array(coord_start_geo).reshape((2,1))
SRTM_args = [SRTM_scale,SRTM_rotation,SRTM_centre_coords]
SRTM_Hexagon = Hexagon(*SRTM_args)

# NOTE: we have to implement the bilinear interpolation function for tph.hexgrid.layerAlgorithm, given we are not using the image grid
dxg = np.mean(np.diff(LOADED['xg']))
dyg = np.mean(np.diff(LOADED['yg']))
wFn = lambda x, y: ( 1 - np.abs(x)/dxg )*( 1 - np.abs(y)/dxy )



# TODO: implement journey coordinates!!!!
# extract the path coordinates from the journeyCoords.txt file. Apply coordinate transform to map into XXg,YYg (cosine latitude transform)
pathCoords,pathNames = tph.load.load_coordinate_list(
    os.path.join('.','data'),
    'journey_coords.txt'
)
# transform the coords from the coordinate space to the transformed coordinate space
pathCoords[0,:] *= es
pathImgCoords = LOADED['CT'].coords2Img(pathCoords)
for n,c in zip(pathNames, pathCoords.T): print(f'{n:>25} @ {c}')
#radiusFn = lambda x: np.sqrt(np.sum(x*x ,axis = 0))
# altering the radius function to work in transformed-coordinate space, not image space
# conversion to km: [pixels] * (angle per pixel) * (km per degree)
conv = (5/6000) * (6371 * np.pi/180)
radiusFn = lambda x: np.sqrt( np.sum(x*x, axis=0) ) / conv



def apply_height_to_image_hexagon(
    SRTM_v: np.ndarray, # vertices extracted from layerAlgorithm
    Image: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
):
    '''Function to interpolate the ImageHex grid onto the Image space, and then apply the height data onto the ImageHex.'''
    # start by creating the vertices of the Hexgaon in the transformed coordinate space.
    #SRTM_v, _ = tph.hexgrid.layerAlgorithm(ImageHex, NUMHEX) # this goes in the main function!!!
    print('begining height interpolation...',end='')
    # NOTE: interpolateGrids works on integer indices for the grid, so we have to promote the coordinates from the transformed coordinate space into the image space coordinates. This can be done using the coordinate transform on the SRTM_v values.
    SRTM_Hex_h = tph.hexgrid.interpolateGrids(SRTM_v,xg,yg,Image)#, f=wFn)
    print('complete')
    # scale to the appropriate height
    SRTM_Hex_h[SRTM_Hex_h > SRTM_cullvalue] = SRTM_Hex_h[SRTM_Hex_h > SRTM_cullvalue] * PRINTER_mm_per_heightunit * PRINTER_units_per_mm
    SRTM_Hex_h[SRTM_Hex_h <= SRTM_cullvalue] = PRINTER_sea_depression*PRINTER_units_per_mm 
    return SRTM_Hex_h


def apply_journey_coordinates_to_image_hexagon(
    SRTM_v: np.ndarray, # lower case v here representing transformed-coordinate space, not image space...
    SRTM_Hex_h: np.ndarray
):
    '''Function to interpolate the JOURNEY COORDINATES onto the hexagonal grid and apply the new heights from the pins to the image hexagon'''
    print('Inserting Journey Coords...',end='')
    # add in the journey coordinates
    no_truth = True
    for JC in pathCoords.T:
        JC = JC.reshape((2,1))
        JC_displacement = radiusFn(SRTM_v - JC)
        truth = JC_displacement <= JC_radius
        print(f'{np.min(JC_displacement)}, {np.max(JC_displacement)}')
        if np.sum(truth): # only if there are points within the hexagon that are valid 
            averageValue = np.sum(SRTM_Hex_h[truth]) / np.sum(truth)
            #print('averageValue.shape={}'.format(averageValue.shape))
            SRTM_Hex_h[truth] = averageValue + (JC_height + np.sqrt( JC_radius**2 - JC_displacement[truth]*JC_displacement[truth] )/JC_radius * JC_rounding_height)*PRINTER_units_per_mm
            no_truth = False
    if no_truth: print('NO TRUTH VALUES...un', end='')
    print('success')
    return SRTM_Hex_h


def apply_image_hexagon_to_print_hexagon_vertices(ImageHex_height):
    '''Function that takes the heightmap from the image hexagon and applies it to the printing hexagon, and subsequently the STL_tile vectors heights'''
    # PrintHex_height required as it includes the base heights also
    print('Applying heights to HexD...',end='')
    HexD[:nv] = ImageHex_height
    print('success')
    print('Applying heights to STL_tile.vectors...', end='')
    for i, face in enumerate(f0):
        for j,pos in enumerate(face):
            # only affect the z=2 coordinate of the vectors
            STL_tile.vectors[i][j][2] = HexD[pos]
    print('success')


def apply_base_image_to_print_hexagon_vertices(BaseImageHex_height):
    '''For a desired base pattern, applies the heightmap from the base image onto the printing hexagonal grid.
    Any base image should start from 0 and show relief as depressions in the surface, accounting for +ve displacement. i.e. min(BaseImageHex_height) == 0
    '''
    # PrintHex_height required as it includes the base heights also
    print('Applying base heights to HexD...',end='')
    HexD[nv:] = BaseImageHex_height
    print('success')
    print('Applying base heights to STL_tile.vectors...', end='')
    for i, face in enumerate(f0):
        for j,pos in enumerate(face):
            # only affect the z=2 coordinate of the vectors
            STL_tile.vectors[i][j][2] = HexD[pos]
    print('success')


def create_STL_from_hexagon(ImageHex: Hexagon, tilename: str):
    '''Function that takes a Hexagon defined in the transformed coordinate space, extracts the relevant height information from the image and applies it to the print STL.
    Then  the print STL is saved to a relevant file.
    '''
    print('*-'*25 + '*\n\n')
    print(f'create_STL_from_hexagon: {tilename=}')
    SRTM_v, _ = tph.hexgrid.layerAlgorithm(ImageHex, NUMHEX)

    # promote the SRTM_v coordinates from the transformed coordinate space into the image space
    SRTM_V = LOADED['CT'].coords2Img(SRTM_v)
    x,_ = LOADED['CT'].coords2Img( np.vstack((LOADED['xg'], LOADED['xg'])) )
    y,_ = LOADED['CT'].coords2Img( np.vstack((LOADED['yg'], LOADED['yg'])) )

    print(f'new x: {np.min(x)} -> {np.max(x)}')
    print(f'new y: {np.min(y)} -> {np.max(y)}')
    print(f'coord range on SRTM_v:')
    print(f'SRTM_v x: {np.min(SRTM_v[0])} -> {np.max(SRTM_v[0])}')
    print(f'SRTM_v y: {np.min(SRTM_v[1])} -> {np.max(SRTM_v[1])}')
    print(f'SRTM_V x: {np.min(SRTM_V[0])} -> {np.max(SRTM_V[0])}')
    print(f'SRTM_V y: {np.min(SRTM_V[1])} -> {np.max(SRTM_V[1])}')
    # NOTE: SRTM_V rather than _v as it has had its coordinates promoted...
    SRTM_h = apply_height_to_image_hexagon(SRTM_V, LOADED['data'], x, y)
    SRTM_h = apply_journey_coordinates_to_image_hexagon(SRTM_v, SRTM_h) # journey coordinates applied using transformed-coordinate space, not image space
    apply_image_hexagon_to_print_hexagon_vertices(SRTM_h)
    #apply_base_image_to_print_hexagon_vertices() # applying the base displacement only needs to be performed once if the base image is constant...

    fname = FNAME_out_fmt.format(tilename, PRINTER_desired_radius, PRINTER_desired_depth, PRINTER_desired_height)
    STL_tile.save(os.path.join(DIR_out, fname))
    print(f'\nSTL saved as {fname=}\n')


# test!!! create root hexagon stl
# root, highlands, islands, lincs, wales, london, dorsetetal
name = 'root'
names_first_propogation = ('highlands', 'islands', 'lincs')
names_second_propogation = ('wales', 'london', 'dorsetetal')
names = (name, *names_first_propogation, *names_second_propogation)
first_propogation = (1,2,5)
second_propogation = (3,5,3)
 # apply heights to the root and create hexagon!
hexagons = [SRTM_Hexagon]
for face in first_propogation:
    H = SRTM_Hexagon.createAdjacentHexagon(face)
    hexagons.append(deepcopy(H))
for face in second_propogation:
    H = H.createAdjacentHexagon(face)
    hexagons.append(deepcopy(H))

print('Hexagons and names created')
print('Starting (image -> print) mapping')
for H, tname in zip(hexagons, names):
    create_STL_from_hexagon(H, tname)
#name = names[0]
#H = hexagons[0]
#create_STL_from_hexagon(H, name)

