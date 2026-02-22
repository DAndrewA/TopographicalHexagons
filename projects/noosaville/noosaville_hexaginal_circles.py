'''Author: Andrew Martin
Creation date: 3/11/24

Script for using topohex to create topographical hexagons for the Noosaville area.

Topography data is 30m SRTM, obtained from https://portal.opentopography.org/

NOTE: in this script, we will apply a transformation to the grid coordinates to transform them into a circular pattern, rather than the traditional hexagon this package provides.
'''

# NOTE
# dxy = 29 in loading: possible beats pattern with 30m which is already displaced from grid -- interpolation to hex grid might not be spikey?
# sensititvity of spikiness to NUMHEX?
# also, 12km seems good, although 6km is also recognisable




import numpy as np
import imageio
import os
from copy import deepcopy

from stl import mesh

import topohex as tph
from topohex.Hexagon import Hexagon

import cartopy.crs as ccrs
from cartopy import img_transform as cimgt

NUMHEX = 700 # 900

DIR_out = 'data/stl'
FNAME_out_fmt = 'noosaville_circle_r{}_d{}_h{}.stl' # radius, depth, height

PRINTER_units_per_mm = 1
PRINTER_desired_radius = 15 # (mm), edge length and longest radius of hexagon
PRINTER_desired_depth = 4 # (mm), thickness of the tile before any topographical displacement is considered
PRINTER_desired_height =5 # (mm), the largest desired thickness of the tile, once maximum topographical displacement is considered.
PRINTER_sea_depression = -1 # (mm), depression from PRINTER_desired_depth that the sea-designated points in the data will be set to.

LOAD_seaval = -10000
SRTM_cullvalue = LOAD_seaval / 25 # emperical cutoff value used in the HexGrid data that will designate if a spot is ocean or not.


# initialisation of the master hexagon onto which height values will be projected.
PRINTER_hexagon = Hexagon(
    scale = PRINTER_desired_radius * PRINTER_units_per_mm
)

# create the coordinates for the PRINTER_hexagon
# This accounts for the horizontal coordinates and face connections, vertical data will be inserted for each hexagon.
v0,f0 = tph.hexgrid.layerAlgorithm(PRINTER_hexagon, NUMHEX)
nv = v0.shape[1] # number of vertices in the hexgrid


# apply scaling to the vertices in order to produce a circle
angles = np.arctan2(
    v0[0,:],
    v0[1,:]
)
radial_scale_factor = 2 * np.cos( 
    np.deg2rad(
        np.mod(np.rad2deg(angles), 60) - 30
    )
) / np.sqrt(3)
v0 = v0 * radial_scale_factor


HexD = np.zeros((nv))

# untextured base, saves on hexagons
v0, HexD, f0 = tph.hexgrid.generateHexBase(NUMHEX, v0, HexD, f0, -PRINTER_desired_depth)
print(f'{np.max(v0)=}')
# initialise the STL object in memory, with horizontal coordinates used. Vertical coordinates will be inserted throughout the script

STL_tile = mesh.Mesh(
    np.zeros(f0.shape[0], dtype=mesh.Mesh.dtype)
)
for i, face in enumerate(f0): # for each face
    for j, pos in enumerate(face): # for each vertex of the triangular face
        STL_tile.vectors[i][j][ [0,1] ] = [*v0[:,pos]] # update the horizontal location of the vertex to correspond to the HexBase grid

# after this, v0 is not required
del v0, angles


# scale by roughly the largest value in the target area
PRINTER_mm_per_heightunit = PRINTER_desired_height / 532

def load_SRTM(sea_val: int = LOAD_seaval):
    '''Function to load in the SRTM data for noosaville
    Also returns XX,YY in reprojected coordinates'''

    relative_path = ['.', 'data', 'srtm','rasters_SRTMGL1']
    load_dir = os.path.join(*relative_path)
    
    fname='output_SRTMGL1.asc'
    # values returned from tph.loadData.load_asc_format are the data, and the metadata for the grid
    print('LOADING DATA:')
    
    IMG,m = tph.load.load_asc_format(
        load_dir, fname, seaVal=sea_val
    )
    print('success')
    
    # convert  coordinates into east and north distances
    # native coordinates
    ny, nx = IMG.shape

    x0 = m[0]
    y0 = m[1]
    dxy = m[2]

    x = x0 + np.arange(nx)*dxy
    y = y0 + np.arange(ny)*dxy
    x,y = np.meshgrid(x,y)

    target_centre = {
        'lon':153.047693,
        'lat':-26.412448,
    }

    proj_native = ccrs.PlateCarree()
    proj_target = ccrs.Orthographic(
        central_longitude = target_centre['lon'],
        central_latitude = target_centre['lat']
    )

    # target coordinates
    r_lim = 30_000
    dxy = 19 # 30 #15 # 15m resolution should interpolate between grid points spaced at 30m

    lim = r_lim - r_lim%dxy + dxy

    target_xy = np.arange(-lim, lim+1, dxy)
    target_x, target_y = np.meshgrid(target_xy, target_xy)

    # transform the image from PlateCarree to the centered orthographic, and interpolate onto the provided target coords
    IMG_transformed = cimgt.regrid(
        array=IMG,
        source_x_coords=x, 
        source_y_coords=y,
        source_proj=proj_native,
        target_proj=proj_target,
        target_x_points=target_x,
        target_y_points=target_y
    )

    return IMG_transformed, target_xy, target_x, target_y

IMG, txy, XX, YY = load_SRTM()

print(f'IMG min={np.min(IMG)}, max={np.max(IMG)}')

# vertically scale the loaded data
#scale_factor=1000
#scale_func = lambda x: x #scale_factor*np.log( x/scale_factor + 1 )
#IMG[IMG >= 0] = scale_func(IMG[ IMG >= 0 ])

###########################
##### INITIAL HEXAGON ON LOADED DATA
###########################
# This is taken from the notebook determine_hexagons.ipynb
SRTM_scale = 12_000 # 12000 # radius of circle (hexagon edge length) in m
SRTM_rotation = 0 # degrees
SRTM_centre_coords = np.array([0,0]).reshape((2,1)) # in the new orthographic coordinates, centre on the origin
SRTM_args = [SRTM_scale,SRTM_rotation,SRTM_centre_coords]
SRTM_Hexagon = Hexagon(*SRTM_args)



def apply_height_to_image_hexagon(
    SRTM_v: np.ndarray, # vertices extracted from layerAlgorithm
    Image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
):
    '''Function to interpolate the ImageHex grid onto the Image space, and then apply the height data onto the ImageHex.'''
    # start by creating the vertices of the Hexgaon in the transformed coordinate space.
    #SRTM_v, _ = tph.hexgrid.layerAlgorithm(ImageHex, NUMHEX) # this goes in the main function!!!
    print('begining height interpolation...',end='')
    # NOTE: interpolateGrids works on integer indices for the grid, so we have to promote the coordinates from the transformed coordinate space into the image space coordinates. This can be done using the coordinate transform on the SRTM_v values.
    SRTM_Hex_h = tph.hexgrid.interpolate_xy_to_hex_coords(SRTM_v,x,y,Image)#, f=wFn)
    print('complete')
    print(f'SRTM_hex_h min={np.min(SRTM_Hex_h)}, max={np.max(SRTM_Hex_h)}')
    # scale to the appropriate height
    SRTM_Hex_h[SRTM_Hex_h > SRTM_cullvalue] = SRTM_Hex_h[SRTM_Hex_h > SRTM_cullvalue] * PRINTER_mm_per_heightunit * PRINTER_units_per_mm
    SRTM_Hex_h[SRTM_Hex_h <= SRTM_cullvalue] = PRINTER_sea_depression*PRINTER_units_per_mm 
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



def create_STL_from_hexagon(ImageHex: Hexagon):
    '''Function that takes a Hexagon defined in the transformed coordinate space, extracts the relevant height information from the image and applies it to the print STL.
    Then  the print STL is saved to a relevant file.
    '''
    print('*-'*25 + '*\n\n')
    print(f'create_STL_from_hexagon')
    SRTM_v, _ = tph.hexgrid.layerAlgorithm(ImageHex, NUMHEX)

    # apply scaling to make circular
    SRTM_v = SRTM_v * radial_scale_factor


    SRTM_h = apply_height_to_image_hexagon(SRTM_v, IMG, txy, txy)
    apply_image_hexagon_to_print_hexagon_vertices(SRTM_h)
    #apply_base_image_to_print_hexagon_vertices() # applying the base displacement only needs to be performed once if the base image is constant...

    fname = FNAME_out_fmt.format(PRINTER_desired_radius, PRINTER_desired_depth, PRINTER_desired_height)
    STL_tile.save(os.path.join(DIR_out, fname))
    print(f'\nSTL saved as {fname=}\n')



create_STL_from_hexagon(SRTM_Hexagon)

