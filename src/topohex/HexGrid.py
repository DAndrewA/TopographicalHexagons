# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 21:58:26 2022

@author: Andrew

This will contain the HexGrid class, that will create a hexagonal grid of points with a given spacing, rotation and extent

The grid will be computed as two sublattices, A and B. A will be centered on the origin, B will be offset in the y direction.
This will give consitency with the hexagons in Hexagon.py
"""

import numpy as np
import matplotlib.pyplot as plt
from . import Hexagon

hexSize = lambda x: 3 * x * (x+1) if x > -1 else -1
layerSize = lambda x: 6*x if x else 1


def layerAlgorithm(targetH,numHexInRadius=3):
    '''
    This algorithm to generate the desired hex grid will start from an initial center point.
    It will then proceed to walk outwards radially one step, and fill in a hexagonal layer.
    For each layer, the vertices and faces will be stored.
    Then it will step radially outward for the enw layer.    
    '''
    scale = targetH.scale / numHexInRadius
    rot = targetH.rotation
    tileH = Hexagon(scale,rot,[0,0]) # this will create a reduced scale hexagon with the target rotation at the origin
    
    vertices = np.array([0,0]).reshape((2,1))
    currentVertex = np.array([0,0]).reshape((2,1))
    
    translationOrder = [2,3,4,5,0,1]
    
    faces = []
    
    # for each layer we want vertices on:
    for layer in np.array(range(numHexInRadius))+1:
        # take the initial step out for the layer
        currentVertex = currentVertex + tileH.vertices[:,0].reshape((2,1))
        layerVertices = currentVertex
        
        for direction in translationOrder:
            for j in range(layer):
                currentVertex = currentVertex + tileH.vertices[:,direction].reshape((2,1))
                layerVertices = np.hstack((layerVertices,currentVertex))
                
        vertices = np.hstack((vertices,layerVertices[:,:-1]))
        
        # now we want to generate the faces (triangles) in each layer
        innerIndex = hexSize(layer-2) + 1
        outerIndex = hexSize(layer-1) + 1
        for direction in translationOrder:
            for j in range(layer-1):
                faces.append([ innerIndex , outerIndex, outerIndex+1 ])
                
                g = innerIndex+1
                if g > hexSize(layer-1): g = hexSize(layer-2)+1
                
                faces.append([ innerIndex , outerIndex+1 , g ]) # particular order to preserve right-handedness or triangles
                
                innerIndex += 1
                outerIndex += 1
            
            # now we've done all the ones along the outside edge, we need to do a corner triangle
            g = outerIndex+1
            if g > hexSize(layer): g = hexSize(layer-1)+1
            
            if innerIndex > hexSize(layer-1): innerIndex = hexSize(layer-2)+1
            
            faces.append([ innerIndex , outerIndex , g ])
            outerIndex += 1
            
    # adjust the origin so that the grid is centered on targetH
    vertices = vertices + targetH.centre
    return vertices, np.array(faces).astype(int)

def coerce_to_range(
    value: np.ndarray, 
    min: int,
    max: int
):
    '''Coerce values to a range if the value is outside of the range [min, max)'''
    too_low = value < min
    too_high = value >= max
    value[too_low] = min
    value[too_high] = max-1
    return value

# this weighting function is made so that x[y] = +-1 gives 0, and x[y] = 0 gives 1
#wFn = lambda x,y: np.cos(np.pi/2 * x)*np.cos(np.pi/2 * y) 
wFn = lambda x,y: (1 - np.abs(x))*(1 - np.abs(y))
def interpolateGrids(v, x, y, D, f=wFn):
    '''
    Function to interpolate the values D on an X-Y cartesian grid onto the HexGrid coordinates.
    This will also allow for a weighting function, f (i.e. linear vs quadratic interpolation.)
    
    This does work on the assumption that X,Y are cartesian grids with unit spacing begining at the origin.
    
    INPUTS:
        v: 2xa matrix of coordinate vectors for hexagon
        x: mx1 matrix, input for meshgrid for Cartesian coordinates
        y: nx1 matrix, input for meshgrid for Cartesian coordinates
        D: nxm matrix, data to be interpolated from Cartesian meshgrid to HexGrid
        f: lambda function, ndarray->ndarray, needs to be able to take multiple elements and act elementwise/ be numpy compatible 
    '''
    HexD = np.zeros((np.size(v,1))) # initialises the data matrix as 0
         
    # we can collapse X and Y to one dimension given they are Cartesian
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    nx,ny = x.size, y.size
    
    
    xUnder = coerce_to_range( np.floor(v[0,:]).astype(int), 0, nx)
    xAbove = coerce_to_range( np.ceil(v[0,:]).astype(int), 0, nx)
    print('xUnder shape: {}'.format(xUnder.shape))
    
    yUnder = coerce_to_range( np.floor(v[1,:]).astype(int), 0, ny)
    yAbove = coerce_to_range( np.ceil(v[1,:]).astype(int), 0, ny)
    print('yUnder shape: {}'.format(yUnder.shape))
    
    # we've now extracted the values of x and y in the grid adjacent to our hexgrid points
    # we will calculate a weighted mean of the corners
    D_xUyU = D[yUnder,xUnder]
    print('D_xUyU shape: {}'.format(D_xUyU.shape))
    D_xUyA = D[yAbove, xUnder]
    D_xAyU = D[yUnder, xAbove]
    D_xAyA = D[yAbove, xAbove]
    
    
    weights = np.array([ f(v[1,:] - yUnder, v[0,:] - xUnder), f(v[1,:] - yAbove, v[0,:] - xUnder), f(v[1,:] - yUnder, v[0,:] - xAbove), f(v[1,:] - yAbove, v[0,:] - xAbove) ])
    print('weights shape: {}'.format(weights.shape))
    
    HexD = ( weights[0,:]*D_xUyU + weights[1,:]*D_xUyA + weights[2,:]*D_xAyU + weights[3,:]*D_xAyA ) / np.sum(weights,axis=0)
    
    return HexD


def interpolate_xy_to_hex_coords(
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    '''Function to interpolate values from a data array D, onto the coordinates described in v, given that D is described over the grid XX,YY
    
    INPUTS:
        v: np.ndarray
            (2,a) array of x,y pairs of coordinates for vertices in the hexgrid
            
        x: np.ndarray
            (nx,) array of x coordinates that the data exists on.
            
        y: np.ndarray
            (ny,) array of y coordinates that the data exists on.
            
        D: np.ndarray
            (ny, nx) array of data values to be interpolated onto v

    OUTPUTS:
        HexD: np.ndarray
            (a,) array of interpolated values from D onto the vertices described in v
    '''
    nx = x.size
    ny = y.size
    a = v.shape[1]
    
    # append new axis to vx and vy to make products with x,y easier
    vx = v[0,:][...,np.newaxis]
    vy = v[1,:][...,np.newaxis]

    # inidices of the grid vertex below and to the left of each specified vertex
    index_x = np.sum(
        vx > x,
        axis=1
    )
    index_y = np.sum(
        vy > y,
        axis=1
    )
    # ensure lower indices are within bounds of array D
    index_x[ index_x >= nx ] = nx-1
    index_y[ index_y >= ny ] = ny-1

    print('interpolate_xy_to_hex_coords: indices calculated')

    # get values on corners surrounding each vertex
    #ll = D[
    #    index_y, index_x
    #]
    #lr = D[
    #    index_y, index_x+1
    #]
    #ul = D[
    #    index_y+1, index_x
    #]
    #ur = D[
    #    index_y+1, index_x+1
    #]
    print('interpolate_xy_to_hex_coords: corner D values identified')
    #xl = x[index_x]
    #xu = x[index_x+1]
    #yl = y[index_y]
    #yu = y[index_y+1]

    # reset dimensionality of vx, vy
    vx = v[0,:]
    vy = v[1,:]
    # bilinear interpolation
    HexD = np.zeros((a))
    # HexD += ll * (xu - vx) * (yu - vy)
    HexD += D[index_y, index_x] * (x[index_x+1] - vx) * (y[index_y+1] - vy)
    print('HexD ll')
    #HexD += lr * (vx - xl) * (yu - vy)
    HexD += D[index_y, index_x+1] * (vx - x[index_x]) * (y[index_y+1] - vy)
    print('HexD lr')
    # HexD += ul * (xu - vx) * (vy - yl)
    HexD += D[index_y+1, index_x] * (x[index_x+1] - vx) * (vy - y[index_y])
    print('HexD ul')
    #HexD += ur * (vx - xl) * (vy - yl)
    HexD += D[ index_y+1, index_x+1] * (vx - x[index_x]) * (vy - y[index_y])
    print('HexD ur')
    #HexD = HexD / (xu-xl) / (yu-yl)
    HexD = HexD / (x[index_x+1] - x[index_x]) / (y[index_y+1] - y[index_y])
    print('HexD divide')

    print('interpolate_xy_to_hex_coords: HexD calculated')
    return HexD



      

def generateHexBase(numHexInRadius, v, HexD, f, baseVal=-20):
    '''
    This function will be to create the faces indices for the base of the hexagon.
    This will only work for a flat base.
    It will use the outer layer of the hexagon to create flat edges at the sides.
    All of the vertices on the bottom will then have triangles with the inner-most point, located directly under the centre.    
    '''
    nv = v.shape[1] # the original number of vertices
    
    # firstly, need to get the last layer and repeat the x-y coordinates, and place a central coordinate at the bottom
    v = np.hstack((v,v[:,-layerSize(numHexInRadius):],v[:,0].reshape((2,1))))
    
    # now need to set the base value for those repeated coordinates
    HexD = np.hstack((HexD , np.ones(layerSize(numHexInRadius)+1)*baseVal))
    
    # This algorithm, simillar to the layerAlgorithm, will generate faces by walking the vertices around and creating faces in pairs
    upperIndex0 = nv - layerSize(numHexInRadius) #{+-1?}
    lowerIndex0 = nv
    
    iUpper = lambda x: (x-upperIndex0)%layerSize(numHexInRadius) + upperIndex0
    iLower = lambda x: (x-lowerIndex0)%layerSize(numHexInRadius) + lowerIndex0
    
    edgeFaces = np.zeros((2*layerSize(numHexInRadius),3))
    for j in range(layerSize(numHexInRadius)):
        edgeFaces[2*j , :] = [iLower(j), iLower(j+1), iUpper(j)]
        edgeFaces[2*j+1 , :] = [iLower(j+1), iUpper(j+1), iUpper(j)]
        
    
    baseFaces = np.zeros((layerSize(numHexInRadius),3))
    # once all of the edge faces have been generated, need to create the faces for the base
    for j in range(layerSize(numHexInRadius)):
        #baseFaces[j,:] = np.array([ j , (j+1)%layerSize(numHexInRadius) , layerSize(numHexInRadius)])
        baseFaces[j,:] = np.array([(j+1)%layerSize(numHexInRadius), j, layerSize(numHexInRadius)])
        
    baseFaces = baseFaces + nv # offset the indices by the number of original vertices
    
    #### NEED TO COMBINE baseFaces AND f
    f = np.vstack((f,edgeFaces,baseFaces)).astype(int)
    
    return v,HexD,f

      

def generateTexturedBase(numHexInRadius,HexD_top,HexD_bottom,layerOffset,v,f):
    '''
    This function will be used to generate the faces list required to stitch two hexagonal objects together vertically.
    '''
    if HexD_top.size != HexD_bottom.size:
        print('PROBLEM: sizes aren\'t the same')
        return None
    
    # The size of HexD should be the same for both, and the same as v[i,:] size
    nv = v.shape[1]
    print('{} vs {}'.format(HexD_top.size,nv))
    
    # need to stack two instances of v, and add on faces with the adjusted indices
    v = np.hstack((v,v))
    # inverting the handedness of the base faces to preserve inner-ness
    fbase = f.copy() + nv
    fbase[:,[0,1]] = fbase[:,[1,0]]
    f = np.vstack((f, fbase)).astype(int)
    del fbase
    
    HexD_top = np.hstack((HexD_top, HexD_bottom + layerOffset))
    
    upperIndex0 = nv - layerSize(numHexInRadius)
    lowerIndex0 = 2*nv - layerSize(numHexInRadius)
    iUpper = lambda x: (x-upperIndex0)%layerSize(numHexInRadius) + upperIndex0
    iLower = lambda x: (x-lowerIndex0 + int(layerSize(numHexInRadius)/2))%layerSize(numHexInRadius) + lowerIndex0
    
    edgeFaces = np.zeros((2*layerSize(numHexInRadius),3))
    for j in range(layerSize(numHexInRadius)):
        edgeFaces[2*j , :] = [iLower(j), iLower(j+1), iUpper(j)]
        edgeFaces[2*j+1 , :] = [iLower(j+1), iUpper(j+1), iUpper(j)]
        
    f = np.vstack((f,edgeFaces)).astype(int)
    
    return v,HexD_top,f


def getCornerIndices(numHexInRadius,cornerDepth,cornerLength):
    '''
    This function will be to generate a list of indices for a given number of hexagon radius of the grid, that satisfy the conditin of being "in a corner".
    
    INPUTS:
        numHexInRadius: int, as used in other functions, the total number of layers to the hexagonal grid.
        cornerDepth: int, <numHexInRadius, the depth of the corner indices provided, in terms of the layers.
        cornerLength: int, <numHexInRadius/2, the extent of where the corner indices are given from a corner (the corner is #0).
    OUTPUTS:
        cornerIndices: 1xm array, the indices of the grid-points that are decided to be a corner.
    '''
    numCorner = cornerDepth * (2*cornerLength - cornerDepth)
    cornerIndices = np.zeros((1,6*numCorner))

    # for each corner, we'll get the required indices
    for c in range(6):
        cIndices = np.zeros((1,numCorner))
        # for each layer in the corner
        cIndTrack = 0
        for l in range(cornerDepth):
            s = cornerLength - l
            cornerIndex = hexSize(numHexInRadius - l - 1) + c*(numHexInRadius-l)
            ind = (np.array(range(-s+1,s)).reshape((1,2*s-1)) + cornerIndex - hexSize(numHexInRadius-l-1) ) % layerSize(numHexInRadius - l) + hexSize(numHexInRadius-l-1)

            g = list(range(cIndTrack,cIndTrack + (2*s - 1))) # g is intorduced as a list of indices in the cIndices array that we want to change
            
            cIndices[0,g] = ind
            cIndTrack += 2*s - 1
        g = list(range(c*numCorner, (c+1)*numCorner))
        cornerIndices[0,g] = cIndices
    return cornerIndices.astype(int)
'''
o = 50

targetH = Hex.Hexagon(o,0,[0,0])
v,f = layerAlgorithm(targetH,o)

fig,ax = plt.subplots(figsize=(20,20))
plt.hold=True

ax.scatter(v[0,:],v[1,:])

#for i,face in enumerate(f):
#    #print('face {}: {}'.format(i,face))
#    a,b,c = [*face]
#    a = int(a)
#    b = int(b)
#    c = int(c)
#    
#    ax.plot(v[0,[a,b]],v[1,[a,b]])
#    ax.plot(v[0,[b,c]],v[1,[b,c]])
#    ax.plot(v[0,[c,a]],v[1,[c,a]])
'''