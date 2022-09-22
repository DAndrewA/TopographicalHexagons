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
import Hexagon as Hex



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
    tileH = Hex.Hexagon(scale,rot,[0,0]) # this will create a reduced scale hexagon with the target rotation at the origin
    
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
    return vertices, faces



def interpolateGrids(v,X,Y,D,f=lambda x,y:-np.sqrt(x*x + y*y)+1):
    '''
    Function to interpolate the values D on an X-Y cartesian grid onto the HexGrid coordinates.
    This will also allow for a weighting function, f (i.e. linear vs quadratic interpolation.)
    
    This does work on the assumption that X,Y are cartesian grids with unit spacing begining at the origin.
    
    INPUTS:
        v: 2xa matrix of coordinate vectors for hexagon
        X: nxm matrix, output of meshgrid for Cartesian coordinates
        Y: nxm matrix, output of meshgrid for Cartesian coordinates
        D: nxm matrix, data to be interpolated from Cartesian meshgrid to HexGrid
        f: lambda function, ndarray->ndarray, needs to be able to take multiple elements and act elementwise/ be numpy compatible 
    '''
    nv = np.size(v,1) # number of vertices
    
    HexD = np.zeros((np.size(v,1))) # initialises the data matrix as 0
         
    # we can collapse X and Y to one dimension given they are Cartesian
    x = np.unique(X,axis=0).reshape((np.size(X,1),))
    y = np.unique(Y,axis=1).reshape((np.size(Y,0),))
    
    # will calculate the differences in an axm and axn matrix between vx and x; and vy and y respectively
    xDiff = -np.full((nv,np.size(x)),x) + v[0,:].reshape((nv,1))
    yDiff = -np.full((nv,np.size(y)),y) + v[1,:].reshape((nv,1))
    
    xUnder = (np.max(xDiff[xDiff < 0],axis=0) + v[0,:]).astype(int)
    xAbove = (np.min(xDiff[xDiff > 0],axis=0) + v[0,:]).astype(int)
    print('xUnder shape: {}'.format(xUnder.shape))
    
    yUnder = (np.max(yDiff[yDiff < 0],axis=0) + v[1,:]).astype(int)
    yAbove = (np.min(yDiff[yDiff > 0],axis=0) + v[1,:]).astype(int)
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
    
    HexD = ( weights[0,:]*D_xUyU + weights[1,:]*D_xUyA + weights[2,:]*D_xAyU + weights[3,:]*D_xAyA ) / np.sum(weights)
    
    return HexD


      

def generateHexBase(numHexInRadius, v, HexD, f, baseVal=-20):
    '''
    This function will be to create the faces indices for the base of the hexagon.
    It will use the outer layer of the hexagon to create flat edges at the sides.
    All of the vertices on the bottom will then have triangles with the inner-most point, located directly under the centre.    
    '''
    nv = np.size(HexD) # the original number of vertices
    
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
        baseFaces[j,:] = np.array([ j , (j+1)%layerSize(numHexInRadius) , layerSize(numHexInRadius)])
        
    baseFaces = baseFaces + nv # offset the indices by the number of original vertices
    
    #### NEED TO COMBINE baseFaces AND f
    f = np.vstack((f,edgeFaces,baseFaces)).T
    
    return v,HexD,f

      
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