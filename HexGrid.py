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
    hexSize = lambda x: 3 * x * (x+1) if x > -1 else -1
    #layerSize = lambda x: 6*x if x else 1
    
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