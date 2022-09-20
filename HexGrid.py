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

class HexGrid:
    '''
    This is the HexGrid class. It will contain the unit lattice vectors and offset vector betweeen the two sublattices.
    These can then be scaled and rotated, and added together linearly to produce a hexagonal grid.
    '''
    
    u1 = np.array([ 1 , 0 ]).reshape((2,1))
    u2 = np.array([ 0.5 , np.sqrt(3)/2 ]).reshape((2,1))
    
    def __init__(self):
        print('creating hex grid')
        a = range(-5,5)
        b = range(-5,5)
        
        self.coordStack = np.array([0,0]).reshape((2,1))
        self.labels = ['']
        
        
        w = lambda x,y: np.abs(x)+np.abs(y)
        g = lambda x,y: np.abs(x)
        # arrange the grid...
        for i in a:
            for j in b:
                
                
                label = '{},{},{}'.format((i,j),w(i,j),g(i,j))
                coord = i*self.u1 + j*self.u2
                print(label)
                print(coord)
                print('---------------')
                
                self.coordStack = np.hstack((self.coordStack,coord))
                self.labels.append(label)
        print('done')
        
HG = HexGrid()


def walkingAlgorithm():
    print('beeop boop')
    '''
    Want to write an algorithm that will generate Hexagons using the hexagon object in a pattern that will allow them to walk out from a central one.
    As they do this, the hexagon's vertices and centre will be appended to the grid coordinates.
    We willl need to cull this coordinate list for the duplicates (2) everytime a new hexagon is made.
    We can then take the heightmap values at these points, and determine some way of assigning edges between vertices.
    '''
    
    '''
    numpy.unique(s,axis=1) will return the unique coordinates. We can also get the indices of the two repeated coordinates, and then use a secondary matrix to track triangles
    i.e. vertices = [centre, v0, v1, ...]
    triangles = [0,1,2],
    '''
    
    '''
    step 0: Initialise the first hexagon, with the correct centre, rotation.
            Will also initialise algorithm with desired radius in hexagons, and in coordinates.
            This will allow it to create the right size of hexagon for filling the grid.
            i.e. radius of 7 to fill a 500pixel radius hexagon => a = 500/7
            
    step 1: Add vertices into vertices/coords matrix, remove duplicates.
    step 2: From a hexagon, there will be 6 triangles, all sharing the centre.
    step 3: Determine if any of the vertices are outside the desired hexagon. If so, halt for this hexagon.
    step 4: Depending on hexagon type, and if inside master hexagon, generate new hexagons, that walk away from the centre.
            This will be the walking algorithm, and may require metadata to be stored in the Hexagon object.
            metadata m = 'i': initial hexagon, spawn Hexagons with m=c0 and c3 (right or left, crawl). Also with m=w2 and w5 (walking up and walking down).
            metadata m = c0: spawn one on side 1, m=c1. Spawn one on side 4, m=c4. Also spawn in w2,w5
            metadata m = c1: spawn one on side 0, m=c0. Also w2, w5.
            metadata m = c3: spawn one c4, w2, w5
            metadata m = c4: spawn one c3, w2, w5
            metadata m = w2: spawn another w2
            metadata m = w5: spawn another w5
            
            Keeping track of these and itterating the algorithm through them will cause hexagons to walk out from the central one until they are culled by step 3.
            This will fill the hexagon with the smaller hexagons and allow the tracking of vertices, et al.
    '''


fig, ax = plt.subplots(figsize=(20,20))
pS = 3
plt.hold = True
ax.scatter(HG.coordStack[0,:],HG.coordStack[1,:])
for i, txt in enumerate(HG.labels):
    ax.annotate(txt,(HG.coordStack[0,i],HG.coordStack[1,i]))


