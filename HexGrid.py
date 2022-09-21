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
        
#HG = HexGrid()


def walkingAlgorithm(targetHexagon,numHexInRadius=2):
    '''
    Want to write an algorithm that will generate Hexagons using the hexagon object in a pattern that will allow them to walk out from a central one.
    As they do this, the hexagon's vertices and centre will be appended to the grid coordinates.
    We will need to cull this coordinate list for the duplicates (2) everytime a new hexagon is made.
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
            metadata m = 'i': initial hexagon, spawn Hexagons with m=c0 and c3 (right or left, crawl). Also with m=iw2 and iw5 (walking up and walking down).
            metadata m = c0: spawn one on side 1, m=c1. Also spawn in w2,w5
            metadata m = c1: spawn one on side 0, m=c0. Also w2, w5.
            metadata m = c3: spawn one c4, w2, w5
            metadata m = c4: spawn one c3, w2, w5
            metadata m = w2: spawn another w2
            metadata m = w5: spawn another w5
            m=iw2: spawn iw2
            m=iw5: spawn iw5
            
            Keeping track of these and itterating the algorithm through them will cause hexagons to walk out from the central one until they are culled by step 3.
            This will fill the hexagon with the smaller hexagons and allow the tracking of vertices, et al.
    '''
    '''
    This function will handle the control flow for the algorithm, and the actual steps 1-3 can be done by a seperate function.
    The order of priority will go as follows: 
        1: i
        2: iw2, iw5     ** Priority 1
        3: w2, w5       ** Priority 1
        4: c0, c3       ** Priority 2
        5: c1, c4       ** Priority 2
        
    By doing this, we can ensure the same algorithms are used for each type of triangle, including with regards to repeated vertices.
    '''
    # initialisation based on targetHexagon and desired number
    scale = targetHexagon.scale / numHexInRadius
    rot = targetHexagon.rotation
    centre = targetHexagon.centre
    m = 'i,1'
    
    hexPriority1 = [Hex.Hexagon(scale,rot,centre,m)]
    hexPriority2 = []
    
    faces = []
    vertices = hexPriority1[0].vertices
    
    z = -1
    try:
        while hexPriority1 or hexPriority2:
            z += 1
            print('Hexagon {}:'.format(z))
            # need to start by popping the current hexagon from the priority brackets. They will either be the first element of hexPriority1, hexPriority2.
            if hexPriority1:
                currentH = hexPriority1.pop(0)
            else: 
                currentH = hexPriority2.pop(0)
                
            # having got the current hexagon, we now extract the faces and vertices
            vertices, faces, outMask = walkingAlgorithm_triangles(currentH, targetH, vertices, faces)
            
            # going for a slightly less efficient method, but it should be comprehensive
            # we now consider the type of hexagon we are looking at. Dependant on type, we create new hexagons and add them to the appropriate bracket.
            m = currentH.meta.split(',')
            m[1] = int(m[1])
            if m[1]:
                if m[0] == 'i':
                    hexPriority1.append(currentH.createAdjacentHexagon( 2,'iw2,'+str(numHexInRadius-1)))
                    hexPriority1.append(currentH.createAdjacentHexagon(5,'iw5,'+str(numHexInRadius-1)))
                    hexPriority2.append(currentH.createAdjacentHexagon(0,'c0,'+str(numHexInRadius-1)))
                    hexPriority2.append(currentH.createAdjacentHexagon(3,'c3,'+str(numHexInRadius-1)))
                elif m[0] == 'iw2':
                    hexPriority1.append(currentH.createAdjacentHexagon(2,'iw2,'+str(m[1]-1) ))
                elif m[0] == 'iw5':
                    hexPriority1.append(currentH.createAdjacentHexagon(5,'iw5,'+str(m[1]-1)))
                elif m[0] == 'w2':
                    hexPriority1.append(currentH.createAdjacentHexagon(2,'w2,'+str(m[1]-1)))
                elif m[0] == 'w5':
                    hexPriority1.append(currentH.createAdjacentHexagon(5,'w5,'+str(m[1]-1)))
                elif m[0] == 'c0':
                    hexPriority1.append(currentH.createAdjacentHexagon(2,'w2,'+str(numHexInRadius-1)))
                    hexPriority1.append(currentH.createAdjacentHexagon(5,'w5,'+str(numHexInRadius-1)))
                    hexPriority2.append(currentH.createAdjacentHexagon(1,'c1,'+str(m[1]-1)))
                elif m[0] == 'c1':
                    hexPriority1.append(currentH.createAdjacentHexagon(2,'w2,'+str(numHexInRadius-1)))
                    hexPriority1.append(currentH.createAdjacentHexagon(5,'w5,'+str(numHexInRadius-1)))
                    hexPriority2.append(currentH.createAdjacentHexagon(0,'c0,'+str(m[1]-1)))
                elif m[0] == 'c3':
                    hexPriority1.append(currentH.createAdjacentHexagon(2,'w2,'+str(numHexInRadius-1)))
                    hexPriority1.append(currentH.createAdjacentHexagon(5,'w5,'+str(numHexInRadius-1)))
                    hexPriority2.append(currentH.createAdjacentHexagon(4,'c4,'+str(m[1]-1)))
                elif m[0] == 'c4':
                    hexPriority1.append(currentH.createAdjacentHexagon(2,'w2,'+str(numHexInRadius-1)))
                    hexPriority1.append(currentH.createAdjacentHexagon(5,'w5,'+str(numHexInRadius-1)))
                    hexPriority2.append(currentH.createAdjacentHexagon(3,'c3,'+str(m[1]-1)))
                else:
                    print('Somethings gone terribly wrong...')
            else:
                print('end of branch')
    except Exception as error:
        print('exception raised')
        return vertices, faces, error
    return vertices,faces,None
            


def walkingAlgorithm_triangles(H,targetH,vertices,faces):
    '''
    For a provided hexagon, and the target hexagon, place the vertices in the matrix and generate the triangular faces.
    If the vertices of the hexagon lie outside the target hexagon (all 6), then we will return a mask value that states to not generate any new hexagons from this one.
    '''
    outMask = 0
    verticesInside = targetH.insideHexagon_coords(H.vertices)
    if np.sum(verticesInside) > 2: # should always be 2,4,6 or 0; 2 or 0 means no new triangles inside targetHex and center must implicitly be outside target hex
        
        numOriginalVertices = np.size(vertices,1)
        
        vertices = np.hstack((vertices,H.vertices[:,verticesInside],H.centre)) # appends the hexagons vertices and centre to the vertices matrix
        __,idx,counts = np.unique(vertices,axis=1,return_counts=True,return_index=True)
        
        vertices = vertices[:,np.sort(idx)] # np.unique reorders list, so undoing the reordering
        # need to reorder counts properly as well.
        
        '''
        
        counts = counts[np.argsort(idx)]
        
        print('idx: {}'.format(idx))
        print('counts: {}'.format(counts))
        
        numRepeatedVertices = np.sum(counts == 2)
        print('numRepeatedVertices: {}'.format(numRepeatedVertices))
            
        repeatedVertices = vertices[:, np.where(counts == 2)].squeeze().reshape((2,numRepeatedVertices))
        
        
        # standard pattern for vertices being faces is: (center given as 6)
        # [0,1,6],[1,2,6],[2,3,6],[3,4,6],[4,5,6],[5,0,6]   + numOriginalVertices - numRepeatedVertices
        # However, if any of vertex 0-5 are repeated, need to get new index in vertices
        # also need a mask for if any of the vertices are outside the hexagon - these should normally occur in pairs.
        if np.sum(verticesInside) == 6:
            print('All vertices inside target.')
            vertexIndices = np.array([0,1,2,3,4,5]) + numOriginalVertices - numRepeatedVertices
            print('vertexIndices: pre-change: {}'.format(vertexIndices))
            centreIndex = 6+numOriginalVertices-numRepeatedVertices
            for j in range(numRepeatedVertices):
                print('H.vertices: {}'.format(H.vertices))
                print(repeatedVertices)
                print('repeatedVertices[:,{}]: {}'.format(j,repeatedVertices[:,j]))
                print('repVertices: {}'.format(  (H.vertices == repeatedVertices[:,j].reshape((2,1))  )))
                
                i = np.where( (H.vertices == repeatedVertices[:,j].reshape((2,1))  ).all(axis=0) )[0][0]
                
                vertexIndices[ i ] = np.where((vertices == repeatedVertices[:,j].reshape((2,1))  ).all(axis=0))[0][0] # stupid indexing to get around the tuple output
                print('vertexIndices[{}]: {}'.format(j,vertexIndices[i]))
            
            print('H.vertices: {}'.format(H.vertices))
            print('repeatedVertices: {}'.format(repeatedVertices))
            print('vertexIndices: {}'.format(vertexIndices))
            
            for t in range(6):
                faces.append( [ vertexIndices[t%6], vertexIndices[(t+1)%6], centreIndex ] )
                
        else: # in this case, we have some of the vertices outside the target hexagon.
            print('{} vertices outside target'.format(6 - np.sum(verticesInside)))
            # create the centre index, isolate the vertices inside the target hexagon    
            centreIndex = np.sum(verticesInside) + numOriginalVertices - numRepeatedVertices
            
            # each new face will be:   [*(indices of H.vertices)[ *newFaces[i] ], centreIndex]
            newFaces = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
            # ONLY IF WE HAVE MISSING VERTICES
            t = ( np.where( np.logical_not(verticesInside) )[0][1] - 2 )%6 # the additional triangle we have to miss out
            facesInside = verticesInside.copy()
            facesInside[t] = False
            
            newFaces = newFaces[facesInside,:]
            print(newFaces)
            
            # now need to assign vertex indices to each vertex within the target hexagon, accounting for repeated vertices
            vertexIndices = -np.ones((6,)) # indices for the outside vertices will remain -1
            n = 0# this will track the number of repeated vertices we've enumerated over
            for i,vert in enumerate(H.vertices.transpose()):
                if verticesInside[i]:
                    vert = vert.reshape((2,1))
                    if np.sum(  (repeatedVertices == vert).all(axis=0)  ) > 0:
                        vertexIndices[i] = np.where( (vertices == vert).all(axis=0) )[0][0]
                        n += 1
                    else:
                        vertexIndices[i] = i - n + numOriginalVertices
                        
            for t in newFaces:
                print('newFace: {}'.format([ *vertexIndices[t], centreIndex  ]))
                faces.append([ *vertexIndices[t], centreIndex  ])
                
    '''
    else: 
        outMask = 1 # if more than 3 vertices are outside the target hexagon, then no triangles can be formed and we have to stop creating hexagons    
        print('Hexagon outside target')
    print('-------------------------------------------------------')
    return vertices,faces,outMask



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
fig, ax = plt.subplots(figsize=(20,20))
pS = 3
plt.hold = True
ax.scatter(HG.coordStack[0,:],HG.coordStack[1,:])
for i, txt in enumerate(HG.labels):
    ax.annotate(txt,(HG.coordStack[0,i],HG.coordStack[1,i]))
'''
'''
o = 1

targetH = Hex.Hexagon(o,0,[0,0])
numInHex = o
v,f,e = walkingAlgorithm(targetH,numInHex)

fig,ax = plt.subplots(figsize=(20,20))
plt.hold=True

ax.scatter(v[0,:],v[1,:])

#rn.shuffle(f)
for i,face in enumerate(f):
    #print('face {}: {}'.format(i,face))
    a,b,c = [*face]
    a = int(a)
    b = int(b)
    c = int(c)
    
    ax.plot(v[0,[a,b]],v[1,[a,b]])
    ax.plot(v[0,[b,c]],v[1,[b,c]])
    ax.plot(v[0,[c,a]],v[1,[c,a]])

raise e
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
