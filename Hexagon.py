# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 20:17:04 2022

@author: Andrew

File containing code for the creation and manipulation of hexagons in vector form.
The vectors can be treated as coordinates, that can then be used with the coordinate transforms to place them on the map.

The hexagons will oriented with a corner in the y-unit vector from the hexagon's origin, with the x-vector bisecting an edge of the hexagon at right-angles.
Faces for the hexagon will be labeled going anti-clockwise, 0-indexed, starting from the face in the x-direction 
"""

import numpy as np

class Hexagon:
    '''
    The hexagon class. This will contain the unit vectors for the hexagon.
    It will also contain the scaling of the hexagon, a (the length of the origin to a vertex).
    It will have the origin in coordinate space and the local rotation of the hexagon.
    
    It will also have methods to place the hexagon in coordinate space given these values, as well as create a new adjacent hexagon off a given face.
    '''
    
    # creation of the unit vectors, reshaped to be collumn vectors
    a0 = np.array([ np.sqrt(3)/2 , -1/2 ]).reshape((2,1))
    a1 = np.array([ np.sqrt(3)/2 , 1/2 ]).reshape((2,1))
    a2 = np.array([ 0 , 1 ]).reshape((2,1))
    a3 = -a0
    a4 = -a1
    a5 = -a2
    unitVertices = np.stack((a0,a1,a2,a3,a4,a5),1).reshape((2,6)) # stack along the second-matrix-axis direction. This allows for rotation matrices later
    
    # creation of unit normals
    n0 = np.array([ 1 , 0 ]).reshape((2,1))
    n1 = np.array([ 1/2 , np.sqrt(3)/2 ]).reshape((2,1))
    n2 = np.array([ -1/2 , np.sqrt(3)/2 ]).reshape((2,1))
    unitNormals = np.stack((n0,n1,n2),1).reshape((2,3))
    
    def __init__(self,scale=1,rotation=0,centre=np.array([0,0]).reshape((2,1)),metadata=None):
        '''
        INPUTS:
            scale: float, the size scale of the hexagon (in image pixel units [image coordinates])
            rotation: float, the rotation of the hexagon relative to the description above [degrees]
            centre: 2x1 array, centre of the hexagon in coordinate-space
        '''
        self.scale = scale
        self.rotation = rotation
        try:
            self.centre = np.array(centre).reshape((2,1))
        except Exception as err:
            print('Exception thrown in reshaping centre (line 50?)')
            print(centre)
            print(np.size(centre))
            print(np.shape(centre))
            raise err
        # generate the vertices for the hexagon based upon the given information
        self.vertices = self.placeHexagon()
        self.meta = metadata
       
    def rotM(self):
        '''
        Function to generate the rotation matrix for the hexagon based on the given angle
        '''
        theta = self.rotation * np.pi/180 # convert to radians
        rotM = np.array([[ np.cos(theta) , -np.sin(theta) ],
                         [ np.sin(theta) , np.cos(theta) ]])
        return rotM
        
       
    def placeHexagon(self):
        '''
        Function to transform the unitVertices into coordinates for the vertices of the hexagon
        Will also rotate and scale the face normals
        '''
        R = self.rotM()
        self.normals = np.matmul(R,self.unitNormals)
        return np.matmul(R,self.unitVertices)*self.scale + self.centre
        
    
    def createAdjacentHexagon(self,face=0,metadata=None):
        '''
        creates a new hexagon object adjacent to the original with the same scaling and orientation.
        The centre of the hexagon is determined by the chosen face.
        '''
        
        vectorPair = [face%6,(face+1)%6] # modulo to acocunt for face 5 having vertex 5 pairing with 0
        centreDisplacement = self.unitVertices[:,vectorPair[0]] + self.unitVertices[:,vectorPair[1]]
        centreDisplacement = centreDisplacement.reshape((2,1))
        
        newCentre = self.centre + np.matmul(self.rotM(), centreDisplacement)*self.scale
        
        return Hexagon(self.scale,self.rotation,newCentre,metadata)

    def insideHexagon_grid(self,X,Y):
        '''
        Function that will return a boolean mask of shape shape(X) (and Y) based on if the points are within the hexagon
        INPUTS:
            X: nxm array, meshgrid of X coordinates
            Y: nxm array, meshgrid of Y coordinates
        '''
        if np.shape(X) != np.shape(Y):
            print('Shapes of X and Y do not match.')
            return None
        
        # initially, recentre the coordinates on the hexagon
        X = X - self.centre[0]
        Y = Y - self.centre[1]
        
        mask = np.zeros(np.shape(X))
        
        for i in [0,1,2]: # for each normal (i)
            #compute the absolute dot product of each coordinate position with the normal vector
            Dot = np.abs(X * self.normals[0,i] + Y * self.normals[1,i])
            # if the dot product is less than sqrt(3)a/2 then its within the hexagon, add one
            mask[ Dot <= (np.sqrt(3)/2 * self.scale) ] += 1
            
        # if the mask value is 3, it is within the hexagon for all normals, renormalise
        mask[mask < 3] = 0
        mask[mask == 3] = 1
        mask = np.array(mask, dtype=bool)
        return mask
    
    def outsideHexagon_grid(self,X,Y):
        '''
        Function that will return a boolean mask which is true for points outside the hexagon.
        '''
        if np.shape(X) != np.shape(Y):
            print('Shapes of X and Y do not match.')
            return None
        
        mask = np.ones(np.shape(X))
        return np.array(mask - self.insideHexagon_grid(X, Y), dtype=bool)
        
    def insideHexagon_coords(self,coords,tol=0.00):
        '''
        Function that should work on coordinate matrices of size 2xn, to return boolean mask of which are within the hexagon.
        INPUTS:
            coords: 2xn matrix of n 2-dimensional coordinates.
        '''
        if np.size(coords,0) != 2:
            print('Coords needs to be 2xn.')
            return None
        
        coords = coords - self.centre
        mask = np.zeros(np.size(coords,1))
        
        for i in [0,1,2]: # for each normal i
            # compute the absolute dot product for each coordinate with the normal vector
            Dot = np.abs( np.sum(  coords * self.normals[:,i].reshape((2,1))  , axis=0))
            # if the dot product is less than sqrt(3)a/2 then its within the hexagon, add one
            mask[ ( Dot <= ( (np.sqrt(3)/2 + tol) * self.scale) ) ] += 1
            
        # if the mask value is 3, it is within the hexagon for all normals, renormalise
        mask[mask < 3] = 0
        mask[mask == 3] = 1
        mask = np.array(mask, dtype=bool)
        return mask
    
    def outsideHexagon_coords(self,coords):
        '''
        Function that will return a boolean mask which is true for points outside the hexagon.
        '''
        if np.size(coords,0) != 2:
            print('Coords needs to be 2xn.')
            return None
        
        mask = np.ones(np.size(coords,1))
        return np.array(mask - self.insideHexagon_coords(coords), dtype=bool)


'''
# testing that the creation of the same hexagon via different routes does give the same centres
H = Hexagon()
H1 = H.createAdjacentHexagon()
H2 = H.createAdjacentHexagon(1)
H3 = H2.createAdjacentHexagon(5)

print(H.insideHexagon_coords(H1.vertices))
print(H.vertices)
print(H1.vertices)
print(H2.vertices)
print(H1.insideHexagon_coords(H2.vertices))
print(H1.outsideHexagon_coords(H2.vertices))
'''
