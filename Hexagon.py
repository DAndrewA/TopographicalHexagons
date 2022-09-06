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
    
    # creation of the unit vectors 
    a0 = np.array([ np.sqrt(3)/2 , -1/2 ])
    a1 = np.array([ np.sqrt(3)/2 , 1/2 ])
    a2 = np.array([ 0 , 1 ])
    a3 = -a0
    a4 = -a1
    a5 = -a2
    unitVertices = np.stack((a0,a1,a2,a3,a4,a5),1) # stack along the second-matrix-axis direction. This allows for rotation matrices later
    
    def __init__(self,scale=1,rotation=0,centre=[0,0]):
        '''
        INPUTS:
            scale: float, the size scale of the hexagon (in image pixel units [image coordinates])
            rotation: float, the rotation of the hexagon relative to the description above [degrees]
            centre: 2x1 array, centre of the hexagon in coordinate-space
        '''
        self.scale = scale
        self.rotation = rotation
        self.centre = np.array(centre)
        
        # generate the vertices for the hexagon based upon the given information
        self.vertices = self.placeHexagon()
       
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
        '''
        return np.reshape(self.centre,(2,1)) + np.matmul(self.rotM(),self.unitVertices)*self.scale
        
    
    def createAdjacentHexagon(self,face=0):
        '''
        creates a new hexagon object adjacent to the original with the same scaling and orientation.
        The centre of the hexagon is determined by the chosen face.
        '''
        
        vectorPair = [face,(face+1)%6] # modulo to acocunt for face 5 having vertex 5 pairing with 0
        centreDisplacement = self.unitVertices[:,vectorPair[0]] + self.unitVertices[:,vectorPair[1]]
        
        newCentre = self.centre + np.matmul(self.rotM(), centreDisplacement)*self.scale
        
        return Hexagon(self.scale,self.rotation,newCentre)

'''
# testing that the creation of the same hexagon via different routes does give the same centres
H = Hexagon()
H1 = H.createAdjacentHexagon()
H2 = H.createAdjacentHexagon(1)
H3 = H2.createAdjacentHexagon(5)
'''