# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:24:42 2022

@author: Andrew

File to deal with the coordinate transforms required for dealing with the hexagons and locations
"""
import numpy as np

class CoordinateTransform:
    '''
    This class will be able to handle transformations for geocoordinates to image coordinate space.
    It will be initiated with either a reference point and a scale or two reference points.
    I will assume that we are working on an f-plane, where the local geometry is flat. 
    
    Between Irun and Santiago De Compostella, this assumption results in a change of distances of ~0.4%, which I deem to be an acceptable distortion.
    '''
    def __init__(self,ref1Img=[0,0],ref1Coords=[0,0],scale=None,ref2Img=None,ref2Coords=None,metadata=None):
        '''
        INPUTS:
            ref1Img: 2x1 array, pixel position of known refernce point in image
            ref1Coords: 2x1 array, geo-coordinates (long,lat) for the reference location
            
            scale: 2x1 array, first element is the x-scale (angle per pixel) and second element is y-scale (angle per pixel)
        
            ref2Img: 2x1 array, elements are position coordinates
            ref2Coords: 2x1 array, elements corresponding to (long,lat) associated with position in ref2Img
        
            metadata: 3x1 array, the metadata output from loadData()
        '''
        # convert input references to numpy vectors
        ref1Img = np.array(ref1Img).reshape((2,1))
        ref1Coords = np.array(ref1Coords).reshape((2,1))
        
        #NOTE, the cellsize parameter in the SRTM data isn't the angular extent in one direction of the cell.
        if metadata: # if the metadata from the SRTM dataset is provided
            angle = metadata[2] # angle is given in degrees for the 90m resolution dataset
            self.scale = np.array([ angle , angle ]).reshape((2,1))# reshapes as a collumn vector
            self.originCoords = np.array([ metadata[0] , metadata[1] ]).reshape((2,1))# reshapes as a collumn vector
            return
        
        
        if scale:
            # the reference point for the Transform will be the origin in image space
            # Need to determine the geo-coordinates at the image origin
            scale = np.array(scale).reshape(2,1) # transform into np.array vector, multiplication is element-wise
            
        elif ref2Img and ref2Coords:
            ref2Img = np.array(ref2Img).reshape((2,1))
            ref2Coords = np.array(ref2Coords).reshape(2,1)
            
            deltaImg = ref2Img - ref1Img
            deltaCoords = ref2Coords - ref1Coords
            
            scale = deltaCoords/deltaImg #in units of angle/pixel by definition
        else:
            print('No second reference or scale given as keyword argument')
            return None
        
        self.scale = scale # reshape as collumn vectors for use in multiple coordinate conversions
        originCoords = ref1Coords - (self.scale * ref1Img)
        self.originCoords = np.reshape(originCoords,(2,1))
        
    
    def coords2Img(self,coords=None):
        '''function to convert from geo-coordinates to image coordinates
        INPUTS:
            coords: 2x1 array, (long,lat) of position to convert
        '''
        if coords is not None:
            coords = np.array(coords)
            
            deltaCoords = coords - self.originCoords
            deltaImg = deltaCoords/self.scale
            return deltaImg
        else:
            print('No geo-coordinates given.')
            return None
        
    def img2Coords(self,img=None):
        '''function to convert from image coordinates to real coordinates
        INPUTS:
            img: 2x1 array, image coordinates for point in image.
        '''
        if img is not None:
            img = np.array(img)
            
            coords = self.originCoords + img*self.scale
            return coords
        else:
            print('No image coordinates given.')
            return None
        
            

