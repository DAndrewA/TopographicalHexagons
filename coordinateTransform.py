# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:24:42 2022

@author: Andrew

File to deal with the coordinate transforms required for dealing with the hexagons and locations
"""
import numpy as np

class coordinateTransform:
    '''
    This class will be able to handle transformations for geocoordinates to image coordinate space.
    It will be initiated with either a reference point and a scale or two reference points.
    I will assume that we are working on an f-plane, where the local geometry is flat. 
    
    Between Irun and Santiago De Compostella, this assumption results in a change of distances of ~0.4%, which I deem to be an acceptable distortion.
    '''
    def __init__(self,ref1Img=[0,0],ref1Coords=[0,0],scale=None,ref2Img=None,ref2Coords=None):
        '''
        INPUTS:
            ref1Img: 2x1 array, pixel position of known refernce point in image
            ref1Coords: 2x1 array, geo-coordinates (lat,long) for the reference location
            
            scale: 2x1 array, first element is the x-scale (angle per pixel) and second element is y-scale (angle per pixel)
        
            ref2Img: 2x1 array, elements are position coordinates
            ref2Coords: 2x1 array, elements corresponding to (lat.,long.) associated with position in ref2Img
        '''
        # convert input references to numpy vectors
        ref1Img = np.array(ref1Img)
        ref1Coords = np.array(ref1Coords)
        
        if scale:
            # the reference point for the Transform will be the origin in image space
            # Need to determine the geo-coordinates at the image origin
            self.scale = np.array(scale) # transform into np.array vector, multiplication is element-wise
            
        elif ref2Img and ref2Coords:
            ref2Img = np.array(ref2Img)
            ref2Coords = np.array(ref2Coords)
            
            deltaImg = ref2Img - ref1Img
            deltaCoords = ref2Coords - ref1Coords
            
            self.scale = deltaCoords/deltaImg #in units of angle/pixel by definition
            
        else:
            print('No second reference or scale given as keyword argument')
            return None
        
        self.originCoords = ref1Coords - (self.scale * ref1Img)
        
    
    def coords2Img(self,coords=None):
        '''function to convert from geo-coordinates to image coordinates
        INPUTS:
            coords: 2x1 array, (lat,long) of position to convert
        '''
        if coords:
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
        if img:
            img = np.array(img)
            
            coords = self.originCoords + img*self.scale
            return coords
        else:
            print('No image coordinates given.')
            return None
        
            

