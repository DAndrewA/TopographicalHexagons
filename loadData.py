# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 15:49:59 2022

@author: Andrew

Will load in the datasets into a desired grid-like format.
"""

import numpy as np
import matplotlib.pyplot as plt
import primefac as primefac



'''
This is the function to load in the .asc file format provided by the SRTM dataset.
'''
def load_asc_format(direc,filename,seaVal=0):
    f = open(direc + filename,'r')
    try: # wrapped in a try statement so we can always close the file    
        # The initial steps are to extract the metadata from the file
        while True: # This loop will run until we've found the line that starts the data
            line = f.readline()
            if line == '\n': # Ignore any blank lines in the file
                pass
            elif 'ncols' in line:
                ncols = int(line.split()[1]) # select the second element in the line
            elif 'nrows' in line:
                nrows = int(line.split()[1])
            elif 'xllcorner' in line:
                xll = float(line.split()[1])
            elif 'yllcorner' in line:
                yll = float(line.split()[1])
            elif 'cellsize' in line:
                cellsize = float(line.split()[1])
            elif 'NODATA' in line:
                NDval = int(line.split()[1])
            else:
                break # if none of the variables are being entered, we have reached the data.
        
        metadata = [xll,yll,cellsize] # long,lat,scale
        #We have now reached the first line with data in it.
        # initialising numpy arrays
        dataArr = np.zeros([ncols,nrows])
        
        # now implement a do-while loop, as we know we can run the first itteration
        lineIndex = 0
        while True:
            lineData = [int(n) for n in line.split()]
            dataArr[nrows-lineIndex-1,:] = lineData #data in inverse-order to get increasing y as going North
            
            line = f.readline()
            lineIndex += 1
            if not line:
                break
        
        f.close()
        
        # as we don't have bathymetry data, I will set all NODATA areas to the seaVal
        dataArr[dataArr == NDval] = seaVal
        return dataArr,metadata
        
    except Exception as err:
        f.close()
        raise(err)
    
  

'''
This function is to downsample a matrix by the minimum factor whilst maintaining the original aspect ratio.
Significant improvements have been made to speed by utilising the meshgrid method rather than nested for loops to get the spaced values.
'''
def downsample_minimum(D):
    # extract the prime factors from the matrix dimensions 
    x = np.size(D,1)
    y = np.size(D,0)
    
    pfacX = primefac.primefac(x)
    pfacY = primefac.primefac(y)
    lcd = x*y
    for f in pfacX:
        for g in pfacY:
            if f == g and f < lcd:
                lcd = f
    if lcd != x*y: # if the two numbers are not coprime
        x = range(0,int(x/lcd))#*lcd
        y = range(0,int(y/lcd))#*lcd
        
        X,Y = np.meshgrid(x,y)
        newD = D[Y*lcd,X*lcd]
    
        print('newD created, size is ({},{})'.format(np.size(newD,0),np.size(newD,1)))
        return newD,lcd
    else:
        print('Dimensions are coprime. Cannot downsample and maintain aspect ratio.')
        return D,1
    
    
    
def load_coordinate_list(direc,filename,commentChar='#'):
    '''
    A function for loading in the self-designated coordinate list file I'm creating.
    The file consists of comments, lines beginining with '#', blank lines (to be ignored)
    and pairs of lines, one starting with a '.' that denotes a place name, and the next
    line being a pair of (lat,long) coordinates.
    '''
    names = []
    coords = []
    f = open(direc + filename,'r')
    try:
        while True: # will go through the whole file, line by line
            line = f.readline()
            # do nothing if the line is blank or a comment
            if line == '\n':
                pass
                #print('blankline - pass')
            elif not line: # if none of the above conditions are met we should be at the end of the file
                #print('breakline: {}'.format(line))
                break
            elif line[0] == commentChar: 
                pass
                #print('commentline - pass')
            # otherwise, we should be at a line begining with a '.'
            elif line[0] == '.':
                # get the current name and coords, converting into the order (long,lat)
                cName = line[1:]
                line = f.readline().split(',')
                cCoords = [ float(line[1]) , float(line[0]) ]
                #print('cName: {}'.format(cName))
                #print('cCoords: {}'.format(cCoords))
                names.append(cName)
                coords.append(cCoords)
            
            
        # convert the coordinates into horizontally stacked collumn vectors
        coords = np.array(coords)
        coords = np.transpose(coords)
        return coords,names
        
    except Exception as err:
        f.close()
        print('FILE CLOSED')
        raise(err)
        