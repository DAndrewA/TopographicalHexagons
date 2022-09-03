# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 15:49:59 2022

@author: Andrew

Will load in the datasets into a desired grid-like format.
"""

import numpy as np
import matplotlib.pyplot as py


direc = 'data\\srtm_35_04\\' # use of \\ is to avoid escape character in filepath
filename = 'srtm_35_04.asc'

'''
This is the function to load in the .asc file format provided by the SRTM dataset.
'''
def load_asc_format(direc,filename):
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
        
        metadata = [xll,yll,cellsize]
        #We have now reached the first line with data in it.
        # initialising numpy arrays
        dataArr = np.zeros([ncols,nrows])
        
        # now implement a do-while loop, as we know we can run the first itteration
        lineIndex = 0
        while True:
            lineData = [int(n) for n in line.split()]
            dataArr[lineIndex,:] = lineData
            
            line = f.readline()
            lineIndex += 1
            if not line:
                break
        
        f.close()
        
        # as we don't have bathymetry data, I will set all NODATA areas to 0
        dataArr[dataArr == NDval] = 0
        return dataArr,metadata
        
    except Exception as err:
        f.close()
        raise(err)
    
    
'''
# example use of code
D,m = load_asc_format(direc,filename)
py.contour(D,10)
''' 
