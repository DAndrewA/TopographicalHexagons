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
            dataArr[nrows-lineIndex-1,:] = lineData
            
            line = f.readline()
            lineIndex += 1
            if not line:
                break
        
        f.close()
        
        # as we don't have bathymetry data, I will set all NODATA areas to 0
        dataArr[dataArr == NDval] = -10 #NDval
        return dataArr,metadata
        
    except Exception as err:
        f.close()
        raise(err)
    
    


def downsample_minimum(D):
    '''
    This function is to downsample an n*m matrix D by their lowest common denominator for the dimensions
    This is done to maintain the aspect ratio of the downsampled matrix.
    '''
    # extract the prime factors from the 
    x = np.size(D,0)
    y = np.size(D,1)
    
    pfacX = primefac.primefac(x)
    pfacY = primefac.primefac(y)
    lcd = x*y
    for f in pfacX:
        for g in pfacY:
            if f == g and f < lcd:
                lcd = f
    if lcd != x*y: # if the two numbers are not coprime
        newx = int(x/lcd)
        newy = int(y/lcd)
        print([newx,newy])
        newD = np.zeros([newx,newy])
        
        for i in range(0,newx):
            for j in range(0,newy):
                newD[i,j] = D[i*lcd,j*lcd]
    
        return newD
    else:
        print('Dimensions are coprime. Cannot downsample and maintain aspect ratio.')
        return
    
    
    

'''
# example use of code

direc = 'data\\srtm_36_04\\' # use of \\ is to avoid escape character in filepath
filename = 'srtm_36_04.asc'

D,m = load_asc_format(direc,filename)
plt.contour(D,10)
''' 

D_combined = np.ones([6000,1]) * -300
for j in [35,36]:
    direc = 'data\\srtm_' + str(j) + '_04\\' # use of \\ is to avoid escape character in filepath
    filename = 'srtm_' + str(j) + '_04.asc'

    D,m = load_asc_format(direc, filename)
    plt.imshow(D,origin='lower',interpolation='bilinear')
    
    D_combined = np.hstack((D_combined,D))
    
D_combined = D_combined[:,1:]
newD = downsample_minimum(D_combined)
plt.imshow(newD,origin='lower',interpolation='bilinear')
#plt.imshow(D_combined,origin='lower',interpolation='bilinear')

newD = downsample_minimum(newD)
newD = downsample_minimum(newD)

#rescaling the heightmap by a function to exagerate lower elevations more
k = 10
#w = lambda x: np.log(x)*k
#w = lambda x: np.sqrt(x)*k
w = lambda x: x/20
newD[newD > 0] = w(newD[newD > 0])

# 3D surface plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x,y = range(np.size(newD,1)), range(np.size(newD,0))
X,Y = np.meshgrid(x,y)
ax.plot_surface(X,Y,newD,rcount=150,ccount=300)
plt.show()