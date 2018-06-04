# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:58:30 2016

@author: markprosser
"""

def numbPlusItself(x):
    return x + x

def numbToPowerOfItself(x):
    return x ** x

def numbPlusItself(x):
    return x + x


def tableToVector(MAT,ii,jj,n):
    #MAT = The Box Mat you want to make into a column
    #ii = the i coordinate of the top left most value
    #jj = the j coordinate of the top left most value
    #n = how many values across e.g. 12
    import numpy as np
    offset=((ii-1)*12)+1 #otherwise gap may appear at beginning of output mat
    a = np.empty(((MAT.shape[0]-ii)*n,1)) 
    for j in range(ii,MAT.shape[0]): #down 
        for i in range(jj,jj+n): #across 
            b=((j*12)-12+i)-offset
            try:        
                a[b,0]=MAT[j,i]
            except ValueError:
                a[b,0]=np.nan
    return a

def runningMean(VEC,rm):
    #VEC - column vector you want to operate on
    #rm - e.g. 3 or 5 - no even numbers
    import numpy as np
    side = int((rm-1)/2)
    z = np.empty((VEC.shape[0],1))
    z[:]=np.NAN
    for i in range(side,VEC.shape[0]-side):
        z[i,0]=np.average(VEC[i-side:i+side+1,0])
    return z

def clear_all(glob_var=globals()):
    """Clears all the variables from the workspace of the spyder application."""
   # gl = globals().copy()
    gl=glob_var.copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(glob_var[var]): continue
        if 'module' in str(glob_var[var]): continue

        del glob_var[var]


def show_plot(figure_id=None):
    import matplotlib.pyplot as plt
    if figure_id is None:
        fig = plt.gcf()
    else:
        # do this even if figure_id == 0
        fig = plt.figure(num=figure_id)

    plt.show()
    plt.pause(1e-9)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()

if __name__ == "__main__":
    clear_all()

