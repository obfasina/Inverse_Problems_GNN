
# Script for generating synthetic PDE data

import numpy as np
import matplotlib.pyplot as plt

#Start with non-homogenous Poisson equation with quadratic source

#Problem domain

# Geometric domain: x from [0,10], y from [0,5] 
# Eigenvalue domain: n = {0,1,....10}, m = {0,1,...10}
# Source is f(x,y) = -x^2 - y^2: 

# Should have 100 10 x 10 meshes where each mesh correson

#Define problem variables
x = np.arange(0,10)
y = np.arange(0,10)
ev_m = np.arange(0,10)
ev_n = np.arange(0,10)

def X(xi,mi):
    return np.sin( ((mi+1)*np.pi*xi) /10 )

def Y(yi,ni):
    return np.sin( ((ni+1)*np.pi*yi) /10 ) 

def source(xi,yi):
    return -xi**2 -yi**2

def Evalue(xvec,yvec,me,ne):

    #Defining constant
    C = -4/ (len(xvec)*len(yvec)*( ((me + 1)**2*np.pi**2/len(xvec)**2) + ((ne + 1)**2*np.pi**2/len(vec)**2) ))

    #summing over space
    interm = []
    for i in range(len(xvec)):
        for j in range(len(yvec)):
            interm.append(source(xvec[i],yvec[j])*X(xvec[i],me)*Y(yvec[j],ne))

    sumint = np.sum(interm)

    return C*sumint


