
# Script for generating synthetic PDE data

import numpy as np
import matplotlib.pyplot as plt

#Start with non-homogenous Poisson equation with quadratic source

#Problem domain

# Geometric domain: x from [0,10], y from [0,5] 
# Eigenvalue domain: n = {0,1,....10}, m = {0,1,...10}
# Source is f(x,y) = -x^2 - y^2: 

# Should have 100 10 x 10 meshes where each mesh correson

#Defining eigenfunctions, source, and eigenvalue
def X(x,m,xvec):
    return np.sin( ((m+1)*np.pi*x) /len(xvec) )

def Y(y,n,yvec):
    return np.sin( ((n+1)*np.pi*y) /len(yvec) ) 

def source(x,y):
    f =(-x**2-y**2)
    return f

def Evalue(x,y,m,n,xvec,yvec):

    #Defining constant
    C = -4/ (len(xvec)*len(yvec)*( ((m + 1)**2*np.pi**2/len(xvec)**2) + ((n + 1)**2*np.pi**2/len(yvec)**2) ))

    #summing over space
    interm = []
    for i in range(len(xvec)):
        for j in range(len(yvec)):

            interm.append(source(x,y)*X(x,m,xvec)*Y(y,n,yvec))

    sumint = np.sum(interm)

    return C*sumint

def Solution(x,y,m,n,xvec,yvec):

    return Evalue(x,y,m,n,xvec,yvec)*X(x,m,xvec)*Y(y,n,yvec)

#Testing synthetic data generation

#Define problem variables
x = np.arange(0,10)
y = np.arange(0,10)
ev_m = np.arange(0,10)
ev_n = np.arange(0,10)


#Data generation

eval_meshlist = []
for a in range(len(ev_m)):
    for b in range(len(ev_n)):

        mesh = []
        for i in range(len(x)):
            for j in range(len(y)):
                mesh.append(Solution(x[i],y[j],ev_m[a],ev_n[b],x,y))

        eval_meshlist.append(mesh)
    
print(len(eval_meshlist[0]))

test = np.array(eval_meshlist[0])
squaremesh = test.reshape((10,10))
plt.imshow(squaremesh)




        



