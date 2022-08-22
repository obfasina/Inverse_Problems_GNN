

#import fenics
from fenics import *
from fenics_adjoint import *
import torch
import torch_fenics
import torch_fenics.numpy_fenics
import numpy as np
from dolfin import *
import pickle
import sympy as sym
import pandas as pd
import os


import argparse
p = argparse.ArgumentParser()
p.add_argument('--new_data',action='store_true',help='Specify whether or not to generate new data')
opt = p.parse_args()


class Wave(torch_fenics.FEniCSModule):
    # Construct variables which can be in the constructor


    def __init__(self):
        # Call super constructor
        super().__init__()

        #Define time settings
        self.T = 30            # final time
        self.num_steps = 30     # number of time steps
        self.dt = self.T / self.num_steps # time step size

        # Create mesh/function space
        nx = ny = 50
        mesh = RectangleMesh(Point(0,0),Point(1,1),nx,ny)


        self.V = FunctionSpace(mesh, 'P', 1)
        mcoord = self.V.tabulate_dof_coordinates()         

        #Save coordinate data
        with open("/home/dami/Inverse_GNN/FEM_output/fenics_coord_wave","wb") as fa:
            pickle.dump(mcoord,fa)
  

    def solve(self, beta):
        
        # Create trial and test functions spaces
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        #Define initial conditions

        #initial field array
        u_n_n = Function(self.V)
        u_n_n.vector()[:] = torch_fenics.numpy_fenics.fenics_to_numpy(beta)[:]
        #u_n_n = interpolate(Constant(0),self.V)

        #Velocity paramter
        #c = Function(self.V)
        #c.vector()[:] = torch_fenics.numpy_fenics.fenics_to_numpy(beta)[:]
        c = interpolate(Constant(0),self.V)
 
        #first order derivative
        u_n = interpolate(Constant(0),self.V)

        #Specifying PDE to be solved
        F = u*v*dx + self.dt*self.dt*c*c * dot(grad(u),grad(v))*dx - (2*u_n - u_n_n)*v*dx
        a, L = lhs(F), rhs(F) 

        #Define boundary condition
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(self.V, Constant(0), boundary)

        # Solve the Wave equation
        u = Function(self.V)
        time_u = np.expand_dims(torch_fenics.numpy_fenics.fenics_to_numpy(u.vector()),axis=1)
        list_timevol = []
        t = 0


        for n in range(self.num_steps):

            # Update current time
            t += self.dt

            # Compute solution
            solve(a == L, u, bc)

            # Update previous solution
            u_n_n.assign(u_n)
            u_n.assign(u)

            #save solution mesh at each time step
            time_u = np.append(time_u,np.expand_dims(torch_fenics.numpy_fenics.fenics_to_numpy(u_n.vector()),axis=1),axis=1)
            
            
        
       
        #save data file
        f = open('fenics_sol.dat','ab')
        np.savetxt(f,time_u[:,1:]) 
        np.loadtxt('fenics_sol.dat') #reloading is required to preserve dimensionality - not sure why           

        # Return the solution
        return u_n

    def input_templates(self):
        # Declare template for fenics to use
        return Constant(np.repeat(0,2601))




# Generate solution data
if opt.new_data == True: #Generate initial solution mesh


    #Define gaussian initial condition or velocity paramater 
    nnodes = 2601 #This number is known from examinning solution mesh a priori in fenics
    incondp = []
    trainsamp = 10000
    for i in range(trainsamp):
        vecrand = np.random.normal(size=nnodes)
        incondp.append(vecrand)
    paramest = torch.tensor(np.array(incondp),dtype=torch.float64)

    #Simulate PDE and save data
    wave = Wave()
    u= wave(paramest).numpy()
    mesh = np.loadtxt('fenics_sol.dat')
    print("paramter estimation",paramest.shape)
    print("solution mesh",mesh.shape)

    with open("/home/dami/Inverse_GNN/FEM_output/fenics_lab_wave","wb") as fc:
        pickle.dump(paramest.numpy(),fc)
    os.system("mv fenics_sol.dat fenics_sol_wave.dat") # Start with new file everytime


if opt.new_data == False: #Generate solution mesh during optimization

    GNNout = np.load("GNN_output.npy").squeeze()
    print("Optimization mesh shape",GNNout.shape)

    nnodes = 2601
    bs = int(GNNout.shape[0]/nnodes) #batch size
   
    findx = 0
    sindx = nnodes
    incondp = []

    for i in range(bs):
        incondp.append(GNNout[findx:sindx])
        findx = findx + nnodes
        sindx = sindx + nnodes

    paramest = torch.tensor(np.array(incondp),dtype=torch.float64)

    #Define PDE class and save data
    wave = Wave()
    u = wave(paramest).numpy()
    os.system("mv fenics_sol.dat fenics_sol_wave_opt.dat") # Start with new file everytime
