import os
import sfepy
import meshio
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle

#script for calling FEM solver
"""
print("Changing to master directory:")
os.chdir("/Users/oluwadamilolafasina/Documents/sfepy-master")
print("Printing working directory:")
os.system("pwd")
print("Running simple script:")
os.system("sfepy-run simple /Users/oluwadamilolafasina/Documents/sfepy-master/sfepy/examples/diffusion/poisson_neumann.py")

#Analyzing output

sol = meshio.read('/Users/oluwadamilolafasina/Documents/sfepy-master/cross-51-0.34.vtk')

#Debug comments

#grad = meshio.read('2_4_2_refined_grad.vtk')
#print(mesh.u)
#print(mesh.point_data)
#print(type(mesh))
#print(type(mesh.point_data))
#print(mesh)
#print(mesh.point_data)
#print(mesh.cell_data)
#print(mesh.__dict__)

#Print out to check veracity of solution
print(sol)
print(sol.point_data)
print(sol.point_data["t"])
print("Type:",type(sol.point_data["t"]))
print("Shape",np.shape(sol.point_data["t"]))
print("DUMBO!",np.shape(sol.points))

data = sol.point_data["t"] 
coord = sol.points[:,:2]

print("Sanity check data:",np.shape(data))
print("Sanity check coordinates:",np.shape(coord))

print(np.mean(data))

np.min(sol.point_data["node_groups"])
"""
#Generating poisson_neumann data
data = []
coord = []
flux = []
mnsol = []

for i in range(1000):

    os.system( "sfepy-run simple /Users/oluwadamilolafasina/Documents/sfepy-master/sfepy/examples/diffusion/poisson_neumann_edit.py -d prm=" + str(i) + " -o /Users/oluwadamilolafasina/Inverse_GNN/FEM_output/pmesh_"+str(i))
    sol = meshio.read("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/" + "pmesh_" + str(i) + ".vtk")
    data.append(sol.point_data['t'])
    coord.append(sol.points[:,:2])
    flux.append(i)
    mnsol.append(np.mean(sol.point_data['t']))

    #Visualization
    #os.system("~/Documents/sfepy-master/resview.py pmesh_" + str(i) + ".vtk -o /Users/oluwadamilolafasina/Inverse_GNN/Figures/flux" + str(i) + ".png --off-screen")

print("Checking Shape:")
print(coord[1].shape)
print(data[1].shape)

print("Checking solution data:")

plt.title("Mean solution vs flux")
plt.plot(flux,mnsol)
plt.ylabel("Mean Solution")
plt.xlabel("Flux")

#Savign data

with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/solution_data","wb") as fp:
    pickle.dump(data,fp)

with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/coordinate_data","wb") as fb:
    pickle.dump(coord,fb)

with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/input_flux","wb") as fc:
    pickle.dump(flux,fc)