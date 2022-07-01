from sympy import solve_triangulated
import torch
from torch import nn
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from synthetic_data import datagen#, sourced
import numpy as np
import matplotlib.pyplot as plt
from Granola_GNN import NodeClassifier
from torch_geometric.loader import DataLoader
import sys
import pickle
import os
from torch.autograd import Variable



#Defining Pytorch geometric graph

def graphgen(gdata,gcoord,glabel):

    "gdata = node solution values; gcoord = node coordinates; glabel = node labels"


    #Defining node features
    nodefeats = torch.tensor(gdata.reshape( (len(gdata),1) ),dtype=torch.float)
    nodelabel = torch.tensor(np.repeat(glabel,len(gdata)),dtype=torch.float)

    # NOTE could swamp method for knn
    #Define edge index
    target = []
    for i in range(len(gdata)): # For each node get 4 nearest neighbors
        target.append(np.argsort( np.sqrt( (gcoord[:,0] - gcoord[i,0])**2 + (gcoord[:,1] - gcoord[i,1])**2 ) )[1:5])
        
    source = []
    for i in range(len(gdata)):
        source.append(np.repeat(i,4))

    source = np.concatenate(source,axis=0)
    target = np.concatenate(target,axis=0)
    edges = torch.tensor(np.concatenate( ( source.reshape((len(source),1)),target.reshape((len(target),1)) ), axis=1 ).T,dtype=torch.long)

    #Define Graph
    graph = Data(x=nodefeats,y=nodelabel,edge_index=edges,pos=gcoord)

    return graph


#Loading data
print("Time to play:")
print("Loading lists of data")
with open("/Users/oluwadamilolafasina/Inverse_GNN/solution_data","rb") as fd:
    data = pickle.load(fd)

with open("/Users/oluwadamilolafasina/Inverse_GNN/coordinate_data","rb") as fe:
    coord = pickle.load(fe)

with open("/Users/oluwadamilolafasina/Inverse_GNN/input_flux","rb") as ff:
    flux = pickle.load(ff)
 
npos = coord[1] #Currently the coordinates are the same for every list in index - need to fix later

"""
#Generating graphs

ind = np.arange(len(data))#Shuffled indices
np.random.shuffle(ind)

pygraphs = []
for i in range(len(data)):
    print(i)
    pygraphs.append(graphgen(data[ind[i]],npos,flux[ind[i]]))


#Saving/Loading graph data
with open("/Users/oluwadamilolafasina/Inverse_GNN/datagraphs","wb") as fg:
    pickle.dump(pygraphs,fg)

"""

with open("/Users/oluwadamilolafasina/Inverse_GNN/datagraphs","rb") as ff:
    pygraphs = pickle.load(ff)


#train/validation/test split (70/20/10) [NOTE: Data has already been shuffled] in data loaders

#Define data loaders
trainloader = DataLoader(pygraphs[:int(round(len(data)*.7))],batch_size=100,shuffle=True)
vadloader = DataLoader(pygraphs[ int(round(len(data)*.7)): int(round(len(data)*.9))],batch_size=100,shuffle=True)
testloader = DataLoader(pygraphs[int(round(len(data)*.9)):],batch_size=100,shuffle=True)



# Script for training GNN 


model = NodeClassifier(num_node_features = 1, hidden_features = 10, nodes = 1) #len(pygraphs[1].x))
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = torch.nn.MSELoss()
model = model.float()

def train():
    loss_all = 0
    c = 0 # Counter for number of batches
    for data in trainloader:

        model.train()   
        optimizer.zero_grad()
        out = model(data.x, data.edge_index,data.pos)

        """
        #Save output and FEM solver
        np.save("GNN_output.npy",out.detach().numpy())
        os.system("python FEM.py")

    
        with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/solution_data_OPT","rb") as fd:
            fdata = pickle.load(fd)

        with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/coordinate_data_OPT","rb") as fe:
            coord = pickle.load(fe)

        with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/input_flux_OPT","rb") as ff:
            flux = pickle.load(ff)

        FEMout = torch.tensor(np.concatenate(fdata,axis=0))# Getting solution MESH data
        nFEMout = Variable(FEMout, requires_grad=True).to(torch.double)
        ndata = data.x.squeeze().to(torch.double)
        loss = criterion( nFEMout, ndata ) #Loss is between input solution mesh and output of FEM solver
        

        #Global pooling of output for graph level classification
        out = torch.squeeze(out)
        nout = []
        for i in range(len(data.y)):
            a = i*int(len(data.x)/len(data.y))
            b = (i+1)*int(len(data.x)/len(data.y))
        
            nout.append(torch.mean(out[a:b]))

        nnout = Variable(torch.tensor(nout, dtype=torch.double),requires_grad=True)
        ny = data.y.to(torch.double)
        """

        nnout = out.squeeze().to(torch.double)
        ny = data.y.to(torch.double)

        loss = criterion( nnout, ny)
        loss_all += loss.item() # adds loss for each batch
        loss.backward()
        optimizer.step()
        c = c + 1
   

    return loss_all/c # reporting average loss per batch



def validate():

    loss_all = 0
    c = 0 # counter for number of batches

    for data in vadloader:
        model.eval()
        out = model(data.x, data.edge_index,data.pos)

        """
        #Save output and FEM solver
        np.save("GNN_output.npy",out.detach().numpy())
        os.system("python FEM.py")

        
        with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/solution_data_OPT","rb") as fd:
            fdata = pickle.load(fd)

        with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/coordinate_data_OPT","rb") as fe:
            coord = pickle.load(fe)

        with open("/Users/oluwadamilolafasina/Inverse_GNN/FEM_output/input_flux_OPT","rb") as ff:
            flux = pickle.load(ff)

        FEMout = torch.tensor(np.concatenate(fdata,axis=0))# Getting solution MESH data
        nFEMout = Variable(FEMout, requires_grad=True).to(torch.double)
        ndata = data.x.squeeze().to(torch.double)
        loss = criterion( nFEMout, ndata ) #Loss is between input solution mesh and output of FEM solver
        """

        nnout = out.squeeze().to(torch.double)
        ny = data.y.to(torch.double)
        loss = criterion( nnout, ny)
        loss_all += loss.item()
        c = c + 1


    return loss_all/c #normalizing by number of batches


svad = []
svtrain = []

for epoch in range(1, 10):
    train_loss = train()
    validate_loss = validate()

    svtrain.append(train_loss)
    svad.append(validate_loss)

    if epoch % 10 == 0: #Print every 10 epochs
        print( "Train Loss:",train_loss,"Validate Loss",validate_loss )

        
    #print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}) #, Validation Loss: {validate_loss:.4f}')


"""
#Testing model with MSE
tloss = 0
for data in testloader:

    model.eval()
    out = model(data.x,data.edge_index,data.pos)
    tloss += criterion(out,data.y)
    #print(tloss)

print("Average Test Loss (MSE)",tloss/len(testloader))
"""


plt.figure()
plt.title("Experiment 1")
plt.plot(svtrain)
plt.plot(svad)
plt.yscale("log")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss (Log-Scale)")
plt.legend(["Train Loss","Validation Loss"])
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/Figures/exp_one_train_vad_3.png")
plt.show()



plt.figure()
plt.title("Experiment 1 - Train Loss")
plt.plot(svtrain)
plt.yscale("log")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss (Log-Scale)")
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/Figures/exp_one_train_3.png")
plt.show()



plt.figure()
plt.title("Experiment 1 - Validation Loss")
plt.plot(svad)
plt.yscale("log")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss (Log-Scale)")
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/Figures/exp_one_vad_3.png")
plt.show()
