# 


import torch
from torch import nn
#from sympy import solve_triangulated
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from synthetic_data import datagen#, sourced
import numpy as np
#import matplotlib.pyplot as plt
from Granola_GNN import NodeClassifier
from torch_geometric.loader import DataLoader
import sys
import pickle
import os
from torch.autograd import Variable
import wandb
import fenics
from torch_cluster import knn_graph


wandb.init(project="fenics_hidden_nodemb_128")
wandb.init(project="trash")
#Defining Pytorch geometric graph


def graphgen(gdata,gcoord,glabel):

    "gdata = node solution values; gcoord = node coordinates; glabel = node labels"

    
    #Defining node features
    nodefeats = torch.tensor(gdata.reshape( (len(gdata),1) ),dtype=torch.float)
    nodelabel = torch.tensor(np.repeat(glabel,len(gdata)),dtype=torch.float)
    
    """
    
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
    
    """   
    

    #Define edge index using KNN
    edges = knn_graph(torch.tensor(gcoord),k=4)

    #Define Graph
    graph = Data(x=nodefeats,y=nodelabel,edge_index=edges,pos=gcoord)

    return graph


#Loading data
print("Time to play:")
print("Loading lists of data")
with open("/home/dami/Inverse_GNN/FEM_output/fenics_sol","rb") as fd:
    data = pickle.load(fd)

with open("/home/dami/Inverse_GNN/FEM_output/fenics_coord","rb") as fe:
    coord = pickle.load(fe)

with open("/home/dami/Inverse_GNN/FEM_output/fenics_lab","rb") as ff:
    lab = pickle.load(ff)
 

data = data.tolist()
lab = lab.tolist()

ndata = [np.array(x) for x in data]
nlab = [np.array(x) for x in lab]


"""
#Generating graphs

ind = np.arange(len(data))#Shuffled indices
np.random.shuffle(ind)

pygraphs = []
for i in range(len(data)):
    print("Graph Number:",i)
    pygraphs.append(graphgen(ndata[ind[i]],coord,nlab[ind[i]]))

#Saving/Loading graph data
with open("/home/dami/Inverse_GNN/datagraphs","wb") as fg:
    pickle.dump(pygraphs,fg)

sys.exit()
"""

with open("/home/dami/Inverse_GNN/datagraphs","rb") as ff:
    pygraphs = pickle.load(ff)


#train/validation/test split (60/20/20) [NOTE: Data has already been shuffled] in data loaders

#Define data loaders
trainloader = DataLoader(pygraphs[:int(round(len(data)*.6))],batch_size=100,shuffle=True)
vadloader = DataLoader(pygraphs[ int(round(len(data)*.6)): int(round(len(data)*.8))],batch_size=100,shuffle=True)
testloader = DataLoader(pygraphs[int(round(len(data)*.8)):],batch_size=100,shuffle=True)

print("graph data",pygraphs[0])


# Script for training GNN 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NodeClassifier(num_node_features = 1, hidden_features = 128, nodes = 1) #len(pygraphs[1].x))
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
criterion = torch.nn.MSELoss()
model = model.float().to(device)
lamda = 1 # regularization paramter

def train(fenicsopt,ep,regp):

    loss_all = 0
    c = 0 # Counter for number of batches
    for data in trainloader:

        #Generate model output
        data = data.to(device)
        model.train()   
        optimizer.zero_grad()
        out = model(data.x, data.edge_index,data.pos)


        if fenicsopt and ep % 3 == 0 and ep > 700: #adjust weights based on solution mesh loss every 10 epochs

            regp = regp + .01
            #Call to fenics
            np.save("GNN_output.npy",out.cpu().detach().numpy())
            os.system("python fenics_datagen.py")

            with open("/home/dami/Inverse_GNN/FEM_output/fenics_optsol","rb") as fh:
                fdata = pickle.load(fh)



            #Computing solution loss
            loss_u = criterion( torch.flatten(fdata).unsqueeze(dim=1).to(torch.double), data.x.cpu().to(torch.double))
            #loss_all += loss.item() # adds loss for each batch
            #loss_u.backward()
            #optimizer.step()


            # Computing parametr loss
            nnout = out.squeeze().to(torch.double)
            ny = data.y.to(torch.double)
            loss_p = criterion( nnout, ny)
            loss_all += loss_p.item() # adds loss for each batch

            loss = loss_p + (regp*loss_u) #Incorporating loss from FEM solver
            loss.backward()
            optimizer.step()
            c = c + 1



        else:
            # Computing loss
            nnout = out.squeeze().to(torch.double)
            ny = data.y.to(torch.double)
            loss = criterion( nnout, ny)
            loss_all += loss.item() # adds loss for each batch
            loss.backward()
            optimizer.step()
            c = c + 1
        
   

    return loss_all/c # reporting average loss per batch



def validate(fenicsopt,ep,regp):

    loss_all = 0
    c = 0 # counter for number of batches

    for data in vadloader:

        data = data.to(device)
        model.eval()
        out = model(data.x, data.edge_index,data.pos)

        if fenicsopt and ep % 3 == 0 and ep > 700:

            regp = regp + .01
            #Call to fenics
            np.save("GNN_output.npy",out.cpu().detach().numpy())
            os.system("python fenics_datagen.py")

            with open("/home/dami/Inverse_GNN/FEM_output/fenics_optsol","rb") as fh:
                fdata = pickle.load(fh)

            #Compute solution loss and adjust weights
            loss_u = criterion( torch.flatten(fdata).unsqueeze(dim=1).to(torch.double), data.x.cpu().to(torch.double))
            #loss_all += loss.item() # adds loss for each batch
            #loss_u.backward()
            #optimizer.step()


            # Computing parametr loss
            nnout = out.squeeze().to(torch.double)
            ny = data.y.to(torch.double)
            loss_p = criterion( nnout, ny)
            loss_all += loss_p.item() # adds loss for each batch

            loss = loss_p + (regp*loss_u) #Incorporating loss from FEM solver

            loss.backward()
            optimizer.step()
            c = c + 1


        else:
            # Computing loss
            nnout = out.squeeze().to(torch.double)
            ny = data.y.to(torch.double)
            loss = criterion( nnout, ny)
            loss_all += loss.item() # adds loss for each batch
            loss.backward()
            optimizer.step()
            c = c + 1

    return loss_all/c #normalizing by number of batches


svad = []
svtrain = []

fopt=True #Select whether you want fenics solver

for epoch in range(1, 1000):

    train_loss = train(fopt,epoch,lamda)
    validate_loss = validate(fopt,epoch,lamda)

    svtrain.append(train_loss)
    svad.append(validate_loss)

    wandb.log({"train loss:":train_loss})
    wandb.log({"validation loss:":validate_loss})

    if epoch % 10 == 0: #Print every 10 epochs

        print("Epoch:",epoch, " Train Loss:",train_loss," Validation Loss:",validate_loss )

        
    #print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}) #, Validation Loss: {validate_loss:.4f}')



#Testing model with MSE
tloss = 0
for data in testloader:

    data.to(device)
    model.eval()
    out = model(data.x,data.edge_index,data.pos)
    nout = torch.squeeze(out)

    print(nout.size())
    print(data.y.size())

    nGNNout = []
    nGNNlab = []
    bs = 100 # batch size
    nnodes = 81
    for i in range(bs):
        a = i*nnodes
        b = (i+1)*nnodes
        nGNNout.append(np.mean(nout[a:b].cpu().detach().numpy()))
        nGNNlab.append(np.mean(data.y[a:b].cpu().detach().numpy()))

    tloss += criterion(nout,data.y)
    #print(tloss)
    wandb.log({"output:":nGNNout})
    wandb.log({"label:":nGNNlab})

    nGNNout = np.array(nGNNout)
    nGNNlab = np.array(nGNNlab)

    print(nGNNout.size)
    print(nGNNlab.size)

    coefdata = np.concatenate((nGNNout.reshape((nGNNout.size,1)),nGNNlab.reshape((nGNNlab.size,1))),axis=1)
    wandb.log({"coefficients":wandb.Table(data=coefdata,columns=["predictions","labels"])})

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
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/Figures/exp_one_train_vad_2.png")
plt.show()



plt.figure()
plt.title("Experiment 1 - Train Loss")
plt.plot(svtrain)
plt.yscale("log")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss (Log-Scale)")
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/Figures/exp_one_train_2.png")
plt.show()



plt.figure()
plt.title("Experiment 1 - Validation Loss")
plt.plot(svad)
plt.yscale("log")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss (Log-Scale)")
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/Figures/exp_one_vad_2.png")
plt.show()

"""
