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
from torch_geometric.data import DataLoader
import sys
import pickle

with open("solution_data","rb") as fd:
    data = pickle.load(fd)

with open("coordinate_data","rb") as fe:
    coord = pickle.load(fe)

with open("input_flux","rb") as ff:
    flux = pickle.load(ff)

"""

# ---------------------------------------   Data import ----------------------------------# 

#Define problem variables
x_i = np.arange(0,10)
y_i = np.arange(0,10)
ev_mi = np.arange(0,10)
ev_ni = np.arange(0,10)
slist = np.arange(10)


#generate [100 x 2] node feature array with each index being a coordinate
x = [] 
for i in range(len(x_i)):
    for j in range(len(y_i)):
        x.append(np.array([i,j]))
nfeat = np.array(x)


#Create edge index array  (Defines graph connectivity using coordinate vector; code block below is specific to graphs with spatial coordinates)
source = []
target = []
for i in range(nfeat.shape[0]):
    ind = np.argwhere(np.all(np.abs(nfeat[i] - nfeat) <= 1,axis=1)).squeeze() # checking node distance on grid
    source.append(np.repeat(i,len(ind)))
    target.append(ind)

source = np.concatenate(source, axis=0)
target = np.concatenate(target, axis=0)

source = np.array(source).reshape((1,source.shape[0]))
target = np.array(target).reshape((1,target.shape[0]))
edge_index = torch.tensor(np.concatenate((source,target),axis=0))



#Defining Train/Test/Split Indeces

ntraingraphs = 700
nvadgraphs = 200
ntestgraphs = 100
ntotgraphs = ntraingraphs + ntestgraphs + nvadgraphs
graphs = np.arange(ntotgraphs)

trainind = np.random.choice(graphs,ntraingraphs,replace=False)
vadind = [x for x in graphs if x not in trainind][:nvadgraphs]
tempind =np.hstack((trainind,vadind))
testind = [x for x in graphs if x not in tempind]




#Getting 100 solution meshes 

train_datalist = []
for i in range(len(trainind)):

    print(i)

    j= int(np.floor(trainind[i]/100)) # specifies the paramter function
    a = int(np.floor(trainind[i]/10)) # Specifies the eigenvalue coefficient

    vmesh = datagen(x_i,y_i,ev_mi,ev_ni,a,sourced(j))
    totfeat = np.concatenate((nfeat,vmesh),axis = 1)
    totparam  = np.array([sourced(j)(x,y) for x in range(10) for y in range(10)])

    train_datalist.append(Data(x=torch.tensor(totfeat,dtype=torch.float),y=torch.tensor(totparam,dtype=torch.float),edge_index=edge_index))

train_loader = DataLoader(train_datalist,batch_size=10,shuffle=True)
torch.save(train_loader,"train_data.pt")
print("Number of training samples",len(train_datalist))



vad_datalist = []
for i in range(len(vadind)):

    print(i)

    j=int(np.floor(vadind[i]/100)) # specifies the paramter function
    a=int(np.floor(vadind[i]/10)) # Specifies the eigenvalue coefficient

    vmesh = datagen(x_i,y_i,ev_mi,ev_ni,a,sourced(j))
    totfeat = np.concatenate((nfeat,vmesh),axis = 1)
    totparam  = np.array([sourced(j)(x,y) for x in range(10) for y in range(10)])

    vad_datalist.append(Data(x=torch.tensor(totfeat,dtype=torch.float),y=torch.tensor(totparam,dtype=torch.float),edge_index=edge_index))

vad_loader = DataLoader(vad_datalist,batch_size=10,shuffle=True)
torch.save(vad_loader,"vad_data.pt")
print("Number of Validation samples",len(vad_datalist))


test_datalist = []
for i in range(len(testind)):

    print(i)

    j= int(np.floor(testind[i]/100)) # specifies the paramter function
    a = int(np.floor(testind[i]/10)) # Specifies the eigenvalue coefficient

    vmesh = datagen(x_i,y_i,ev_mi,ev_ni,a, sourced(j))
    totfeat = np.concatenate((nfeat,vmesh),axis = 1)
    totparam  = np.array([sourced(j)(x,y) for x in range(10) for y in range(10)])

    test_datalist.append(Data(x=torch.tensor(totfeat,dtype=torch.float),y=torch.tensor(totparam,dtype=torch.float),edge_index=edge_index))


test_loader = DataLoader(test_datalist,batch_size=10,shuffle=True)
torch.save(test_loader,'test_data.pt')
print("Number of test samples",len(test_datalist))




#Check's number of btches
#for batch in train_loader:
#    print(batch)


"""

#Load train/test data

train_loader = torch.load('train_data.pt')
test_loader = torch.load('test_data.pt')
vad_loader = torch.load('vad_data.pt')


# Script for training GNN 


model = NodeClassifier(num_node_features = 3, hidden_features = 3, num_classes = 1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = torch.nn.MSELoss()
model = model.float()

def train():
    loss_all = 0
    c = 0 # Counter for number of batches
    for data in train_loader:

        model.train()   
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        print("Model output:",out.size())
        loss = criterion( out, data.y )
        
        loss.backward()
        loss_all += loss.item() # adds loss for each batch
        optimizer.step()
        c = c + 1
        sys.exit()

    return loss_all/c # reporting average loss per batch



def validate():

    loss_all = 0
    c = 0 # counter for number of batches

    for data in vad_loader:
        model.eval()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss_all += loss.item()
        c = c + 1


    return loss_all/c #normalizing by number of batches


svad = []
svtrain = []

for epoch in range(1, 200):
    train_loss = train()
    validate_loss = validate()

    svtrain.append(train_loss)
    svad.append(validate_loss)

    if epoch % 10 == 0: #Print every 10 epochs
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Validation Loss: {validate_loss:.4f}')


#Testing model with MSE
tloss = 0
for data in test_loader:

    model.eval()
    out = model(data.x,data.edge_index)
    tloss += criterion(out,data.y)
    #print(tloss)

print("Average Test Loss (MSE)",tloss/len(test_loader))


plt.figure()
plt.title("Experiment 8")
plt.plot(svtrain)
plt.plot(svad)
plt.yscale("log")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss (Log-Scale)")
plt.legend(["Train Loss","Validation Loss"])
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/exp/exp_8_train_vad.png")
plt.show()

plt.figure()
plt.title("Experiment 8 - Train Loss")
plt.plot(svtrain)
plt.yscale("log")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss (Log-Scale)")
plt.legend(["Train Loss","Validation Loss"])
plt.show()
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/exp/exp_8_train.png")

plt.figure()
plt.title("Experiment 8 - Validation Loss")
plt.plot(svad)
plt.yscale("log")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss (Log-Scale)")
plt.show()
plt.savefig("/Users/oluwadamilolafasina/Inverse_GNN/exp/exp_8_vad.png")