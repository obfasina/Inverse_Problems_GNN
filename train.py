import torch
from torch import nn
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from synthetic_data import datagen
import numpy as np
import matplotlib.pyplot as plt
from Granola_GNN import NodeClassifier

# Defining data

#Testing synthetic data generation

#Define problem variables
x_i = np.arange(0,10)
y_i = np.arange(0,10)
ev_mi = np.arange(0,10)
ev_ni = np.arange(0,10)


fmesh = datagen(x_i,y_i,ev_mi,ev_ni)
print(fmesh.shape)
plt.imshow(fmesh)

#generate 100-d node feature array with each index being a coordinate

x = [] 
for i in range(len(x_i)):
    for j in range(len(y_i)):
        x.append(np.array([i,j]))

nfeat = np.array(x)


#generate 100-d node label array with each index being a coordinate

nlabel = [] 
for i in range(len(x_i)):
    for j in range(len(y_i)):
        nlabel.append(fmesh[i,j])

nlabel = np.array(nlabel)

#Create edge index array 
source = []
target = []
for i in range(nfeat.shape[0]):
    ind = np.argwhere(np.all(np.abs(nfeat[i] - nfeat) <= 1,axis=1)).squeeze()
    source.append(np.repeat(i,len(ind)))
    target.append(ind)

source = np.concatenate(source, axis=0)
target = np.concatenate(target, axis=0)

source = np.array(source).reshape((1,source.shape[0]))
target = np.array(target).reshape((1,target.shape[0]))
edge_index = torch.tensor(np.concatenate((source,target),axis=0))

#First index is index of source nodes
#Second index is index of target nodes


data = Data(x=torch.tensor(nfeat,dtype=torch.float),y=torch.tensor(nlabel,dtype=torch.float),edge_index=edge_index)
print("TEST",data.x.dtype)
print("TEST AGAIN",data.edge_index.dtype)

# Script for training GNN 


model = NodeClassifier(num_node_features = 2, hidden_features = 2, num_classes = 100)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = torch.nn.MSELoss(requires_grad=True)
model = model.float()

def train():

    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    print("Output",out.size())
    print("Target",data.y.size())
    pred = torch.argmax(out, dim=1)
    print(pred)
    print(data.y)
    loss = criterion( pred, data.y )
    print(loss)
    loss.backward()
    optimizer.step()

    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {test_acc}')