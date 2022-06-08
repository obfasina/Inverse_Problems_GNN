import torch
from torch import nn
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from synthetic_data import datagen
import numpy as np
import matplotlib.pyplot as plt



#Building Convolutional layer
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='mean') #  Specify aggregation scheme
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        #Linear Transformation
        x = self.lin(x)

        #Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j):
        # x_j has shape [E, in_channels]

        #Aggregates neighboring nodes

        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]


        new_embedding = torch.cat([aggr_out, x], dim=1)
        
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        
        return new_embedding

#Defining GNN
class NodeClassifier(torch.nn.Module):
    def __init__(self,num_node_features,hidden_features,num_classes):
        super(NodeClassifier,self).__init__()

        self.conv1 = SAGEConv(num_node_features,hidden_features)
        self.conv2 = SAGEConv(hidden_features,hidden_features)
        self.conv3 = SAGEConv(hidden_features,hidden_features)
        self.bn1 = torch.nn.BatchNorm1d(hidden_features)
        self.lin1 = torch.nn.Linear(hidden_features,num_classes)
        self.act1 = torch.nn.ReLU()


    def forward(self,x, edge_index):
        
        
        x = self.conv1(x,edge_index)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x,edge_index)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv3(x,edge_index)
        x = self.lin1(x)

        return x

