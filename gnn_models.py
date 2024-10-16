
import torch
import torch.nn as nn
from torch_geometric.nn import  global_mean_pool,  GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import ipdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        hidden = 128
        self.conv1 = GCNConv(7, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(512)

        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(512,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 512)
        

        self.dropout = nn.Dropout(0.5)



    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x) 
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch = batch)

        return global_mean_pool(y[0], y[3])
    
class ppi_model(nn.Module):
    def __init__(self):
        super(ppi_model, self).__init__()
        self.GCN = GCN()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 13)

    def forward(self, batch_1, p_x_1, p_edge_1, batch_2, p_x_2, p_edge_2
                ):


        x1 = self.GCN(p_x_1, p_edge_1, batch_1)
        x2 = self.GCN(p_x_2, p_edge_2, batch_2)
        x = torch.mul(x1, x2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
