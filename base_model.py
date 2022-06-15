import networkx as nx
import numpy as np
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import sys
import scipy
import sklearn
import json
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import pickle as pkl


from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

        
 
class GraphConvolution_dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
            super(GraphConvolution_dense, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))

            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, w, b):
        
        if w==None and b==None:
            alpha_w=alpha_b=beta_w=beta_b=0
        else:
        
            alpha_w=w[0]
            beta_w=w[1]
            alpha_b=b[0]
            beta_b=b[1]

        support = torch.mm(input, self.weight*(1+alpha_w)+beta_w)
        output = torch.mm(adj, support)
        

        if self.bias is not None:
            return output + self.bias*(1+alpha_b)+beta_b
        else:
            return output

        
class GCN_dense(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_dense, self).__init__()

        self.gc1 = GraphConvolution_dense(nfeat, nhid)
        self.gc2 = GraphConvolution_dense(nhid, nhid)
        self.generater=nn.Linear(nfeat, (nfeat+1)*nhid*2+(nhid+1)*nhid*2)
        
        
        self.dropout = dropout

    def forward(self, x, adj,w1=None, b1=None,w2=None,b2=None):
        return self.gc1(x, adj, w1, b1)
        
        x = F.relu(self.gc1(x, adj, w1, b1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, w2, b2)
        return x
        

        
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_emb(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_emb, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        
        return self.gc1(x,adj)
        return F.dropout(self.gc1(x,adj), self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GPN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout


        proj_dim=32

        self.fc1 = torch.nn.Linear(nhid, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

