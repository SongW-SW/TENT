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


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    GraphConvolution作为一个类，首先需要定义其相关属性。
    本文中主要定义了其输入特征in_feature、输出特征out_feature两个输入，
    以及权重weight和偏移向量bias两个参数，同时调用了其参数初始化的方法
    """

    def __init__(self, in_features, out_features, bias=True):#参数属性定义
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 由于weight是可以训练的，因此使用parameter定义
        # 由于bias是可以训练的，因此使用parameter定义
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#参数初始化
        #为了让每次训练产生的初始参数尽可能的相同，从而便于实验结果的复现，可以设置固定的随机数生成种子。
        stdv = 1. / math.sqrt(self.weight.size(1))
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        self.weight.data.uniform_(-stdv, stdv)
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):#正向传播
        #此处主要定义的是本层的前向传播，通常采用的是A∗X∗W的计算方法。
        #由于A是一个sparse变量，因此其与X进行卷积的结果也是稀疏矩阵

        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        # torch.spmm(a,b)是稀疏矩阵相乘
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):#字符串表达
        #__repr__()方法是类的实例化对象用来做“自我介绍”的方法，
        #默认情况下，它会返回当前对象的“类名+object at+内存地址”，
        #而如果对该方法进行重写，可以为其制作自定义的自我描述信息。
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphConvolution_dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True):#参数属性定义
            super(GraphConvolution_dense, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            # 由于weight是可以训练的，因此使用parameter定义
            # 由于bias是可以训练的，因此使用parameter定义
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):#参数初始化
        #为了让每次训练产生的初始参数尽可能的相同，从而便于实验结果的复现，可以设置固定的随机数生成种子。
        stdv = 1. / math.sqrt(self.weight.size(1))
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        self.weight.data.uniform_(-stdv, stdv)
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
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


    def permute(self, input_adj, input_feat, drop_rate=0.1):
        
        #return input_adj

        adj_random=torch.rand(input_adj.shape).cuda()+torch.eye(input_adj.shape[0]).cuda()
        
        feat_random=np.random.choice(input_feat.shape[0], int(input_feat.shape[0]*drop_rate), replace=False).tolist()
        
        masks=torch.zeros(input_feat.shape).cuda()
        masks[feat_random]=1
        
        random_tensor=torch.rand(input_feat.shape).cuda()

        return input_adj*(adj_random>drop_rate), input_feat*(1-masks)+random_tensor*masks

    def forward(self, x, adj,w1=None, b1=None,w2=None,b2=None):


        x = F.relu(self.gc1(x, adj, w1, b1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, w2, b2)
        return x



class Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 由于weight是可以训练的，因此使用parameter定义

        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):#参数初始化
        #为了让每次训练产生的初始参数尽可能的相同，从而便于实验结果的复现，可以设置固定的随机数生成种子。
        stdv = 1. / math.sqrt(self.weight.size(1))
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        self.weight.data.uniform_(-stdv, stdv)
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,input,w=None,b=None):
        if w!=None:


            return torch.mm(input,w)+b
        else:
            return torch.mm(input,self.weight)+self.bias



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

