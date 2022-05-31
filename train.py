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
import scipy.sparse as sp
import time
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
import contrast_util

#import GCL.losses as L
#import GCL.augmentors as A

#from GCL.eval import get_split, LREvaluator
#from GCL.models import DualBranchContrast
from base_model import GCN_dense
from base_model import GCN_emb
#from base_model import GCN
from base_model import GPN_Encoder

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata

valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}
def load_data_pretrain(dataset_source):
    
    if dataset_source=='ogbn-arxiv':
    
        from ogb.nodeproppred import NodePropPredDataset

        dataset = NodePropPredDataset(name = dataset_source)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0] # graph: library-agnostic graph object

        n1s=graph['edge_index'][0]
        n2s=graph['edge_index'][1]
              
        num_nodes = graph['num_nodes']
        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    
        
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
        features=torch.FloatTensor(graph['node_feat'])
        labels=torch.LongTensor(labels).squeeze()
        
        #class_list_test = random.sample(list(range(40)),20)
        #train_class=list(set(list(range(40))).difference(set(class_list_test)))
        #class_list_valid = random.sample(train_class, 5)
        #class_list_train = list(set(train_class).difference(set(class_list_valid)))
        #json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))
        class_list_train,class_list_valid,class_list_test=json.load(open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train,idx_valid,idx_test=[],[],[]
        
        for i in range(labels.shape[0]):
            if labels[i] in class_list_train:
                idx_train.append(i)
            elif labels[i] in class_list_valid:
                idx_valid.append(i)
            else:
                idx_test.append(i)
        print(labels.shape)

    elif dataset_source in valid_num_dic.keys():
        n1s = []
        n2s = []
        for line in open("./few_shot_data/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        data_train = sio.loadmat("./few_shot_data/{}_train.mat".format(dataset_source))
        data_test = sio.loadmat("./few_shot_data/{}_test.mat".format(dataset_source))

        num_nodes = max(max(n1s),max(n2s)) + 1
        labels = np.zeros((num_nodes,1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        
        


        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)



        train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))
        class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

        class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

        class_list_train = list(set(train_class).difference(set(class_list_valid)))

        #json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))

        class_list_train,class_list_valid,class_list_test=json.load(open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train,idx_valid,idx_test=[],[],[]
        for idx_,class_list_ in zip([idx_train,idx_valid,idx_test],[class_list_train,class_list_valid,class_list_test]):
            for class_ in class_list_:
                idx_.extend(id_by_class[class_])
  

    elif dataset_source=='cora-full':
        adj, features, labels, node_names, attr_names, class_names, metadata=load_npz_to_sparse_graph('./dataset/gnn-benchmark/data/npz/cora_full.npz')
        
        

        
        sparse_mx = adj.tocoo().astype(np.float32)
        indices =np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        
        n1s=indices[0].tolist()
        n2s=indices[1].tolist()
        

        adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
        adj= sparse_mx_to_torch_sparse_tensor(adj)
        features=features.todense()
        features = torch.FloatTensor(features)
        labels=torch.LongTensor(labels).squeeze()
        
        
        
        
        
        #features=features.todense()
        
        
        #class_list_test = random.sample(list(range(70)),25)
        #train_class=list(set(list(range(70))).difference(set(class_list_test)))
        #class_list_valid = random.sample(train_class, 20)
        #class_list_train = list(set(train_class).difference(set(class_list_valid)))
        #json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))
        
        class_list_train,class_list_valid,class_list_test=json.load(open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train,idx_valid,idx_test=[],[],[]
        
        for i in range(labels.shape[0]):
            if labels[i] in class_list_train:
                idx_train.append(i)
            elif labels[i] in class_list_valid:
                idx_valid.append(i)
            else:
                idx_test.append(i)
        

    write_to_g_meta=False
    if write_to_g_meta:
        label_pkl={}
        name=[]
        label=[]
        for idx in idx_train:
            name.append('0_{}'.format(idx))
            label.append(labels[i].item())
            label_pkl['0_{}'.format(idx)]= labels[i].item()
            
        dataframe = pd.DataFrame({'a_name':name,'b_name':label})
        dataframe.to_csv("./G-Meta/new_data/{}/train.csv".format(dataset_source),index=False,sep=',')
        
        name=[]
        label=[]
        for idx in idx_valid:
            name.append('0_{}'.format(idx))
            label.append(labels[i].item())
            label_pkl['0_{}'.format(idx)]= labels[i].item()
        dataframe = pd.DataFrame({'a_name':name,'b_name':label})
        dataframe.to_csv("./G-Meta/new_data/{}/val.csv".format(dataset_source),index=False,sep=',')
        
        name=[]
        label=[]
        for idx in idx_test:
            name.append('0_{}'.format(idx))
            label.append(labels[i].item())
            label_pkl['0_{}'.format(idx)]= labels[i].item()
        dataframe = pd.DataFrame({'a_name':name,'b_name':label})
        dataframe.to_csv("./G-Meta/new_data/{}/train.csv".format(dataset_source),index=False,sep=',')
        
        np.save("./G-Meta/new_data/{}/features.npy".format(dataset_source),features.numpy())
        pkl.dump(label_pkl,open("./G-Meta/new_data/{}/label.pkl".format(dataset_source),'wb'))
        graph=[dgl.from_scipy(sparse_mx)]
        pkl.dump(graph,open('./G-Meta/new_data/{}/graph_dgl.pkl'.format(dataset_source),'wb'))
        
        print(1/0)
        
        
    class_train_dict=defaultdict(list)
    for one in class_list_train:
        for i,label in enumerate(labels.numpy().tolist()):
            if label==one:
                class_train_dict[one].append(i)


    print(len(idx_train))
    print(len(idx_train)+len(idx_valid))
    print(features.shape[0])

    return adj, features, labels, idx_train, idx_valid, idx_test, n1s,n2s,class_train_dict


def neighborhoods_(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    #adj = torch.tensor(adj, dtype=torch.float)
    #adj=adj.to_dense()
    #print(type(adj))
    if use_cuda:
        adj = adj.cuda()
    #hop_adj = power_adj = adj
    
    
    #return (adj@(adj.to_dense())+adj).to_dense().cpu().numpy().astype(int)
    
    hop_adj=adj+torch.sparse.mm(adj,adj)

    hop_adj=hop_adj.to_dense()
    #hop_adj = (hop_adj > 0).to_dense()
    
    #for i in range(n_hops - 1):
        #power_adj = power_adj @ adj
        #prev_hop_adj = hop_adj
        #hop_adj = hop_adj + power_adj
        #hop_adj = (hop_adj > 0).float()
        
    hop_adj=hop_adj.cpu().numpy().astype(int)
    
    return (hop_adj>0).astype(int)
        
    #return hop_adj.cpu().numpy().astype(int)

def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    #adj = torch.tensor(adj, dtype=torch.float)
    #adj=adj.to_dense()
    #print(type(adj))
    if n_hops==1:
        return adj.cpu().numpy().astype(int)
    
    
    if use_cuda:
        adj = adj.cuda()
    #hop_adj = power_adj = adj


    
    #for i in range(n_hops - 1):
        #power_adj = power_adj @ adj
    hop_adj = adj+adj@ adj
    hop_adj = (hop_adj > 0).float()

        
        
    np.save(hop_adj.cpu().numpy().astype(int),'./neighborhoods_{}.npy'.format(dataset))
        
    return hop_adj.cpu().numpy().astype(int)
    

def InforNCE_Loss(anchor, sample, tau, all_negative=False):

    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0]==sample.shape[0]


    pos_mask=torch.eye(anchor.shape[0],dtype=torch.float).cuda()
    neg_mask=1.-pos_mask


    sim = _similarity(anchor, sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    if not all_negative:
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    else:
        log_prob = - torch.log(exp_sim.sum(dim=1, keepdim=True))

    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    
    #print(1)
    #return -loss[:10].mean()

    #print(sim)
    
    
    return -loss.mean(), sim

from torch_geometric.nn import GINConv
class GIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): 
        super(GIN, self).__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Linear(nfeat, nhid), 
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nhid), 
        )
        self.conv1 = GINConv(self.mlp1)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, x, edge_index): 
        #edge_index = (adj > 0).nonzero().t()
        #row, col = edge_index
        x = self.conv1(x, edge_index)
        #return self.fc(x)

        #print(x)
        #print(1/0)
        

        return x

from torch_geometric.nn import GCNConv   
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)


        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        #x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x    

    
    
    
    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', default=True,help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--pretrain_lr', type=float, default=0.05,
                    help='Initial learning rate.')
#权重衰减
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--pretrain_dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', default='dblp',
                    help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')


args = parser.parse_args(args=[])

#args.use_cuda = torch.cuda.is_available()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)

loss_f=nn.CrossEntropyLoss()
# Load data


N=10
K=5
n_hops=1

#for dataset in ['ogbn-arxiv','Amazon_clothing','Amazon_eletronics','dblp','cora-full']:

for dataset in ['cora-full',]:

    adj_sparse, features, labels, idx_train, idx_val, idx_test,n1s,n2s,class_train_dict = load_data_pretrain(dataset)

    #features/=features.max(0,keepdim=True)[0]

    #print(adj_sparse[[0,1,2,3]])
    
    #args.hidden1=features.shape[-1]
    
    adj=adj_sparse.to_dense()
    

    #print(adj[:10,:10])

    #edge_index=[[one1,one2] for one1,one2 in zip(n1s,n2s)]
    edge_index=torch.LongTensor([n1s,n2s])

    #total_neighbors=neighborhoods(adj=adj, n_hops=n_hops, use_cuda=True)
    total_neighbors=adj.cpu().numpy()
    
    #import json
    #neighbors=[]
    #for i in range(total_neighbors.shape[0]):
    #    neighbors.append(total_neighbors[i].nonzero()[0].tolist())
    #json.dump(neighbors,open('neighbors_{}.json'.format(dataset),'w'))
    #continue
    
    
    
    model = GCN_dense(nfeat=args.hidden1,
                nhid=args.hidden2,
                nclass=labels.max().item() + 1,
                dropout=args.pretrain_dropout)

    #generater=nn.Linear(args.hidden1, (args.hidden1+1)*args.hidden2*2+(args.hidden2+1)*args.hidden2*2)
    
    GCN_model=GCN_emb(nfeat=features.shape[1],
                nhid=args.hidden1,
                nclass=labels.max().item() + 1,
                dropout=args.pretrain_dropout)

    classifier=nn.Linear(args.hidden1,labels.max().item() + 1)
    
    optimizer = optim.Adam([{'params': model.parameters()},
                                   {'params': GCN_model.parameters()},{'params': classifier.parameters()}],lr=args.pretrain_lr, weight_decay=args.weight_decay)


    if args.use_cuda:
        model.cuda()
        GCN_model.cuda()
        features = features.cuda()
        #
        #generater=generater.cuda()
        #adj = adj.cuda()
        adj_sparse=adj_sparse.cuda()
        labels = labels.cuda()
        classifier=classifier.cuda()

    def pre_train(epoch):
        #classes=np.random.choice(list(range(len(class_train_dict))),N,replace=False)
        
        emb_features=GCN_model(features,adj_sparse)
        #emb_features=features

        target_idx=[]
        target_new_idx=[]
        target_graph_adj_and_feat=[]
        pos_graph_adj_and_feat=[]


        classes=np.random.choice(list(class_train_dict.keys()),N,replace=False).tolist()

        #for novel sample
        classes.extend([-1]*N)

        for i in classes:

            #sample from one specific class
            if i!=-1:
                pos_node_idx=np.random.choice(class_train_dict[i],K,replace=False)



            #build target subgraph

            while True:
                if i!=-1:
                    idx=np.random.choice(class_train_dict[i],1,replace=False)[0] 
                else:
                    idx=np.random.choice(idx_test,1,replace=False)[0] 

                target_neighbors=total_neighbors[idx].nonzero()[0]
                if len(target_neighbors)<=1 or idx in pos_node_idx: 
                    continue
                else:
                    break
                    
            target_neighbors_2hop=[]
            for one in target_neighbors:
                target_neighbors_2hop.extend(total_neighbors[one].nonzero()[0])
            target_neighbors=list(set(target_neighbors_2hop))
            
            
            target_new_idx.append(target_neighbors.index(idx))
            target_idx.append(idx)
            target_graph_adj=adj[target_neighbors,:][:,target_neighbors].cuda()
            #target_graph_edge_index= (target_graph_adj > 0).nonzero().t()

            target_graph_feat=emb_features[target_neighbors]
            
            #print(target_graph_feat.shape)

            target_graph_adj_and_feat.append([target_graph_adj,target_graph_feat])


            if i!=-1:
                #build class-ego subgraph, if i>=N we generate other classes
                pos_node_idx=pos_node_idx

                pos_graph_neighbors=torch.nonzero(adj[pos_node_idx,:].sum(0)).squeeze()
                pos_graph_adj=adj[pos_graph_neighbors,:][:,pos_graph_neighbors].cuda()
                

                pos_class_graph_adj=torch.eye(pos_graph_neighbors.shape[0]+1,dtype=torch.float).cuda()
                pos_class_graph_adj[1:,1:]=pos_graph_adj

                #pos_graph_edge_index= (pos_class_graph_adj > 0).nonzero().t()

                pos_class_graph_feat=torch.cat([emb_features[pos_node_idx].mean(0,keepdim=True),emb_features[pos_graph_neighbors]],0)

                pos_graph_adj_and_feat.append([pos_class_graph_adj,pos_class_graph_feat])



        model.train()  
        GCN_model.train()
        optimizer.zero_grad()

        
        class_generate_emb=torch.stack([sub[1][0] for sub in pos_graph_adj_and_feat],0).mean(0)
        parameters=model.generater(class_generate_emb)
        
        #parameters=torch.zeros(parameters.shape,dtype=torch.float).cuda()
        
        gc1_parameters=parameters[:(args.hidden1+1)*args.hidden2*2]
        gc2_parameters=parameters[(args.hidden1+1)*args.hidden2*2:]
        
        gc1_w=gc1_parameters[:args.hidden1*args.hidden2*2].reshape([2,args.hidden1,args.hidden2])
        gc1_b=gc1_parameters[args.hidden1*args.hidden2*2:].reshape([2,args.hidden2])
        
        gc2_w=gc2_parameters[:args.hidden2*args.hidden2*2].reshape([2,args.hidden2,args.hidden2])
        gc2_b=gc2_parameters[args.hidden2*args.hidden2*2:].reshape([2,args.hidden2])
        
        
        

        target_embs=[]
        for i in range(N+N):
            sub_adj=target_graph_adj_and_feat[i][0]
            sub_feat=target_graph_adj_and_feat[i][1]
            new_idx=target_new_idx[i]
            #target_embs.append(model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b)[new_idx])
            if i<N:
                target_embs.append(model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b).mean(0))
            else:
                target_embs.append(model(sub_feat,sub_adj).mean(0))
                               
        target_embs=torch.stack(target_embs,0)


        
        class_ego_embs=[]
        for sub_adj, sub_feat in pos_graph_adj_and_feat:
            class_ego_embs.append(model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b)[0])
        class_ego_embs=torch.stack(class_ego_embs,0)

        loss=0

        class_contras_loss, similarity=InforNCE_Loss(target_embs[:N],class_ego_embs,tau=0.2)
        
        unlabeled_loss, _ =InforNCE_Loss(target_embs[N:],class_ego_embs,tau=0.2,all_negative=True)
        
        loss_supervised=loss_f(classifier(emb_features[idx_train]), labels[idx_train])


        loss+=class_contras_loss

        loss+=loss_supervised


        loss+=unlabeled_loss

        loss.backward()
        optimizer.step()
        

        labels_train=labels[target_idx[:N]]
        for j, class_idx in enumerate(classes[:N]):
            labels_train[labels_train==class_idx]=j

        acc_train = accuracy(similarity, labels_train)



        if epoch%20==0:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss.item()), #'loss_dis: {:.4f}'.format(dis_loss.item()),
                  'loss_sup: {:.4f}'.format(loss_supervised.item()),
                  'loss_constra: {:.4f}'.format(class_contras_loss.item()),
                  'loss_unlabel: {:.4f}'.format(unlabeled_loss.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()))

    # Train model
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        pre_train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    torch.save(model.state_dict(),'./saved_models/GIN_{}_{}_epochs_class_ego_contrast.pth'.format(dataset,args.epochs))
    torch.save(GCN_model.state_dict(),'./saved_models/GIN_{}_{}_epochs_class_ego_contrast_GCN.pth'.format(dataset,args.epochs))
    
    
    
    
    
    
