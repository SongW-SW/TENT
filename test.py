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
from base_model import GCN
from base_model import GraphConvolution
import time
import datetime
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score


from torch_geometric.data import Data
from base_model import GCN_dense
from base_model import GCN_emb

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
    if len(output.shape)==2:
        preds = output.max(1)[1].type_as(labels)
    else:
        preds=output
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



def task_generator(id_by_class, class_list, n_way, k_shot, m_query, maximum_value_train_each_class=None):

    # sample class indices
    class_selected = np.random.choice(class_list, n_way,replace=False).tolist()
    id_support = []
    id_query = []
    for cla in class_selected:
        if maximum_value_train_each_class:
            temp = np.random.choice(id_by_class[cla][:maximum_value_train_each_class], k_shot + m_query,replace=False)
        else:
            temp = np.random.choice(id_by_class[cla], k_shot + m_query,replace=False)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected



def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M
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
def load_data(dataset_source):
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
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
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
        
        
        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)
        
        
    elif dataset_source in valid_num_dic.keys():
        n1s = []
        n2s = []
        for line in open("few_shot_data/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))

        data_train = sio.loadmat("few_shot_data/{}_train.mat".format(dataset_source))
        train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))

        data_test = sio.loadmat("few_shot_data/{}_test.mat".format(dataset_source))
        class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

        labels = np.zeros((num_nodes, 1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

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

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

        #class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

        #class_list_train = list(set(train_class).difference(set(class_list_valid)))


    elif dataset_source=='cora-full':
        adj, features, labels, node_names, attr_names, class_names, metadata=load_npz_to_sparse_graph('./dataset/gnn-benchmark/data/npz/cora_full.npz')
        
        

        
        sparse_mx = adj.tocoo().astype(np.float32)
        indices =np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        
        n1s=indices[0].tolist()
        n2s=indices[1].tolist()
        
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        
        adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
        adj= sparse_mx_to_torch_sparse_tensor(adj)
        features=features.todense()
        features = torch.FloatTensor(features)
        labels=torch.LongTensor(labels).squeeze()
        
        
        
        #print(labels.max())
        
        #features=features.todense()
        
        
        #class_list_test = random.sample(list(range(69)),25)
        #train_class=list(set(list(range(69))).difference(set(class_list_test)))
        #class_list_valid = random.sample(train_class, 19)
        #class_list_train = list(set(train_class).difference(set(class_list_valid)))
        #json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))
        
        class_list_train,class_list_valid,class_list_test=json.load(open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        
        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)
        
        
        
        
    class_list_train,class_list_valid,class_list_test=json.load(open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


parser = argparse.ArgumentParser()

parser.add_argument('--use_cuda', action='store_true',default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
#parser.add_argument('--seed', type=int, default=1, help='Random seed.')


parser.add_argument('--train_episodes', type=int, default=1000,
                    help='Number of episodes to train.')
parser.add_argument('--episodes', type=int, default=100,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--test_mode', type=str, default='LR')



parser.add_argument('--way', type=int, default=10, help='way.')
parser.add_argument('--shot', type=int, default=3, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=20)
parser.add_argument('--dataset', default='dblp', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')
args = parser.parse_args(args=[])
args.cuda = torch.cuda.is_available() and args.use_cuda


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# -------------------------Meta-training------------------------------


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class GPN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return x


class GPN_Valuator(nn.Module):
    """
    For the sake of model efficiency, the current implementation is a little bit different from the original paper.
    Note that you can still try different architectures for building the valuator network.

    """

    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Valuator, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc3(x)

        return x
    
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
        self.fc = nn.Linear(nhid, nclass)

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
    
def InforNCE_Loss(anchor, sample, tau):

    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0]==sample.shape[0]

    pos_mask=torch.eye(anchor.shape[0],dtype=torch.float).cuda()
    neg_mask=1.-pos_mask


    sim = _similarity(anchor, sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    
    #print(1)
    #return -loss[:10].mean()

    #print(sim)
    
    
    return -loss.mean(), sim

def train(class_selected, id_support, id_query, n_way, k_shot):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def test(class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def LR_test(class_selected, id_support, id_query, n_way, k_shot):
    def normalize(x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    encoder.eval()
    
    use_subgraph=True
    use_similarity=False
    
    if not use_subgraph:

        #embeddings = encoder(features, adj)
        embeddings=emb_features
        z_dim = embeddings.size()[1]
        ## embedding lookup
        support_embeddings = embeddings[id_support]
        support_embeddings = support_embeddings.view([-1, z_dim])
        query_embeddings = embeddings[id_query]

    else:
        target_graph_adj_and_feat=[]
        target_new_idx=[]
        init_features_support=[]
        for idx in id_support:

            target_neighbors=total_neighbors[idx].nonzero()[0]
            target_neighbors_2hop=[]
            for one in target_neighbors:
                target_neighbors_2hop.extend(total_neighbors[one].nonzero()[0])
            target_neighbors=list(set(target_neighbors_2hop))

            target_new_idx.append(target_neighbors.index(idx))
            target_graph_adj=adj[target_neighbors,:][:,target_neighbors].cuda()
            target_graph_feat=emb_features[target_neighbors]
            target_graph_adj_and_feat.append([target_graph_adj,target_graph_feat])
            init_features_support.append(emb_features[idx])

        query_graph_adj_and_feat=[]
        query_new_idx=[]
        for idx in id_query:
            target_neighbors=total_neighbors[idx].nonzero()[0]
            target_neighbors_2hop=[]
            for one in target_neighbors:
                target_neighbors_2hop.extend(total_neighbors[one].nonzero()[0])
            target_neighbors=list(set(target_neighbors_2hop))
            query_new_idx.append(target_neighbors.index(idx))
            target_graph_adj=adj[target_neighbors,:][:,target_neighbors].cuda()
            target_graph_feat=emb_features[target_neighbors]
            query_graph_adj_and_feat.append([target_graph_adj,target_graph_feat])



        class_generate_emb=torch.stack(init_features_support,0).mean(0)
        parameters=task_model.generater(class_generate_emb)
        
        #parameters=torch.zeros(parameters.shape,dtype=torch.float).cuda()
        

        gc1_parameters=parameters[:(args.hidden1+1)*args.hidden2*2]
        gc2_parameters=parameters[(args.hidden1+1)*args.hidden2*2:]

        gc1_w=gc1_parameters[:args.hidden1*args.hidden2*2].reshape([2,args.hidden1,args.hidden2])
        gc1_b=gc1_parameters[args.hidden1*args.hidden2*2:].reshape([2,args.hidden2])

        gc2_w=gc2_parameters[:args.hidden2*args.hidden2*2].reshape([2,args.hidden2,args.hidden2])
        gc2_b=gc2_parameters[args.hidden2*args.hidden2*2:].reshape([2,args.hidden2])


        target_embs=[]
        for i in range(len(target_graph_adj_and_feat)):
            sub_adj=target_graph_adj_and_feat[i][0]
            sub_feat=target_graph_adj_and_feat[i][1]
            new_idx=target_new_idx[i]
            #target_embs.append(task_model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b)[new_idx])
            target_embs.append(task_model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b).mean(0))
        support_embeddings=torch.stack(target_embs,0)

        target_embs=[]
        for i in range(len(query_graph_adj_and_feat)):
            sub_adj=query_graph_adj_and_feat[i][0]
            sub_feat=query_graph_adj_and_feat[i][1]
            new_idx=query_new_idx[i]
            target_embs.append(task_model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b)[new_idx])
        query_embeddings=torch.stack(target_embs,0)
    
        
        if use_similarity:
            def _similarity(h1: torch.Tensor, h2: torch.Tensor):
                h1 = F.normalize(h1)
                h2 = F.normalize(h2)
                return h1 @ h2.t()

            similarity=_similarity(query_embeddings,support_embeddings)

            query_ys = torch.LongTensor([class_selected.index(i) for i in labels[id_query]]).numpy()
            query_ys_pred=torch.argmax(similarity,-1).detach().cpu().numpy()


            return metrics.accuracy_score(query_ys, query_ys_pred), f1(query_ys_pred, query_ys)
        
        
    
    
    
    support_features = normalize(support_embeddings).detach().cpu().numpy()
    query_features = normalize(query_embeddings).detach().cpu().numpy()

    support_ys = torch.LongTensor([class_selected.index(i) for i in labels[id_support]]).numpy()
    query_ys = torch.LongTensor([class_selected.index(i) for i in labels[id_query]]).numpy()

    clf = LogisticRegression(penalty='l2',
                             random_state=0,
                             C=1.0,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')
    clf.fit(support_features, support_ys)
    query_ys_pred = clf.predict(query_features)



    return metrics.accuracy_score(query_ys, query_ys_pred), f1(query_ys_pred, query_ys)


n_way = args.way
k_shot = args.shot
n_query = args.qry
meta_test_num = 50
meta_valid_num = 50

# Sampling a pool of tasks for validation/testing

from collections import defaultdict

results=defaultdict(dict)



pretrain_epochs=2000

save_time=datetime.datetime.now()

for repeat in range(1):
    

    names = ['Amazon_eletronics','dblp','cora-full','ogbn-arxiv']
    #names=['dblp']
    #names=['ogbn-arxiv']
    #names=['cora-full']
    #for dataset in ['dblp','Amazon_clothing','Amazon_eletronics']:
    for dataset in names:


        #dataset = args.dataset
        adj_sparse, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(dataset)


        #features/=features.max(0,keepdim=True)[0]

        adj=adj_sparse.to_dense()


        total_neighbors=adj.cpu().numpy()

        encoder=GCN_emb(nfeat=features.shape[1],
                nhid=args.hidden1,
                nclass=labels.max().item() + 1,
                dropout=0)

        encoder.load_state_dict(torch.load('./saved_models/GIN_{}_{}_epochs_class_ego_contrast_GCN.pth'.format(dataset,pretrain_epochs)))

        # encoder.load_state_dict(torch.load('/home/yzj/Desktop/DL_LZJ/pytorch/GCN_Test/models/GCN.pth'))

        task_model=GCN_dense(nfeat=args.hidden1,
                    nhid=args.hidden2,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)

        task_model.load_state_dict(torch.load('./saved_models/GIN_{}_{}_epochs_class_ego_contrast.pth'.format(dataset,pretrain_epochs)))




        if args.cuda:
            encoder.cuda()

            features = features.cuda()
            #adj = adj.cuda()
            labels = labels.cuda()
            adj_sparse=adj_sparse.cuda()
            degrees = degrees.cuda()
            task_model=task_model.cuda()

        emb_features=encoder(features,adj_sparse)

        for n_way in [5,10]:
            for k_shot in [3,5]:
                # Train model
                t_total = time.time()
                meta_train_acc = []

                for episode in range(args.train_episodes):

                    if args.test_mode!='LR':
                        id_support, id_query, class_selected = task_generator(id_by_class, class_list_train, n_way, k_shot, m_query=1,maximum_value_train_each_class=10)
                        acc_train, f1_train = train(class_selected, id_support, id_query, n_way, k_shot)
                        meta_train_acc.append(acc_train)

                        if episode > 0 and episode % 10 == 0:
                            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))


                #valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]
                test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, m_query=20) for i in range(meta_test_num)]



                meta_test_acc = []
                meta_test_f1 = []
                for idx in range(meta_test_num):
                    id_support, id_query, class_selected = test_pool[idx]








                    if args.test_mode!='LR':
                        acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                    else:
                        acc_test, f1_test = LR_test(class_selected, id_support, id_query, n_way, k_shot)
                    meta_test_acc.append(acc_test)
                    meta_test_f1.append(f1_test)

                    if idx%25==0 and idx!=0:
                        print("Task Num: {} Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(idx,np.array(meta_test_acc).mean(axis=0),
                                                                                np.array(meta_test_f1).mean(axis=0)))


                results[dataset]['{}-way {}-shot'.format(n_way,k_shot)]=[np.array(meta_test_acc).mean(axis=0),
                                                                np.array(meta_test_f1).mean(axis=0)]

                json.dump(results,open('./result_{}.json'.format(save_time),'w'))






    keys = ['5-way 3-shot', '5-way 5-shot', '10-way 3-shot', '10-way 5-shot']

    for name in names:
        print(name.replace('_','-')+'&' + '&'.join(keys)+'\\\\')
        print('Result' + '&' + '&'.join(
            ['{:.2f}'.format(one * 100) for one in [results[name][key][0] for key in keys]]) + '\\\\')
