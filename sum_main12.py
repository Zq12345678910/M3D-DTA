import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from torch_geometric.nn import GINConv, global_add_pool,global_mean_pool
from math import pi as PI
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.autograd import Variable
from loss import get_mse,all_metrics
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
import argparse
from torch_geometric.data import Data as DATA
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from collections import OrderedDict
import rdkit.Chem as Chem
import networkx as nx
from torch.utils.data import Dataset
from rdkit.Chem import AllChem
import os
from pathlib import Path
import torch
from rdkit import Chem

class GIN_Layer(nn.Module):
    def __init__(self, input_dim=64,hidden_dim=128):
        super(GIN_Layer, self).__init__()
        nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1=GINConv(nn1)
        self.act1=nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
    def forward(self,x,edge_index):
        x=self.conv1(x,edge_index)
        x=self.act1(x)
        x=self.bn1(x)
        return x
class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, cutoff, dropout_rate):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.mlp = Sequential(
            Linear(num_gaussians, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.lin = Linear(hidden_channels, hidden_channels, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v, dist, dist_emb, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff).view(-1, 1) + 1.0)
        W = self.mlp(dist_emb) * C
        # W = self.dropout(W)
        v = self.lin(v)
        e = v[j] * W
        return e
class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate):
        super(update_v, self).__init__()
        self.act = nn.ReLU()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v, e, edge_index):
        _, i = edge_index
        out = scatter(e, i, dim=0)
        out = self.lin1(out)
        out = self.act(out)
        # out = self.dropout(out)
        out = self.lin2(out)
        if len(out) != len(v):
            zeros = torch.zeros_like(v)
            zeros[:len(out), :] = out
            out = zeros
        return v + out
class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate):
        super(update_u, self).__init__()
        self.lin1 = Linear(3*hidden_channels, hidden_channels)
        self.act = nn.ReLU()
        self.lin2 = Linear(hidden_channels, hidden_channels)
        # self.lin3 = Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = scatter(v, batch, dim=0)  #
        v = self.lin2(v)
        v = self.act(v)
        # v = self.dropout(v)
        return v
class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)  # start 부터 stop까지 일정 간격의 50개의 number 생성
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
class SchNet_hidden(torch.nn.Module):
    def __init__(self, cutoff, num_layers, hidden_channels, num_gaussians, dropout_rate):
        super(SchNet_hidden, self).__init__()
        self.cutoff = cutoff
        self.drug_emb1 = nn.Embedding(100, hidden_channels)
        self.drug_emb2 = nn.Embedding(10, hidden_channels)
        self.drug_emb3 = nn.Embedding(10, hidden_channels)
        self.drug_emb4 = nn.Embedding(10, hidden_channels)
        self.lin1 = Linear(4 * hidden_channels, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)
        self.update_vs = torch.nn.ModuleList(
            [update_v(hidden_channels, dropout_rate) for _ in range(num_layers)])
        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_gaussians, cutoff, dropout_rate) for _ in range(num_layers)])
        self.update_u = update_u(hidden_channels, dropout_rate)

    def forward(self, z, pos, batch):
        z = z.long()
        z0, z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        z0 = self.drug_emb1(z0)
        z1 = self.drug_emb2(z1)
        z2 = self.drug_emb3(z2)
        z3 = self.drug_emb4(z3)
        z = torch.cat([z0, z1, z2, z3], dim=1)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=100)  # Based on pos, return edge_index within cutoff
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)  # return distance between edge
        dist_emb = self.dist_emb(dist)
        v = self.lin1(z)
        x=[]
        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v, dist, dist_emb, edge_index)
            v = update_v(v, e, edge_index)
            x.append(v)
        u = self.update_u(torch.cat(x,dim=-1), batch)
        return u
class BiLSTM_hidden(nn.Module):
    def __init__(self, lstm_dim=64, hidden_dim=128,dropout_rate=0.1):
        super(BiLSTM_hidden, self).__init__()
        self.protein_lstm1 = nn.LSTM(lstm_dim, hidden_dim, 2, batch_first=True, bidirectional=True,dropout=dropout_rate)

    def forward(self,protein):
        protein1,_=self.protein_lstm1(protein)
        return protein1
class GIN_hidden(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GIN_hidden, self).__init__()
        self.GIN1 = GIN_Layer(input_dim=input_dim, hidden_dim=hidden_dim)
        self.GIN2 = GIN_Layer(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.GIN3 = GIN_Layer(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.lin1=nn.Linear(3*hidden_dim,hidden_dim)
    def forward(self,x,edge_index,batch):
        x1 = self.GIN1(x, edge_index)
        x2 = self.GIN2(x1, edge_index)
        x3 = self.GIN3(x2, edge_index)
        x4=torch.cat([x1,x2,x3],dim=-1)
        x = global_mean_pool(x4, batch)
        x=self.lin1(x)
        return x
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            # nn.ELU(),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)

class Read_Layer(nn.Module):
    def __init__(self,smile_dim=128,lstm_dim=2400,protein_dim=128,dropout_rate=0.25):
        super(Read_Layer, self).__init__()
        self.smile_lin1=nn.Linear(smile_dim, smile_dim*2)
        self.lstm_lin1 = nn.Linear(128,128)
        self.lstm_lin2 = nn.Linear(128, 128)
        self.graph_lin1 = nn.Linear(protein_dim,smile_dim*2)

        self.lin_s1=nn.Linear(smile_dim*2, smile_dim*10)
        self.lin_s2=nn.Linear(smile_dim * 10, smile_dim)
        self.lin_s3 = nn.Linear(smile_dim, 1)
        self.act1 = nn.ReLU()
        self.act = nn.ELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(0.5)
        self.att=Attention(128,16)

    def forward(self, smile,lstm,graph):
        lstm=self.act1(lstm)
        lstm=self.dropout2(lstm)
        lstm=self.lstm_lin1(lstm)
        lstm = self.act1(lstm)
        lstm = self.dropout2(lstm)
        lstm=self.lstm_lin2(lstm)

        p=self.att(torch.cat([lstm.unsqueeze(1),graph.unsqueeze(1)],dim=1))+lstm+graph

        features = torch.cat([smile,p],dim=1)

        features=self.lin_s1(features)
        features=self.act1(features)
        features=self.dropout1(features)

        features = self.lin_s2(features)
        features = self.act1(features)

        features = self.lin_s3(features)
        return features

class GINATT_Model(nn.Module):
    def __init__(self,protein_vocab=25,input_dim=64,hidden_dim=128,lstm_len=2600,dropout_rate=0.25,device=None):
        super(GINATT_Model,self).__init__()
        self.lstm_len = lstm_len
        self.device = device
        self.lstm_protein_emb1=nn.Embedding(protein_vocab, input_dim, padding_idx=0)
        # self.lstm_protein_emb2 = nn.Embedding(protein_vocab*protein_vocab, input_dim, padding_idx=0)
        self.BiLSTM1 = BiLSTM_hidden(lstm_dim=input_dim,hidden_dim=hidden_dim,dropout_rate=dropout_rate)
        # self.BiLSTM2 = BiLSTM_hidden(lstm_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.pooling1=nn.AdaptiveMaxPool1d(1)
        self.lin1=nn.Linear(args.protein_up,hidden_dim)

        self.SchNet_hidden1 = SchNet_hidden(cutoff=5.0, num_layers=3, hidden_channels=hidden_dim,
                                           num_gaussians=50, dropout_rate=dropout_rate)

        self.graph_protein_emb1 = nn.Embedding(protein_vocab, input_dim)
        self.GIN_hidden1 = GIN_hidden(input_dim=input_dim, hidden_dim=hidden_dim)

        # self.readout1=Read_Layer(smile_dim=hidden_dim,lstm_dim=lstm_len,protein_dim=hidden_dim,dropout_rate=dropout_rate)
        self.readout1 = Read_Layer(smile_dim=hidden_dim, lstm_dim=hidden_dim, protein_dim=hidden_dim,
                                   dropout_rate=dropout_rate)
    def ngram(self,x,lens=2):
        xin_x = torch.zeros_like(x)
        a,b=xin_x.size()
        zeros=torch.zeros(a,b+lens-1)
        zeros[:,:b]=x
        x=zeros
        if lens==2:
            he=torch.Tensor([22,1])
            he=he.view(1,2)
        elif lens==3:
            he = torch.Tensor([22*22,22, 1])
            he = he.view(1, 3)
        else:
            he=None
        xin_x = torch.cat(
            list(map(lambda i: torch.sum(x[:, i:i + lens] * he, dim=-1, keepdim=True, dtype=torch.long), range(b))),
            dim=-1).to(args.device)
        # xin_x = torch.cat([torch.sum(x[:, i:i + lens] * he, dim=-1,keepdim=True,dtype=torch.long) for i in range(b)],dim=-1).to(args.device)
        return xin_x

    def bi_fetch(self, rnn_outs, seq_lengths):
        a, b, _ = rnn_outs.size()
        batch_size, max_len = a, b
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).to(args.device))
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).to(args.device))
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).to(args.device) * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).to(args.device)

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index.to(torch.long))  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self,data1,data2,data3):
        smile_x,smile_pos,smile_batch=data1.x,data1.pos,data1.batch
        smile_xs = self.SchNet_hidden1(smile_x,smile_pos,smile_batch)

        lstm_x,lens=data2.x,data2.lens
        lstm_x = torch.reshape(lstm_x, (-1, self.lstm_len))
        # lstm_x1 =self.ngram(x=lstm_x,lens=2)
        lstm_x = self.lstm_protein_emb1(lstm_x)
        # lstm_x1 = self.lstm_protein_emb2(lstm_x1)
        lstm_x=self.BiLSTM1(lstm_x)
        # lstm_x1=self.BiLSTM2(lstm_x1)
        # lstm_x=lstm_x+lstm_x1
        lstm_xs=self.pooling1(lstm_x).squeeze(-1)
        # lstm_xs=torch.cat([lstm_x[:,-1,:128],lstm_x[:,0,128:]],dim=-1)
        lstm_xs=self.lin1(lstm_xs)


        graph_x,graph_edge_index,graph_batch = data3.x,data3.edge_index,data3.batch
        graph_x = self.graph_protein_emb1(graph_x)
        graph_xs=self.GIN_hidden1(graph_x,graph_edge_index,graph_batch)

        output = self.readout1(smile_xs,lstm_xs,graph_xs)
        return output
#蛋白质定长编码
def protein(seq=None,up=1200):
    protein_embbed = []
    for i in range(len(seq)):
        protein_embbed.append(lstm_table[seq[i]])

    if len(protein_embbed)>up:
        protein_embbed=protein_embbed[:up]
    elif len(protein_embbed)==up:
        pass
    else:
        padding = [0] * (up- len(protein_embbed))
        protein_embbed.extend(padding)
    return protein_embbed
#蛋白质图特征编码
def Graph_protein(seq=None):
    protein_embbed = []
    for i in range(len(seq)):
        protein_embbed.append(lstm_table[seq[i]])
    return protein_embbed
#蛋白质图邻接矩阵
def target_to_graph(target_key, contact_dir):
    target_edge_index = []
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_edge_index
#药物图特征+邻接矩阵+三维坐标
def SMILES(smiles=None):
    mol=Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=1, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    coord = mol.GetConformer(id=0).GetPositions()
    x = []
    pos=[]
    for i, atom in enumerate(mol.GetAtoms()):
        x.append([atom.GetAtomicNum(),atom.GetDegree(),atom.GetTotalNumHs(),atom.GetImplicitValence()])
        pos.append(coord[i])
    x = np.array(x)
    pos=np.array(pos)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    Graph = nx.Graph(edges).to_directed()

    adj_matrix = np.zeros((int(mol.GetNumAtoms()), int(mol.GetNumAtoms())))
    for start, end in Graph.edges:
        adj_matrix[start, end] = 1
    adj_matrix += np.matrix(np.eye(adj_matrix.shape[0]))

    row, col = np.where(adj_matrix > 0)
    edge_index = []
    for i, j in zip(row, col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    return torch.tensor(x), torch.tensor(edge_index),torch.tensor(pos,dtype=torch.float)

def data_process(df):
    print(len(set(df['compound_iso_smiles'])),len(set(df['target_sequence'])),len(set(df['target_key'])))

    smiles_x, smiles_index, smiles_pos = {}, {}, {}
    for smile in set(df['compound_iso_smiles']):
        smiles_x[smile], smiles_index[smile], smiles_pos[smile] = SMILES(smiles=smile)

    lstm_protein_x, graph_protein_x, graph_protein_index={}, {}, {}
    df.drop_duplicates(subset=['target_key'], inplace=True)
    print(len(df))
    for key, seq in zip(df['target_key'], df['target_sequence']):
        lstm_protein_x[key]=torch.tensor(protein(seq=seq,up=args.protein_up))
        graph_protein_x[key] = torch.tensor(Graph_protein(seq=seq))
        graph_protein_index[key]=torch.tensor(target_to_graph(target_key=key,contact_dir=args.pconsc4_path))
    return lstm_protein_x, graph_protein_x, graph_protein_index, smiles_x, smiles_index, smiles_pos

def create_dataset1():
    train_fold = json.load(open(args.train_fold_txt))  # len=5
    train_fold = [ee for e in train_fold for ee in e]  # davis len=25046
    valid_fold = json.load(open(args.test_fold_txt))  # davis len=5010
    ligands = json.load(open(args.ligands_can_txt), object_pairs_hook=OrderedDict)  # davis len=68
    proteins = json.load(open(args.proteins_txt), object_pairs_hook=OrderedDict)  # davis len=442
    affinity = pickle.load(open(args.Y, "rb"), encoding='latin1')  # davis len=68

    drugs = []
    prots = []
    prot_keys = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)

    if args.dataset_name == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    opts = ['train', 'test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)  # not NAN
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]  # Train fold contains index information for both drug and target
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open('{}/{}_{}.csv'.format(args.data_save_path,args.dataset_name,opt), 'w') as f:
            f.write('compound_iso_smiles,target_key,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], args.pconsc4_path):  # Check if there are aln and pconsc4 files
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')  # csv format
    print('\ndataset:', args.dataset_name)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)):', len(set(drugs)),'---len(set(prots)):', len(set(prots)))
    print('finish',args.dataset_name,' csv file')

def valid_target(key, dataset):
    contact_file = os.path.join(dataset, key + '.npy')
    if os.path.exists(contact_file):
        return True
    else:
        return False

class DTADataset(Dataset):
    def __init__(self,typel, lstm_protein_x,graph_protein_x,graph_protein_index,smiles_x, smiles_index,smiles_pos):
        if typel == 'train':
            df = pd.read_csv(args.train_csv)
            self.smiles_data_path = args.train_smiles_data
            self.lstm_proteins_data_path = args.train_lstm_proteins_data
            self.graph_proteins_data_path = args.train_graph_proteins_data
        else:
            df = pd.read_csv(args.test_csv)
            self.smiles_data_path = args.test_smiles_data
            self.lstm_proteins_data_path = args.test_lstm_proteins_data
            self.graph_proteins_data_path = args.test_graph_proteins_data
        self.smiles_data, self.lstm_proteins_data, self.graph_proteins_data = [], [], []
        self.process(df, lstm_protein_x,graph_protein_x,graph_protein_index,smiles_x, smiles_index,smiles_pos)
    def process(self, df, lstm_protein_x,graph_protein_x,graph_protein_index,smiles_x, smiles_index,smiles_pos):
        if not self.smiles_data_path.exists() or not self.lstm_proteins_data_path.exists() or not self.graph_proteins_data_path.exists():
            for i in tqdm(range(len(df))):
                smile = df.loc[i, 'compound_iso_smiles']
                key = df.loc[i, 'target_key']
                y = df.loc[i, 'affinity']

                smile_x = smiles_x[smile]
                smile_edge_index = smiles_index[smile]
                smile_pos = smiles_pos[smile]
                lstm_x = lstm_protein_x[key]
                graph_x = graph_protein_x[key]
                graph_edge_index = graph_protein_index[key]
                if len(lstm_x)<=len(graph_x):
                    lenth=len(lstm_x)
                else:
                    lenth=len(graph_x)

                smile_data = DATA(x=torch.Tensor(smile_x),
                                  edge_index=torch.LongTensor(smile_edge_index).transpose(1, 0),
                                  pos=smile_pos,
                                  y=torch.FloatTensor([y]))
                lstm_data = DATA(x=torch.Tensor(lstm_x),lens=torch.Tensor([lenth]))
                graph_data = DATA(x=torch.Tensor(graph_x),
                                  edge_index=torch.LongTensor(graph_edge_index).transpose(1, 0))
                self.smiles_data.append(smile_data)
                self.lstm_proteins_data.append(lstm_data)
                self.graph_proteins_data.append(graph_data)
            torch.save(self.smiles_data, self.smiles_data_path)
            torch.save(self.lstm_proteins_data, self.lstm_proteins_data_path)
            torch.save(self.graph_proteins_data, self.graph_proteins_data_path)
        else:
            self.smiles_data = torch.load(self.smiles_data_path)
            self.lstm_proteins_data=torch.load(self.lstm_proteins_data_path)
            self.graph_proteins_data=torch.load(self.graph_proteins_data_path)
    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        return self.smiles_data[idx],self.lstm_proteins_data[idx],self.graph_proteins_data[idx]

def train_model(model=None,train_loader=None,epoch=None,optimizer=None,device=None):
    model.train()
    loss_fn = torch.nn.MSELoss()
    for data1, data2, data3 in tqdm(train_loader, desc='train_epoch{}'.format(epoch), ncols=100):
        data1 = data1.to(device)
        data2 = data2.to(device)
        data3 = data3.to(device)
        optimizer.zero_grad()
        output = model(data1, data2, data3)
        loss = loss_fn(output.float(), data1.y.view(-1, 1).float().to(device)).float()
        loss.backward()
        optimizer.step()
    return model

def val_model(model=None,valid_loader=None,epoch=None,device=None):
    model.eval()
    eval_total_preds = torch.Tensor()
    eval_total_labels = torch.Tensor()
    with torch.no_grad():
        for data1, data2, data3 in tqdm(valid_loader, desc='valid_epoch{}'.format(epoch), ncols=100):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)
            eval_output = model(data1, data2, data3)
            eval_total_preds = torch.cat((eval_total_preds, eval_output.cpu()), 0)
            eval_total_labels = torch.cat((eval_total_labels, data1.y.view(-1, 1).cpu()), 0)
    return eval_total_labels.numpy().flatten(), eval_total_preds.numpy().flatten()

def test_model(model=None,test_loader=None,device=None):
    model.eval()
    test_total_preds = torch.Tensor()
    test_total_labels = torch.Tensor()
    with torch.no_grad():
        for data1, data2, data3 in tqdm(test_loader, desc='test_epoch{}'.format(0), ncols=100):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)
            test_output = model(data1, data2, data3)
            test_total_preds = torch.cat((test_total_preds, test_output.cpu()), 0)
            test_total_labels = torch.cat((test_total_labels, data1.y.view(-1, 1).cpu()), 0)
    return test_total_labels.numpy().flatten(), test_total_preds.numpy().flatten()



parser = argparse.ArgumentParser()
#参数
parser.add_argument('--dataset_name', default='davis')#0.0001
parser.add_argument('--seed', type=int, default=19)
parser.add_argument('--batch_size', type=int,default=128)#32
parser.add_argument('--epochs', type=int , default=1500)#100
parser.add_argument('--lr', default=0.0001)#0.0001
parser.add_argument('--device', default=0)
parser.add_argument('--num_workers', default=1)
parser.add_argument('--fold', default=3)
parser.add_argument('--protein_up', default=1200)
parser.add_argument('--smiles_up', default=400)
#数据分割
parser.add_argument('--create_dataset', default='create_dataset1')
#初始数据
parser.add_argument('--train_fold_txt', default='folds/train_fold_setting1.txt')
parser.add_argument('--test_fold_txt', default='folds/test_fold_setting1.txt')
parser.add_argument('--ligands_can_txt', default='ligands_can.txt')
parser.add_argument('--proteins_txt', default='proteins.txt')
parser.add_argument('--Y', default='Y')
parser.add_argument('--pconsc4_path', default='pconsc4')
#保存数据
parser.add_argument('--data_save_path', default='save1')
parser.add_argument('--train_csv', default='train')
parser.add_argument('--test_csv', default='test')
parser.add_argument('--train_smiles_data', default='train')
parser.add_argument('--train_lstm_proteins_data', default='train')
parser.add_argument('--train_graph_proteins_data', default='train')
parser.add_argument('--test_smiles_data', default='test')
parser.add_argument('--test_lstm_proteins_data', default='test')
parser.add_argument('--test_graph_proteins_data', default='test')
parser.add_argument('--model_save_path', default='model')
args = parser.parse_args()
lstm_table = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14,'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20, 'X':21}
args.train_fold_txt = Path('/tmp/GINATT/data/{}/{}'.format(args.dataset_name,args.train_fold_txt))
args.test_fold_txt = Path('/tmp/GINATT/data/{}/{}'.format(args.dataset_name,args.test_fold_txt))
args.ligands_can_txt = Path('/tmp/GINATT/data/{}/{}'.format(args.dataset_name,args.ligands_can_txt))
args.proteins_txt = Path('/tmp/GINATT/data/{}/{}'.format(args.dataset_name,args.proteins_txt))
args.Y = Path('/tmp/GINATT/data/{}/{}'.format(args.dataset_name,args.Y))
args.pconsc4_path=Path('/tmp/GINATT/data/{}/{}'.format(args.dataset_name,args.pconsc4_path))

args.data_save_path = Path('/tmp/GINATT/{}/{}'.format(args.data_save_path,args.dataset_name))
if not args.data_save_path.exists():
    os.makedirs(args.data_save_path)
args.train_csv = Path('{}/{}_{}.csv'.format(args.data_save_path,args.dataset_name,args.train_csv))
args.test_csv= Path('{}/{}_{}.csv'.format(args.data_save_path,args.dataset_name,args.test_csv))
if args.create_dataset == 'create_dataset1':
    if not args.test_csv.exists() or not args.train_csv.exists():
        create_dataset1()
    args.train_smiles_data = Path('{}/{}_smiles_data.pt'.format(args.data_save_path, args.train_smiles_data))
    args.train_lstm_proteins_data = Path(
        '{}/{}_lstm_proteins_data.pt'.format(args.data_save_path, args.train_lstm_proteins_data))
    args.train_graph_proteins_data = Path(
        '{}/{}_graph_proteins_data.pt'.format(args.data_save_path, args.train_graph_proteins_data))
    args.test_smiles_data = Path('{}/{}_smiles_data.pt'.format(args.data_save_path, args.test_smiles_data))
    args.test_lstm_proteins_data = Path(
        '{}/{}_lstm_proteins_data.pt'.format(args.data_save_path, args.test_lstm_proteins_data))
    args.test_graph_proteins_data = Path(
        '{}/{}_graph_proteins_data.pt'.format(args.data_save_path, args.test_graph_proteins_data))

    df = pd.read_csv(args.train_csv)
    df1 = pd.read_csv(args.test_csv)
    if not args.train_smiles_data.exists() or not args.train_lstm_proteins_data.exists() or not args.train_graph_proteins_data.exists()\
            or not args.test_smiles_data.exists() or not args.test_lstm_proteins_data.exists() or not args.test_graph_proteins_data.exists():
        lstm_protein_x, graph_protein_x, graph_protein_index, smiles_x, smiles_index, smiles_pos = data_process(pd.concat([df,df1]))
    else:
        lstm_protein_x, graph_protein_x, graph_protein_index, smiles_x, smiles_index, smiles_pos = None, None, None, None, None, None
    train_dataset = DTADataset(typel='train', lstm_protein_x=lstm_protein_x, graph_protein_x=graph_protein_x,
                               graph_protein_index=graph_protein_index, smiles_x=smiles_x, smiles_index=smiles_index,
                               smiles_pos=smiles_pos)
    test_dataset = DTADataset(typel='test', lstm_protein_x=lstm_protein_x, graph_protein_x=graph_protein_x,
                              graph_protein_index=graph_protein_index, smiles_x=smiles_x, smiles_index=smiles_index,
                              smiles_pos=smiles_pos)
    print('Train size:', len(train_dataset))
    print('Test size:', len(test_dataset))


model = GINATT_Model(protein_vocab=22, input_dim=128, hidden_dim=128, lstm_len=args.protein_up,
                         dropout_rate=0.2, device=args.device)
model = model.to(args.device)
args.model_save_path = Path('{}/{}'.format(args.data_save_path,args.model_save_path))
if not args.model_save_path.exists():
    os.makedirs(args.model_save_path)
# SchNet_params = list(map(id, model.SchNet_hidden1.parameters()))
# base_params = filter(lambda p: id(p) not in SchNet_params,
#                      model.parameters())
# print(SchNet_params)
# print(base_params)
# optimizer = torch.optim.Adam(
#             [{'params': base_params},
#             {'params': model.SchNet_hidden1.parameters(), 'lr': args.lr*2}], lr=args.lr, weight_decay=5e-4)
# schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-4, last_epoch=-1)

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

optimizer = Adam(model.parameters(), lr=args.lr)

best_model = None
best_mse = 10000000
best_epoch = 0
ci, rm2, mse = 0, 0, 0
for epoch in range(1, args.epochs + 1):
    print('Training on {} samples...'.format(len(train_dataset)))
    print('Testing on {} samples...'.format(len(test_dataset)))

    train_model(model=model, train_loader=train_loader, epoch=epoch, optimizer=optimizer, device=args.device)
    L, P = val_model(model=model, valid_loader=test_loader, epoch=epoch, device=args.device)
    val = get_mse(L, P)
    if val < best_mse:
        best_mse = val
        best_epoch = epoch
        # L1, P1 = test_model(model=model, test_loader=test_loader, device=args.device)
        ci, rm2 = all_metrics(L, P)
        torch.save(model, '{}/{}_{}-epoch_{}_ci_{}_rm2_{}_mse_{}.pth'.format(args.model_save_path,
                                                                             args.dataset_name, args.fold, best_epoch,
                                                                             ci, rm2, best_mse))
        print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
    else:
        print('current mse: ', val, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse)

