import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from Relation_batch import *


# Recurrent neural network (many-to-many)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch=4, df=64, dk=32, Nr=4, use_LSTM=True):
        super(RNN, self).__init__()
        self.df = df
        self.dk = dk
        self.Nr = Nr
        self.atom_num = 37
        self.use_LSTM = use_LSTM
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch = batch
        self.hidden = self.init_hidden()

        # FC layer to joints and bones, to transform normalized atom feature to high dimension
        self.fc_joints = nn.Linear(6, self.df)
        self.fc_bones = nn.Linear(9, self.df)

        # Relation module on Atom features
        self.relation1 = RelationModule(n_relations = self.Nr, appearance_feature_dim=self.df,key_feature_dim = self.dk, num_parts=self.atom_num)
        self.FC_relation1 = nn.Linear(self.df, self.df)
        # transform atom feature matrix to class feature matrix, [N, appearance_feature_dim]
        #self.W_c = nn.Parameter(nn.init.uniform_(torch.empty(self.num_classes, self.atom_num), -1/np.sqrt(self.atom_num), 1/np.sqrt(self.atom_num)))
        self.W_c = nn.Linear(self.atom_num, self.num_classes, bias=False)
        # Relation module on class features
        self.relation2 = RelationModule(n_relations = self.Nr, appearance_feature_dim=self.df,key_feature_dim = self.dk, num_parts=self.num_classes)
        self.temp_relation2 = TemporalRelation(feat_dim = self.df) # temporal relation module to smooth the class-level feature, with the two t-1, t+1 attention feature and current feature
        self.FC_relation2 = nn.Linear(self.df, self.df)
        self.FC_temp2 = nn.Linear(self.df, self.df)

        # classifiers
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.fc_out = nn.ModuleList()
        if self.use_LSTM:
            self.lstm_out = nn.ModuleList()
        for i in range(self.num_classes):
            if self.use_LSTM:
                self.lstm_out.append(nn.LSTM(self.df, self.hidden_size, self.num_layers, batch_first=True))
                self.fc_out.append(nn.Linear(self.hidden_size, 1))
            else:
                self.fc_out.append(nn.Linear(self.df, 1))



    def init_hidden(self):
        '''
        Initialize hidden state parameter for backend LSTM
        h_0, c_0 of shape (num_layers * num_directions, batch, hidden_size):
        '''
        h0 = nn.init.xavier_uniform_(torch.empty(self.num_layers, 1, self.hidden_size)).cuda()
        c0 = nn.init.xavier_uniform_(torch.empty(self.num_layers, 1, self.hidden_size)).cuda()
        return (Variable(h0), Variable(c0))

    def atom_FC(self, x):
        '''
        Input: (batch_size, seq_length, 273) 273 = 20*6 + 17*9
        Output: (batch_size, seq_length, 37, df)
        '''
        batch_size, seq_length, feat_num = x.size()
        joints = torch.reshape(x[:,:,0:20*6], (batch_size, seq_length, 20, 6))
        bones = torch.reshape(x[:,:,20*6:], (batch_size, seq_length, 17, 9))
        # normalize the offset of joints and bones
        joints[:, :, :, 3:] = F.normalize(joints[:, :, :, 3:], p=1, dim=2)
        bones[:, : , :, 6:] = F.normalize(bones[:, : , :, 6:], p=1, dim=2)
        return self.tanh(torch.cat((self.fc_joints(joints), self.fc_bones(bones)), -2))

    def forward(self, x):
        '''
        Take batch of frames as input to compute attention weight
        '''
        atom_feats = self.atom_FC(x) # (batch_size, seq_length, 37, df)
        relation_feats = [] # relation features
        T = atom_feats.size()[1] # sequence length
        for i in range(T//self.batch+1):
            #print('batch: %d/%d' % (i*self.batch, T))
            start_t = i*self.batch  # start frame of batch forward
            end_t = min(((i+1)*self.batch, T))  # end frame of batch forward
            if start_t == end_t: # start frame == end_frame, do not need another batch
                break
            atoms = self.relation1((atom_feats[0,start_t:end_t,:,:])) # [B, N_atoms, df]
            atoms = self.relu(self.FC_relation1(atoms))               # [B, N_atoms, df]
            atoms = self.W_c(atoms.permute(0, 2, 1).contiguous())                # [B, df, N_classes]
            atoms = atoms.permute(0, 2, 1).contiguous()                 # [B, N_classes, df]
            class_feats = self.relation2((atoms))           # [B, N_classes, df]
            #print(class_feats.size())
            relation_feats.append(class_feats)

        relation_feature = torch.cat(relation_feats,dim=0)    # [t, N_classes, df]
        relation_feature = self.relu(self.FC_relation2(relation_feature)) # FC after RM
        relation_feature = self.tanh(self.FC_temp2(self.temp_relation2(relation_feature))) # FC after TM
        relation_feature = relation_feature.view(1, atom_feats.size()[1], self.num_classes, self.df) # [batch_size, t, num_classes, df]

        pred = []
        for i in range(self.num_classes):
            if self.use_LSTM:
                out, _ = self.lstm_out[i](relation_feature[:,:,i,:])
                #out, _ = self.lstm_out[i](relation_feature[:,:,i,:], self.hidden)
                pred.append(self.tanh(self.fc_out[i](out)))
            else:
                pred.append(self.tanh(self.fc_out[i](relation_feature[:,:,i,:])))
        out = torch.stack(pred, dim=-1) # [batch_size, t, 1, num_classes]
        out = out.view(1, atom_feats.size()[1], -1)
        return out
