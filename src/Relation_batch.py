'''
Relation modules:
    Spatial Relation Module - Relation Module
    Relation Unit of Spatial Relation Module for atom/class level features - RelationUnit
    Temporal Relation Module - TemporalRelation
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RelationModule(nn.Module):
    '''
    Relation module for atom feature or class-level feature, decided by 'isAtom'.
    atom feature            -       isAtom = True
    class level feature     -       isAtom = False

    Difference is how to compute geometric attention weight.
    atom feature            -       take in geo_feature, transform by (pos_transform, sigmoid), then apply linear transform by WG
    class level feature     -       just a [N, N] learnable parameter, N is the same with geo_feature_dim, which is the number of classes
    '''
    def __init__(self,n_relations = 4, appearance_feature_dim=64,key_feature_dim = 8, num_parts=37):
        super(RelationModule, self).__init__()
        self.Nr = n_relations
        self.N = num_parts
        self.value_feature_dim = int(appearance_feature_dim/ self.Nr)
        #self.isAtom = isAtom
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, self.value_feature_dim, self.N))

    def forward(self, f_a):
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                concat = self.relation[N](f_a)
                isFirst=False
            else:
                concat = torch.cat((concat, self.relation[N](f_a)), -1)
        return concat+f_a

class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim,key_feature_dim, value_feature_dim, N):
        super(RelationUnit, self).__init__()
        self.dim_k = key_feature_dim
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=False)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=False)
        self.WV = nn.Linear(appearance_feature_dim, value_feature_dim, bias=False)
        self.w_fix = nn.Parameter(nn.init.uniform_(torch.empty(N, N), -1/np.sqrt(N), 1/np.sqrt(N)))

    def forward(self, f_a):
        '''
        Input dim: [B, N, df]
            B - batch_size frames
            N - atom/class numbers
            df - relation feature dimension
        Output dim: [B, N, df]
        '''
        B, N, _ = f_a.size()
        mask = torch.eye(N).byte().to(device)

        w_k = self.WK(f_a) # [B, N, dim_k]
        w_k = w_k.view(B, N,1,self.dim_k)
        w_q = self.WQ(f_a) # [B, N, dim_k]
        w_q = w_q.view(B, 1,N,self.dim_k)

        scaled_dot = torch.sum((w_k*w_q),-1 ) # dot product of Key, Query matrix
        scaled_dot = scaled_dot / np.sqrt(self.dim_k) # [B, N, N]
        w_mn = scaled_dot.view(B,N,N)  # w_mn is the attention weight of key m and query n
        w_mn.masked_fill_(mask, 0) # set diagonal element w_nn to 0
        w_mn = nn.functional.softmax(w_mn, dim=-1)
        w_mn = w_mn + self.w_fix
        w_mn.masked_fill_(mask, 0) # set diagonal element w_nn to 0
        #w_mn = nn.functional.softmax(w_mn, dim=-1)

        w_v = self.WV(f_a)  #[B, N, dv]

        w_mn = w_mn.view(B,N,N,1) # [B,N,N,1]
        w_v = w_v.view(B, N, 1, -1) # [B, N, 1, dv]
        #w_v = w_v.repeat(N,1,1) # [N, N, dv]

        output = w_mn*w_v # [B, N, N, dv]
        #print('attention out dim:', output.size())

        output = torch.sum(output,-2)
        return output # [B,N,v_dim]

class TemporalRelation(nn.Module):
    def __init__(self, feat_dim, time_window=1):
        super(TemporalRelation, self).__init__()
        self.time_window = time_window
        self.feat_dim = feat_dim
        self.WT = nn.Linear(self.feat_dim, self.feat_dim, bias=False)

    def forward(self, feats):
        # feats - [t, num_classes, df]
        relation_feature = []
        att_feats = self.WT(feats)  # [t, num_classes, df]
        for t in range(0, att_feats.size()[0], 1):
            if t<self.time_window:
                prev = att_feats[0,:,:]
            else:
                prev = att_feats[t-1,:,:]
            if t== (att_feats.size()[0]-1):
                next = att_feats[t,:,:]
            else:
                next = att_feats[t+1,:,:]
            relation_feature.append(prev+next)
        relation_feature = torch.stack(relation_feature,dim=0) # [t, num_classes, df]
        return relation_feature/2 + feats
