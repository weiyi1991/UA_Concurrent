'''
Customize our dataset loader

Author: Yi Wei
'''
import torch
from torch.utils.data.dataset import Dataset
import os
import pickle
#from utils.data_proc_UCLA import *

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f, encoding='latin1')
    return ret_di

class ConActDataset(Dataset):
    '''
    Customized Dataset class for concurrent action recognition dataset.
    '''
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir # folder
        self.train = train # flag for training
        seqList = os.listdir(self.root_dir) # sequence folder name
        seqList.sort()
        #print(seqList)
        self.seq_list = [] # sequence file name list for training/testing set
        # get training/testing data and label list, even # for training, odd # for testing
        if self.train:
            self.seq_list = [seq for seq in seqList if (self._get_seq_num(seq) % 2) == 0]
        else:
            self.seq_list = [seq for seq in seqList if (self._get_seq_num(seq) % 2) == 1]


    def __getitem__(self, index):
        data = load_dict(os.path.join(self.root_dir, self.seq_list[index]))
        return data['feat'], data['label']

    def __len__(self):
        return len(self.data_list)

    def _get_seq_num(self, file_name):
        # return the sequence id num of a sequence, e.g.: 'sequence_001' -> 1
        seq_name = file_name.split('.')[0]
        return int(seq_name.split('_')[-1])

if __name__ == '__main__':
    data_root = '/home/yi/PycharmProjects/relation network/data'
    ds = ConActDataset(data_root)
    d1 = ds.__getitem__(2)
    input, label = d1
    print(input, label)
    print(input.shape, label.shape)
