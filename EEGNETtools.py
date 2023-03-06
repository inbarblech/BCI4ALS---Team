# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:21:12 2023

@author: marko
"""

import pandas as pd
import glob
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

def split_test_train_val(all_indexes_len):
    all_indexes = range(0,all_indexes_len,1)
    pick_indexes = int(all_indexes_len*0.7)
    train_indexes = random.sample(all_indexes, pick_indexes)
    
    test_val_indexes = np.setdiff1d(all_indexes, train_indexes)
    pick_val_indexes = int(len(test_val_indexes)*0.5)
    val_indexes = random.sample(sorted(test_val_indexes), pick_val_indexes)
    test_indexes = np.setdiff1d(test_val_indexes, val_indexes)
    return train_indexes, test_indexes, val_indexes

def read_input_x(path_,ch_nums,limited_number):
    first = True
    files_list = glob.glob(path_ + "\*")
    print(len(files_list))
    i = 0
    continue_ = False
    
    for file in files_list:
        data = pd.read_csv(file)
        first_ch = True
        for ch_num in ch_nums:
            if(len(data[str(ch_num)])<87): 
                continue_ = True
                break
            if(max(abs(data[str(ch_num)]))>0.018): 
                continue_ = True
                break
            ch_ = np.array(data[str(ch_num)][0:87]).reshape(1,1,87,1).astype('float32')
            if(first_ch):
                first_ch = False
                ch = ch_
            else:                
                ch = np.concatenate((ch, ch_), axis = 3)
        if(continue_)   :
            continue_ = False
            continue
        if(first):
            first = False
            x = ch
        else:
            x = np.vstack((x,ch)) 
        if(limited_number !=0 and i>=limited_number): break
        i+=1
    return x

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120
        
        # Layer 1
        #self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.conv1 = nn.Conv2d(1, 16, (1, 5), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        #self.fc1 = nn.Linear(4*2*7, 1)
        self.fc1 = nn.Linear(4*2*5, 3)
        

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        #x = x.reshape(-1, 4*2*7)
        x = x.reshape(-1, 4*2*5)
        x = torch.sigmoid(self.fc1(x))
        return x