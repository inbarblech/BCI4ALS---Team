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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

channels_names = ['C3','C4','Cz','FC1','FC2','FC5','FC6','CP1','CP2','CP5','CP6']
#channels_names = ['0','1','2','3','4','5','6','7','8','9','10']
chosen_channels = channels_names
target_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\target"
other_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\other"
gf_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\gap filler"
AVERAGE_INPUT=1

def evaluate(model, X, Y, params = ["acc"]):
    results = []
    
        
    inputs = Variable(torch.from_numpy(X))
    predicted = model(inputs)
    
    predicted = predicted.data.cpu().numpy()
    
    for param in params:
        if param == 'acc':
            l = []
            res = []
            for y,p in zip(Y,predicted):
                l.append(np.argmax(y))
                res.append(np.argmax(p))
            results.append(accuracy_score(l, res))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted, multi_class='ovr'))
            results.append(0)
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
    return results


def split_test_train_val(all_indexes_len):
    all_indexes = range(0,all_indexes_len,1)
    pick_indexes = int(all_indexes_len*0.7)
    train_indexes = random.sample(all_indexes, pick_indexes)
    
    test_val_indexes = np.setdiff1d(all_indexes, train_indexes)
    pick_val_indexes = int(len(test_val_indexes)*0.5)
    val_indexes = random.sample(sorted(test_val_indexes), pick_val_indexes)
    test_indexes = np.setdiff1d(test_val_indexes, val_indexes)
    return train_indexes, test_indexes, val_indexes

def average_input_x(input_x):
    av_len = int(input_x.shape[2]/3)
    averaged_input_x = np.empty(input_x.shape[0]*av_len*input_x.shape[3]).reshape(input_x.shape[0],1,av_len,input_x.shape[3])
    for sample in range(input_x.shape[0]):
        for i in range (av_len):
            for ch in range(input_x.shape[3]):
                values = input_x[sample, 0, i*3: (i+1)*3, ch]
                averaged_input_x[sample][0][i][ch] = np.mean(values)
    
    return averaged_input_x

def read_input_x(path_,ch_nums,limited_number):
    files_list = glob.glob(path_ + "\*")
    data = np.load(files_list[0])
    for file in files_list[1:]:
        data_next = np.load(file) 
        data = np.concatenate((data,data_next), axis = 0)
    
    if limited_number!=0:
        return np.nan_to_num(data[:limited_number,:,:,:])
    else:
        return np.nan_to_num(data)

def get_data_for_EEGNet():
    target_x = read_input_x(target_path,chosen_channels,0)   
    other_x = read_input_x(other_path,chosen_channels, 0)   
    #limit gap fillers to the number of targets/others
    gf_x = read_input_x(gf_path,chosen_channels ,max(target_x.shape[0], other_x.shape[0]))   
    print(target_x.shape, other_x.shape, gf_x.shape)

    if(AVERAGE_INPUT >1):
        print("Average")
        target_x = average_input_x(target_x)
        other_x = average_input_x(other_x)
        gf_x = average_input_x(gf_x)
        print(target_x.shape, other_x.shape, gf_x.shape)
    else: print('No average')

    target_train_x, target_test_x, target_val_x = split_test_train_val(target_x.shape[0])
    other_train_x, other_test_x, other_val_x = split_test_train_val(other_x.shape[0])
    gf_train_x, gf_test_x, gf_val_x = split_test_train_val(gf_x.shape[0])
    print(len(target_train_x), len(target_test_x),len(target_val_x), len(other_train_x),len(other_test_x),len(other_val_x),
          len(gf_train_x),len(gf_test_x),len(gf_val_x))

    X_train = np.nan_to_num((np.vstack((target_x[target_train_x,:,:], other_x[other_train_x,:,:], gf_x[gf_train_x,:,:])).astype('float32')))
    y_train = np.nan_to_num(np.array([0]*len(target_train_x)+[1]*len(other_train_x)+[2]*len(gf_train_x)).astype('float32'))
    print(X_train.shape, y_train.shape)

    X_val = np.nan_to_num(np.vstack((target_x[target_val_x,:,:], other_x[other_val_x,:,:], gf_x[gf_val_x,:,:])).astype('float32'))
    y_val = np.nan_to_num(np.array([0]*len(target_val_x)+[1]*len(other_val_x)+[2]*len(gf_val_x)).astype('float32'))
    print(X_val.shape, y_val.shape)
    
    X_test = np.nan_to_num(np.vstack((target_x[target_test_x,:,:], other_x[other_test_x,:,:], gf_x[gf_test_x,:,:])).astype('float32'))
    y_test = np.nan_to_num(np.array([0]*len(target_test_x)+[1]*len(other_test_x)+[2]*len(gf_test_x)).astype('float32'))
    print(X_test.shape, y_test.shape)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


class EEGNet(nn.Module):
    def __init__(self, number_of_channels):
        super(EEGNet, self).__init__()
        self.T = 120
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, number_of_channels), padding = 0) #Change to support input from the number of channels
        #self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
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
        self.fc1 = nn.Linear(4*2*5, 3) #Change to support 88 timepoints from the number of channels
        

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
        #x = x.view(-1, 4*2*7)
        x = x.reshape(-1, 4*2*5) #Change to support 88 timepoints from the number of channels; view to reshape
        #x = torch.sigmoid(self.fc1(x))
        m = nn.Softmax(dim=1) #Change to support 3 classses
        x = m(self.fc1(x))
        return x
def EEG_class_visualize(epoch_loss_values, acc_tr, acc_v, acc_ts, auc_tr, auc_v, auc_ts):
    plt.title("Av Loss  ")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()

    x = [i + 1 for i in range(len(acc_tr))]
    y = [acc_tr, acc_v, acc_ts]
    plt.title("ACC")
    plt.xlabel("epoch")
    lineObjects = plt.plot(x, np.transpose(y))
    plt.legend(['acc_tr', 'acc_v', 'acc_ts'])
    plt.show()
    
    x = [i + 1 for i in range(len(auc_tr))]
    y = [auc_tr, auc_v, auc_ts]
    plt.title("AUC")
    plt.xlabel("epoch")
    lineObjects = plt.plot(x, np.transpose(y))  # y describes 3 lines
    plt.legend(['auc_tr', 'auc_v', 'auc_ts'])
    plt.show()