# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:21:12 2023

@author: marko
"""
import numpy as np
import torch
import sklearn
from EEGNETtools import EEGNet
from EEGNETtools import chosen_channels
from EEGNETtools import read_input_x
from torch.autograd import Variable

def EEGNET_get_epoch_type(target_x, other_x, gf_x, tr):
    print(target_x.shape, other_x.shape, gf_x.shape)
    X = np.vstack((target_x, other_x, gf_x))
    y = np.array([0]*len(target_x)+[1]*len(other_x)+[2]*len(gf_x)).astype('float32')
    print(X.shape, y.shape)
    
    net = EEGNet(len(chosen_channels))
    net.load_state_dict(torch.load("C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\best_metric_model.pth"))

    with torch.no_grad():
        inputs = Variable(torch.from_numpy(X))
        predicted = net(inputs)
    pred_target_list = []   
    target_list = []
    for pred, y_true in zip(predicted, y):
        if(pred[0]>tr): pred_target = True
        else: pred_target = False
        if(int(y_true) == 0): target = True
        else: target = False
        print(pred_target, target, y_true, pred[0])
        pred_target_list.append(pred_target)
        target_list.append(target)
        
    ar = sklearn.metrics.confusion_matrix(pred_target_list, target_list, )
    print(ar)
    return pred_target_list
    
def get_data(target_path, other_path, gf_path):
    target_x = read_input_x(target_path,chosen_channels,0)   
    other_x = read_input_x(other_path,chosen_channels, 0)   
    #limit gap fillers to the number of targets/others
    gf_x = read_input_x(gf_path,chosen_channels ,max(target_x.shape[0], other_x.shape[0]))   
    return target_x, other_x, gf_x
def EEGNET_get_epoch_type_from_filetarget_path(target_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\target\\data",
                                                  other_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\other\\data",
                                                  gf_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\GF\\data", tr = 0.22246733):

    target_x, other_x, gf_x = get_data(target_path, other_path, gf_path)
    pred_target_list = EEGNET_get_epoch_type(target_x, other_x, gf_x, tr)
    return pred_target_list 


if __name__ == '__main__':

     #pred_target_list = EEGNET_get_epoch_type_from_filetarget_path()
     target_x, other_x, gf_x = get_data(target_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\target\\data", 
                                        other_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\other\\data", 
                                        gf_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\GF\\data")
     pred_target_list = EEGNET_get_epoch_type(target_x, other_x, gf_x, tr= 0.22246733)
     print(type(target_x), target_x.shape)
 
    

