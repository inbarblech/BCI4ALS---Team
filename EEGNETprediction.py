# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:21:12 2023

@author: marko
"""
import numpy as np
import torch
from EEGNETtools import EEGNet
import sklearn

target_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\target\\data"
other_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\other\\data"
gf_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\GF\\data"

from EEGNETtools import read_input_x
from torch.autograd import Variable


if __name__ == '__main__':
    channel = [0,2,7,8,9]
    target_x = read_input_x(target_path,channel,0)   
    other_x = read_input_x(other_path,channel, 0)   
    #limit gap fillers to the number of targets/others
    gf_x = read_input_x(gf_path,channel ,max(target_x.shape[0], other_x.shape[0]))   
    print(target_x.shape, other_x.shape, gf_x.shape)
    X = np.vstack((target_x, other_x, gf_x))
    y = np.array([0]*len(target_x)+[1]*len(other_x)+[2]*len(gf_x)).astype('float32')
    print(X.shape, y.shape)
    
    tr = 0.22246733
    net = EEGNet()
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

