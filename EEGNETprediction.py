# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:21:12 2023

@author: marko
"""
import numpy as np
import torch
import sklearn
import os
from EEGNETtools import EEGNet
from EEGNETtools import chosen_channels
from EEGNETtools import read_input_x
from torch.autograd import Variable
Data_Path = os.path.join(os.path.join(os.getcwd(), os.pardir), "BCI_data")
Segmented_Data_Path = os.path.join(Data_Path, "segmented_data")
EEGnet_Path = os.path.join(Segmented_Data_Path, "for_EEGNET")

def EEGNET_predict_target(on_x,off_x):
    net = EEGNet(len(chosen_channels))
    net.load_state_dict(torch.load(os.join(EEGnet_Path, "best_metric_model_debi.pth")))
    
    with torch.no_grad():
        input1 = Variable(torch.from_numpy(on_x))
        predicted1 = net(input1)
        input2 = Variable(torch.from_numpy(off_x))
        predicted2 = net(input2)
        
    index1 = np.argmax(predicted1)  #If index is 0, then this is the target. 
    index2 = np.argmax(predicted2)  #If index is 0, then this is the target. 
    if index1 == 0 and index2!=0: 
        print('Light On')
        return 'Light On' 
    elif index1 != 0 and index2==0: 
            print('Light Off')
            return 'Light Off' 
    else:
         print('Failed to recognize') #None was recognized as target or both were recognized as target 
         return 'Failed to recognize'


def EEGNET_get_epoch_type(target_x, other_x, gf_x, tr):
    print(target_x.shape, other_x.shape, gf_x.shape)
    X = np.vstack((target_x, other_x, gf_x))
    y = np.array([0]*len(target_x)+[1]*len(other_x)+[2]*len(gf_x)).astype('float32')
    print(X.shape, y.shape)
    
    net = EEGNet(len(chosen_channels))
    net.load_state_dict(torch.load("best_metric_model.pth"))

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
def EEGNET_get_epoch_type_from_filetarget_path(target_path=os.path.join(EEGnet_Path, 'target'),
                                               other_path=os.path.join(EEGnet_Path, 'other'),
                                               gf_path=os.path.join(EEGnet_Path, 'gap filler'), tr=0.22246733):

    target_x, other_x, gf_x = get_data(target_path, other_path, gf_path)
    pred_target_list = EEGNET_get_epoch_type(target_x, other_x, gf_x, tr)
    return pred_target_list 


if __name__ == '__main__':

     #pred_target_list = EEGNET_get_epoch_type_from_filetarget_path()
     target_x, other_x, gf_x = get_data(target_path=os.path.join(EEGnet_Path, 'target'),
                                        other_path=os.path.join(EEGnet_Path, 'other'),
                                        gf_path=os.path.join(EEGnet_Path, 'gap filler'))
     pred_target_list = EEGNET_get_epoch_type(target_x, other_x, gf_x, tr=0.3)
     print(type(target_x), target_x.shape)
     print(type(pred_target_list), len(pred_target_list))
 
    

