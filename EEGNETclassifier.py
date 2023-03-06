# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:27:57 2023

@author: marko
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

import os
import torch.optim as optim
import math
import sklearn.metrics


from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from EEGNETtools import split_test_train_val
from EEGNETtools import read_input_x
from EEGNETtools import EEGNet
from torch.autograd import Variable
import torch
import torch.nn as nn

target_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\target\\data"
other_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\other\\data"
gf_path = "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\segmented_data\\GF\\data"



def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 17
    
    predicted = []
    if(int(len(X)/batch_size>1)): 
        range_r = int(len(X)/batch_size)
    else: 
        range_r = 1
    for i in range(range_r):
        s = i*batch_size
        if (s+batch_size > len(X)): 
            e = len(X)
        else: 
            e = i*batch_size+batch_size
        
        inputs = Variable(torch.from_numpy(X[s:e]))
        pred = model(inputs)
        
        predicted.append(pred.data.cpu().numpy())
        
        
    inputs = Variable(torch.from_numpy(X))
    predicted = model(inputs)
    
    predicted = predicted.data.cpu().numpy()
    predicted_ = np.zeros((predicted.shape[0],))
    
    for param in params:
        if param == 'acc':
            for i in range(predicted.shape[0]):
                raw = predicted[i,:]
                index = np.argmax(raw)
                predicted_[i] = index
            result = accuracy_score(Y, predicted_)
            results.append(result)
        if param == "auc":
            for i in range(predicted.shape[0]):
                predicted_sum = predicted[i,0] + predicted[i,1]+predicted[i,2]
                predicted[i,0] = predicted[i,0]/predicted_sum
                predicted[i,1] = predicted[i,1]/predicted_sum
                predicted[i,2] = predicted[i,2]/predicted_sum
            results.append(roc_auc_score(Y, predicted, multi_class = 'ovo'))
        if param == "recall":
            results.append(recall_score(Y, (predicted)))
        if param == "precision":
            results.append(precision_score(Y, predicted))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall/ (precision+recall))
    return results

if __name__ == '__main__':
    net = EEGNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    channel = [0,2,7,8,9]
    #channel = [8]
    target_x = read_input_x(target_path,channel,0)   
    other_x = read_input_x(other_path,channel, 0)   
    #limit gap fillers to the number of targets/others
    gf_x = read_input_x(gf_path,channel ,max(target_x.shape[0], other_x.shape[0]))   
    print(target_x.shape, other_x.shape, gf_x.shape)
    
    target_train_x, target_test_x, target_val_x = split_test_train_val(target_x.shape[0])
    other_train_x, other_test_x, other_val_x = split_test_train_val(other_x.shape[0])
    gf_train_x, gf_test_x, gf_val_x = split_test_train_val(gf_x.shape[0])
    print(len(target_train_x), len(target_test_x),len(target_val_x), len(other_train_x),len(other_test_x),len(other_val_x),
          len(gf_train_x),len(gf_test_x),len(gf_val_x))
    
    X_train = np.vstack((target_x[target_train_x,:,:], other_x[other_train_x,:,:], gf_x[gf_train_x,:,:]))
    y_train = np.array([0]*len(target_train_x)+[1]*len(other_train_x)+[2]*len(gf_train_x)).astype('float32')
    print(X_train.shape, y_train.shape)

    X_val = np.vstack((target_x[target_val_x,:,:], other_x[other_val_x,:,:], gf_x[gf_val_x,:,:]))
    y_val = np.array([0]*len(target_val_x)+[1]*len(other_val_x)+[2]*len(gf_val_x)).astype('float32')
    print(X_val.shape, y_val.shape)
    
    X_test = np.vstack((target_x[target_test_x,:,:], other_x[other_test_x,:,:], gf_x[gf_test_x,:,:]))
    y_test = np.array([0]*len(target_test_x)+[1]*len(other_test_x)+[2]*len(gf_test_x)).astype('float32')
    print(X_test.shape, y_test.shape)
    batch_size = 17
    epoch_loss_values = []
    acc_metric_values_train = []
    acc_metric_values_val = []
    acc_metric_values_test = []
    best_metric = -1
    auc_metric_values_train = []
    auc_metric_values_val = []
    auc_metric_values_test = []
    
    
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    for epoch in range(400):  # loop over the dataset multiple times
        if(epoch%200 ==0): print("\nEpoch ", epoch)
        
        running_loss = 0.0
        step = 0
        net.train()
        for i_b in range(int(len(X_train)/batch_size)):
            s = i_b*batch_size
            if(s>=len(X_train)):break
            step +=1
            if(len(X_train)<s+batch_size): e = len(X_train)
            else: e = s+batch_size        
                
            inputs = torch.from_numpy(X_train[s:e])
            lables = np.zeros((e-s,3))
            j = 0
            for y in y_train[s:e]:
                if y == 0: lables[j,0]=1
                elif y == 1: lables[j,1]=1
                else: lables[j,2]=1
                j+=1
            #lables = torch.FloatTensor(np.array([y_train[s:e]]).T*1.0)
            lables = torch.FloatTensor(lables)
            
            # wrap them in Variable
            inputs, lables = Variable(inputs), Variable(lables)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, lables)
            loss.backward()                
            optimizer.step()
            
            running_loss += loss.item()
            
        params = ["acc", "auc"]
        epoch_loss_values.append(running_loss/step)
        net.eval()
        with torch.no_grad():
            results_train = evaluate(net, X_train, y_train, params)
            acc_metric_values_train.append(results_train[0])
            auc_metric_values_train.append(results_train[1])
    
            results_val = evaluate(net, X_val, y_val, params)
            acc_metric_values_val.append(results_val[0])
            auc_metric_values_val.append(results_val[1])
            if  results_val[0]  > best_metric:
                best_metric = results_val[0]
                best_metric_epoch = epoch + 1
                torch.save(net.state_dict(), os.path.join(
                    "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team", "best_metric_model.pth"))
                print("saved new best metric model", best_metric)
    
            results_test = evaluate(net, X_test, y_test, params)
            acc_metric_values_test.append(results_test[0])
            auc_metric_values_test.append(results_test[1])
            if(epoch%200 ==0): 
                print("Training Loss ", running_loss/step, epoch)
                print("Train - ", results_train)
                print("Val - ", results_val)
                print("Test - ", results_test)
    ch_name = " "
    for ch in channel:
        ch_name+=str(ch) + '_'
    
    plt.figure("train", (12, 6))
    plt.subplot(1, 1, 1)
    plt.title("Av Loss  " +ch_name)
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()
    
    plt.subplot(1, 1, 1)
    plt.title("Val AUC"+ch_name)
    x = [i + 1 for i in range(len(auc_metric_values_train))]
    y = [auc_metric_values_train, auc_metric_values_val, auc_metric_values_test]
    plt.xlabel("epoch")
    plt.plot(x, np.transpose(y))
    plt.show()
    
    plt.subplot(1, 1, 1)
    plt.title("Val ACC"+ch_name)
    x = [i + 1 for i in range(len(acc_metric_values_train))]
    y = [acc_metric_values_train, acc_metric_values_val, acc_metric_values_test]
    plt.xlabel("epoch")
    plt.plot(x, np.transpose(y))
    plt.show()
    
    with torch.no_grad():
        inputs = Variable(torch.from_numpy(X_train))
        predicted = net(inputs)
    
    target_pred = []
    target_label = []
    for p,y in zip(predicted, y_train):       
        target_pred.append(p[0])
        if(int(y) == 0): target_label.append(1)
        else: target_label.append(0)
    
    fpr, tpr, thresholds =sklearn.metrics.roc_curve(target_label, target_pred)
    roc_auc = sklearn.metrics.roc_auc_score(target_label, target_pred)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    for tpr_, trh in zip(tpr, thresholds):
        print(tpr_, trh)
