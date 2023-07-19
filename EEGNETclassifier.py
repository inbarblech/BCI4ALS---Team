# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 20:31:09 2023

@author: marko
"""
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os

from EEGNETtools import chosen_channels as channels
from EEGNETtools import get_data_for_EEGNet, evaluate, EEGNet, EEG_class_visualize

if __name__ == '__main__':

    net = EEGNet(len(channels))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    
    X_train, y_train, X_val, y_val, X_test, y_test = get_data_for_EEGNet()
    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    batch_size = 20
    epoch_loss_values = []
    auc_tr = []
    acc_tr = []
    auc_v = []
    acc_v = []
    auc_ts = []
    acc_ts = []
    best_metric = -1

    classify = True
    step = 0

    if classify:
        for epoch in range(400):  # loop over the dataset multiple times
            print("\nEpoch ", epoch)

            running_loss = 0.0
            for i in range(int(len(X_train)/batch_size)+1):
                s = i*batch_size
                if i*batch_size+batch_size > len(X_train): e = len(X_train)
                else: e = i*batch_size+batch_size
                step+=1
                
                inputs = torch.from_numpy(X_train[s:e])
                l = np.zeros((e-s,3))
                for i, y in enumerate(y_train[s:e]):
                    l[i,int(y)] = 1
                labels = torch.FloatTensor(np.array(l*1.0))

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()


                optimizer.step()

                #running_loss += loss.data[0]
                running_loss += loss.item() #change data to item
            
            # Validation accuracy
            params = ["acc", "auc", "fmeasure"]
            epoch_loss_values.append(running_loss/step)
            auc_tr.append(evaluate(net, X_train, y_train, params)[1])
            acc_tr.append(evaluate(net, X_train, y_train, params)[0])
            auc_v.append(evaluate(net, X_val, y_val, params)[1])
            acc_v.append(evaluate(net, X_val, y_val, params)[0])
            auc_ts.append(evaluate(net, X_test, y_test, params)[1])
            acc_ts.append(evaluate(net, X_test, y_test, params)[0])
            print(params)
            print("Training Loss ", running_loss)
            print("Train - ", evaluate(net, X_train, y_train, params))
            print("Validation - ", evaluate(net, X_val, y_val, params))
            print("Test - ", evaluate(net, X_test, y_test, params))

            if  acc_v[-1] > best_metric:
                best_metric = acc_v[-1]
                best_metric_epoch = epoch + 1
                torch.save(net.state_dict(), os.path.join(
                    "C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team", "best_metric_model_debi.pth"))
                print("saved new best metric model", best_metric)
        EEG_class_visualize(epoch_loss_values, acc_tr, acc_v, acc_ts, auc_tr, auc_v, auc_ts)

    
    #Test
    net.load_state_dict(torch.load("C:\\Users\\marko\\bci\\exercises\\BCI4ALS---Team\\best_metric_model_debi.pth"))
    with torch.no_grad():
        inputs = Variable(torch.from_numpy(X_test))
        predicted = net(inputs).numpy().astype(np.float32)
    class_pred = []
    for y, p in zip(y_test, predicted):
        print (y, np.argmax(p).astype(np.float32))
        class_pred.append(np.argmax(p).astype(np.float32))
    class_pred = np.array(class_pred)
    correct_answers = y_test[np.where(y_test == class_pred)]
    print("All correct predictions: {}, Total test: {}, Total accuracy: {:.0%} ".format(len(correct_answers), len(y_test), len(correct_answers)/len(y_test)))
    correct_target = len(correct_answers[np.where(correct_answers == 0)])
    total_target = len(y_test[np.where(y_test ==0)])
    print("Target correct predictions: {}, Targets in test: {}, Target accuracy: {:.0%} ".format(correct_target, total_target, correct_target/total_target))
    correct_other = len(correct_answers[np.where(correct_answers == 1)])
    total_other = len(y_test[np.where(y_test ==1)])
    print("Other correct predictions: {}, Other in test: {}, Other accuracy: {:.0%} ".format(correct_other, total_other, correct_other/total_other))
    correct_gap_f = len(correct_answers[np.where(correct_answers == 2)])
    total_gap_f = len(y_test[np.where(y_test ==2)])
    print("Gap filler correct predictions: {}, Gap filler  in test: {}, Gap filler  accuracy: {:.0%} ".format(correct_gap_f, total_gap_f, correct_gap_f/total_gap_f))


