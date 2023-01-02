# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:52:17 2022

@author: marko
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from filter_ import cut_edges 


def get_scaled(m_lst):
    f_list = []
    for i in range(9):
        f_list.append(pd.DataFrame())
    i = 0
    for mr in m_lst: 
        c_name = "ep" + str(i)
        i +=1
        for ch in range(9):
            c = mr[:,ch:ch+1]
            c = c.reshape(c.shape[0])
            c = (c-(sum(c)/len(c)))/c.std() #Normalize
            f_list[ch]= pd.concat([f_list[ch], pd.DataFrame({c_name:c})], axis=1)
    i = 0
    scaled_list=[]
    for f in f_list:
        #intermidiat save of all epochs per channel
        #f.to_csv(os.path.join(N_SEG_PER_CH, RECORDED_FILE.split(".")[0] + str(i)+".csv"))
        i +=1

        scaled = f.mean(axis=1)/f.std(axis=1) #average and normalize per all epochos
        scaled_list.append(scaled)
    return scaled_list

def plot_segments_all(tr_lst, o_lst, signal_time, markers_placement, scale = True):
    tr_lst = get_scaled(tr_lst)
    o_lst = get_scaled(o_lst)
    i = 0
    for tr, o in zip(tr_lst, o_lst):
        plt.title("Channel " + str(i + 1))
        i+=1
        x_range=(signal_time - markers_placement)*1000
        #aligne x_range and y
        if(x_range.shape[0]>tr.shape[0]): x_range = x_range[0:tr.shape[0]]
        if(x_range.shape[0]>o.shape[0]): x_range = x_range[0:o.shape[0]]
        plt.plot(x_range, tr[0:x_range.shape[0]])
        plt.plot(x_range, o[0:x_range.shape[0]], color = 'red')
        plt.ylim(tr.min(),tr.max())
        plt.show()

def plot_data(markers_list, markers_time_stamps, channels_data, time_stamps_data, title, cut_start=0,  cut_stop=0, center = False, zoomin = False):
    #if enabled, cut the beginning and the end artifacts 
    channels_data_full = channels_data #save for the frequency presentation 
    if cut_start>0 or cut_stop >0:
        channels_data, time_stamps_data, markers_time_stamps, markers_list = cut_edges(channels_data, time_stamps_data, markers_time_stamps, markers_list, cut_start = cut_start, cut_stop = cut_stop)

    #plot time series 
    plt.title(title)    
    x_range = time_stamps_data

    if(center): #move to 0 for presentation
        plt.plot(x_range, channels_data - channels_data.sum(axis = 0)/channels_data.shape[0] )
    else:
        print("no center")
        plt.plot(x_range, channels_data) 
    plt.gca().legend((np.arange(1,channels_data.shape[0]+1)))

    #Add markers      
    marker_y = [0]
    for m, m_x in zip(markers_list, markers_time_stamps):      
        plt.plot([m_x],marker_y, 'go',label='marker', markersize=5, markeredgecolor="red", markerfacecolor="green")
    if(zoomin):
        plt.margins(x=4, y=-0.25)    
    plt.show()
    
    #plot frequencies 
    plt.title(title)
    for ch in channels_data_full.transpose():
        plt.psd(ch, Fs = 125)
    if(zoomin):
        plt.margins(x=0, y=-0.3)  
        plt.ylim(-8, 11)
    plt.show()

def plot_segments(signal_segment_list, time_segment_list, markers_placement_list, target, num_of_channels):
    seg_freq = False
    for signal_segment, signal_time, marker in zip(signal_segment_list, time_segment_list, markers_placement_list):        
        x_range=(signal_time - marker)*1000 #shift to place marker in x= 0
        plt.plot(x_range, signal_segment)
        marker_x = [0]
        marker_y = [0]
        plt.plot(marker_x, marker_y, 'go',label='marker', markersize=10, markeredgecolor="red", markerfacecolor="green")
        plt.ylim(signal_segment.min(), signal_segment.max())
        plt.title(target)
        plt.xlabel('msec') 
        plt.legend()
        plt.show()
        if (seg_freq):
            for ch in signal_segment.transpose():
                plt.psd(ch, Fs = 125)
        plt.show()
    