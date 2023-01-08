# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:52:17 2022

@author: marko
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
folder = "C:\\Users\\marko\\bci\\exercises\\upon Tomer's request"

def get_scaled(m_lst):
    f_list = []
    for i in range(m_lst[0].shape[1]): #create data frame fo each channel
        f_list.append(pd.DataFrame())
    i = 0
    for mr in m_lst: 
        c_name = "ep" + str(i)
        i +=1
        for ch in range(mr.shape[1]):
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

def plot_data(markers_list, markers_time_stamps, channels_data, time_stamps_data, title, center = False, jupyter_b = False):
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
        if(m=='Circle-t' or m=='triangle-t'): plt.plot([m_x],marker_y, 'go',label='marker', markersize=5, markeredgecolor="blue", markerfacecolor="blue")
        elif(m=='triangle' or m ==  "Circle"): plt.plot([m_x],marker_y, 'go',label='marker', markersize=5, markeredgecolor="red", markerfacecolor="red")
        else: plt.plot([m_x],marker_y, 'go',label='marker', markersize=5, markeredgecolor="grey", markerfacecolor="grey")
    plt.savefig(os.path.join(folder, title))
    if(jupyter_b == False):
        plt.show()
    
    
    #plot frequencies 
    if(jupyter_b == False):
        plt.title(title)
        for ch in channels_data.transpose():
            plt.psd(ch, Fs = 125)
        plt.savefig(os.path.join(folder, title))    
        plt.show()

def plot_each_segment_all_ch(signal_segment_list, time_segment_list, markers_placement_list, target, num_of_channels):
    seg_freq = False
    i = 0
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
        plt.savefig(os.path.join(folder, target + str(i)))
        i = i+1
        plt.show()
        if (seg_freq):
            for ch in signal_segment.transpose():
                plt.psd(ch, Fs = 125)
            plt.savefig(os.path.join(folder, target + "fr" + str(i)))
            plt.show()
        
def plot_all_segments_scaled_av_per_ch(tr_lst, o_lst, g_lst, signal_time, markers_placement, scale = True):
    tr_lst = get_scaled(tr_lst)
    o_lst = get_scaled(o_lst)
    if(len(g_lst)!=0): g_lst = get_scaled(g_lst)
    i = 0
    for tr, o, g in zip(tr_lst, o_lst, g_lst):
        plt.title("Scaled Channel " + str(i))
        x_range=(signal_time - markers_placement)*1000
        #aligne x_range and y
        if(x_range.shape[0]>tr.shape[0]): x_range = x_range[0:tr.shape[0]]
        #if(x_range.shape[0]>g.shape[0]): x_range = x_range[0:g.shape[0]]
        if(len(g_lst)!=0): 
            if(x_range.shape[0]>o.shape[0]): x_range = x_range[0:o.shape[0]]
        plt.plot(x_range, tr[0:x_range.shape[0]])
        plt.plot(x_range, o[0:x_range.shape[0]], color = 'red')
        if(len(g_lst)!=0): plt.plot(x_range, g[0:x_range.shape[0]], color = 'grey')
        #plt.ylim(tr.min(),tr.max())
        plt.savefig(os.path.join(folder, "Scaled Channel " + str(i)))
        plt.show()   
        i+=1
        
def plot_all_segments_raw_av_per_ch(tr_lst, o_lst, g_lst, signal_time, markers_placement, recored_file_name, current_target):
    ch_tr= []
    for i in range(tr_lst[0].shape[1]): ch_tr.append(pd.DataFrame())
    
    ch_o= []
    for i in range(o_lst[0].shape[1]): ch_o.append(pd.DataFrame())

    ch_g= []
    if(len(g_lst)!=0): 
        for i in range(g_lst[0].shape[1]): ch_g.append(pd.DataFrame())    
    
    ep  = 0
    for tr in tr_lst: #each tr is a different epoch
        for i in range(tr.shape[1]):
            ch_tr[i]= pd.concat([ch_tr[i], pd.DataFrame({"ep" + str(ep):tr[:,i]})], axis=1) #add current epoch values to all channels
        ep += 1
    target_epochs_num = ep

    ep  = 0
    for o in o_lst:
        for i in range(o.shape[1]):
            ch_o[i]= pd.concat([ch_o[i], pd.DataFrame({"ep" + str(ep):o[:,i]})], axis=1)
        ep += 1
        if(ep>target_epochs_num): break #assure the same number of epochs for target and none target 
        
    if(len(g_lst)!=0):
        print("g_lst")
        ep  = 0
        for g in g_lst:
            for i in range(g.shape[1]):
                ch_g[i]= pd.concat([ch_g[i], pd.DataFrame({"ep" + str(ep):g[:,i]})], axis=1)
            ep += 1
            if(ep>target_epochs_num): 
                print("break", ep)
                break #assure the same number of epochs for target and none target 
    
    x_range=(signal_time - markers_placement)*1000 #time of the first epoch, normalized to set marker at 0 is used as x axis
    
    i = 0
    for tr, o, g in zip(ch_tr, ch_o, ch_g): #separate graph for each channel
        plt.title (recored_file_name + "_ch " + str(i))
        i +=1
        if(x_range.shape[0]>tr.shape[0]): x_range = x_range[0:tr.shape[0]] #assure the same dimention for x and y
        if(x_range.shape[0]>o.shape[0]): x_range = x_range[0:o.shape[0]]
        if(x_range.shape[0]>g.shape[0]): x_range = x_range[0:g.shape[0]]
        plt.plot(x_range, tr[0:x_range.shape[0]].mean(axis=1))
        plt.plot(x_range, o[0:x_range.shape[0]].mean(axis=1), color = 'red')
        if(len(g_lst)!=0): plt.plot(x_range, g[0:x_range.shape[0]].mean(axis=1), color = 'grey')
        plt.savefig(os.path.join(folder, recored_file_name + "Av Channel " + str(i)))
        if(current_target == "triangle"): none_target = "circle"
        else: none_target = "triangle"
        plt.gca().legend([current_target + " target", none_target + " none target", "rect gap filler"])
        plt.show()

    