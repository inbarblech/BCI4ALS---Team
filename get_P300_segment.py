import os
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
from filter_ import filter_sig as flt
import pandas as pd 



PRIO_MARKER = 0.200
POST_MARKER = 0.500

def plot_full_signal():
    plot_full_signal()
    signal = get_all()
    print(type(signal))
    x_range = np.arange(2500, 5000,1) 
    av = sum(signal)/len(signal)
    signal = signal[2500:5000]-av
    plt.plot(x_range, np.array(signal), linewidth=0.5)
    plt.ylim(signal.min(), signal.max())
    plt.show()
    #print(signal_segment)
    plt.show()
    

def get_all(recording_folder = os.path.join(os.getcwd(), "Recordings\\EEG_05_12_2.xdf")):
    data, header = pyxdf.load_xdf(recording_folder)
      
    #get data from GUI streem
    line_= data[1] 
    channels_data = np.array(line_["time_series"])
    channels_data = flt(channels_data)
    return channels_data



def get_P300_segment(recording_folder = os.path.join(os.getcwd(), "Recordings\\EEG_05_12_2.xdf"), target = 'Target'):
    data, header = pyxdf.load_xdf(recording_folder)
    
    #get data from markers streem
    line_= data[0] 
    markers_list = np.array(line_["time_series"])
    markers_time_stamps = np.array(line_["time_stamps"])
    
    #get data from GUI streem
    line_= data[1] 
    channels_data = np.array(line_["time_series"])
    channels_data = flt(channels_data)

    time_stamps_data = np.array(line_["time_stamps"])
     
    #cut the relevant signal accroding to the markers
    start_marker_search_index = 0
    signal_segment_list = [] 
    len_min = 1000
    signal_segment_df_list = []
    
    for i in range(markers_list.shape[0]):
        if(markers_list[i] == target): 
        #if(markers_list[i] == 'Circle'):#currently EGI.xdf has "Circle". Replace with Target once recored
            start_index = -1
            stop_index = -1
            for index in range(start_marker_search_index, time_stamps_data.shape[0]):
                if time_stamps_data[index]>markers_time_stamps[i]-PRIO_MARKER:
                    start_index = index
                    start_marker_search_index = start_index
                    break
            if (start_index == -1):
                print("failed to find start index")
                break            
            for index in range(start_marker_search_index, time_stamps_data.shape[0]):
                if time_stamps_data[index]>markers_time_stamps[i]+POST_MARKER:
                    stop_index = index
                    start_marker_search_index = index
                    break
            if(stop_index == -1):
                print("failed to find stop index")
                stop_index = time_stamps_data.shape[0] - 1
                signal_segment = channels_data[start_index:stop_index, :]
                signal_segment_list.append(signal_segment)
                break
            signal_segment = channels_data[start_index:stop_index, :]
            time_segment = time_stamps_data[start_index:stop_index]            
            len_min = min(len_min, len(signal_segment))
            d = {'Index': range(0, len(signal_segment)), 'Time':time_segment}
            for i in range (0, signal_segment.shape[1]):
                d['Cannel' + str(i)]= signal_segment[:,i]
            signal_segment_df = pd.DataFrame(data=d)                
            signal_segment_list.append(signal_segment)
            signal_segment_df_list.append(signal_segment_df)
    print(len_min)
    return  signal_segment_list, time_segment, len_min, signal_segment_df_list

#markernames = ['Target', 'Other', 'inter']

if __name__ == '__main__':

    signal_segment_list_target, time_segment, len_min, signal_segment_df_list = get_P300_segment(target='Target')
    #signal_segment_list_other, time_segment, len_min, signal_segment_df_list = get_P300_segment(target='Other')
    print(signal_segment_df_list)
    
    
    for signal_segment in signal_segment_list_target:
        signal_segment = signal_segment[0:len_min]
        x_range = np.arange(0, len(signal_segment),1) 
        av = sum(signal_segment)/len(signal_segment)
        signal_segment = signal_segment-av
        plt.plot(x_range, np.array(signal_segment))
        plt.ylim(signal_segment.min(), signal_segment.max())
        #print(signal_segment)
        plt.show()
        
