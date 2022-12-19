import os
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
from filter_ import filter_sig as flt
import csv



PRIO_MARKER = 0.200
POST_MARKER = 0.500
RECORDED_FILE = "EEG_12_12.xdf"
#RECORDED_FILE = "EEG_5_12_1.xdf"
CURRENT_FOLDER = os.getcwd()
RECORDING_FOLDER = os.path.join(CURRENT_FOLDER, "Recordings")
RECORDING_FILE = os.path.join(RECORDING_FOLDER, RECORDED_FILE)
DATA_FOLDER = os.path.join(CURRENT_FOLDER, "segmented_data")
OTHER_DATA_FOLDER = os.path.join(DATA_FOLDER, "other\\data")
TARGET_DATA_FOLDER = os.path.join(DATA_FOLDER, "target\\data")

def get_signal(recording_file = RECORDING_FILE):
    data, header = pyxdf.load_xdf(recording_file)    
    #get data from markers streem
    line_= data[0] 
    markers_list = np.array(line_["time_series"])
    markers_time_stamps = np.array(line_["time_stamps"])
    
    #get data from GUI streem
    line_= data[1] 
    channels_data = np.array(line_["time_series"])
    channels_data = flt(channels_data) ##filter channel data
    time_stamps_data = np.array(line_["time_stamps"])
    return markers_list, markers_time_stamps, channels_data, time_stamps_data
        


def plot_full_signal(start = 2500, stop = 5000):
    markers_list, markers_time_stamps, channels_data, time_stamps_data = get_signal()
    print(type(channels_data))
    x_range = np.arange(start, stop,1) 
    av = sum(channels_data)/len(channels_data)
    signal = channels_data[start:stop]-av
    plt.plot(x_range, np.array(signal), linewidth=0.5)
    plt.ylim(signal.min(), signal.max())
    plt.show()
    plt.show()
    

def get_P300_segment(recording_file = RECORDING_FILE, target = 'Target'):
    
    markers_list, markers_time_stamps, channels_data, time_stamps_data = get_signal(recording_file = recording_file)
     
    #cut the relevant signal accroding to the markers
    end_of_data_s = time_stamps_data.shape[0]    
    start_marker_search_index = 0
    signal_segment_list = [] 
    time_segment_list = []
    markers_placement_list  = []
    for i in range(markers_list.shape[0]): 
        if(markers_list[i] == target): 
            start_index = -1
            stop_index = -1
            marker_index = -1
            #find the beginning of the data segment
            for index in range(start_marker_search_index, end_of_data_s):
                if time_stamps_data[index]>=markers_time_stamps[i]-PRIO_MARKER:
                    start_marker_search_index = index
                    start_index = index
                    break
            if (start_index == -1):
                print("failed to find start index")
                break  
            #find markers place inside the segment 
            for index in range(start_marker_search_index, end_of_data_s):
                if time_stamps_data[index]>=markers_time_stamps[i]:
                    start_marker_search_index = index
                    marker_index = index
                    break
            if(marker_index == -1):
                print("failed to find marker_index ")
                start_marker_search_index = end_of_data_s - 1
                marker_index = end_of_data_s - 1
                stop_index = end_of_data_s - 1
            for index in range(start_marker_search_index, end_of_data_s):
                if time_stamps_data[index]>markers_time_stamps[i]+POST_MARKER:
                    start_marker_search_index = index
                    stop_index = index
                    break
            if(stop_index == -1):
                print("failed to find stop index")
                start_marker_search_index = stop_index = end_of_data_s - 1
            signal_segment = channels_data[start_index:stop_index, 0:9]
            signal_segment_list.append(signal_segment)
            time_segment = time_stamps_data[start_index:stop_index]
            time_segment_list.append(time_segment)
            markers_placement_list.append(markers_time_stamps[i])
                   
    return  signal_segment_list, time_segment_list, markers_placement_list

def plot_segments(signal_segment_list, time_segment_list, markers_placement_list, target):
    for i in range(len(signal_segment_list)):        
        signal_segment = signal_segment_list[i][:,0:8]
        signal_time = time_segment_list[i]
        x_range=(signal_time - markers_placement_list[i])*1000
        av = sum(signal_segment)/len(signal_segment)
        signal_segment = signal_segment-av
        plt.plot(x_range, np.array(signal_segment))
        marker_x = [0]
        marker_y = [0]
        plt.plot(marker_x,marker_y, 'go',label='marker', markersize=10, markeredgecolor="red", markerfacecolor="green")
        plt.ylim(signal_segment.min(), signal_segment.max())
        plt.title(target)
        plt.xlabel('msec') 
        plt.legend()
        plt.show()

def save_data(signal_segment_list, time_segment_list, markers_placement_list, target):
    #save segments to file for the future use
    i = 0
    for signal_segment, time_segment in zip(signal_segment_list,time_segment_list):
        recoreded_file_name = RECORDED_FILE.split(".")[0]
        if(target == 'Target'):            
            data_file = os.path.join(TARGET_DATA_FOLDER, recoreded_file_name + "_" + str(i) + ".csv")
        else:
            data_file = os.path.join(OTHER_DATA_FOLDER, recoreded_file_name + "_" + str(i) + ".csv")
        i += 1
        with open(data_file,'w') as f:
            # create the csv writer
            writer = csv.writer(f)    
            # write a row to the csv file
            header = np.arange(0, signal_segment.shape[1])
            header = np.hstack((["ts"], header))
            writer.writerow(header)
            print(signal_segment.shape)
            for line, ts in zip(signal_segment, time_segment):
                writer.writerow(np.hstack(([ts], line)))
    
#markernames = ['Target', 'Other', 'inter']
if __name__ == '__main__':
    #read, filter, and segement target data 
    signal_segment_list_target, time_segment_list_target, markers_placement_list_target = get_P300_segment(target='Target')
    #read, filter, and segement other  data 
    signal_segment_list_other, time_segment_list_other, markers_placement_list_other = get_P300_segment(target='Other')
    
    #save target segements
    save_data(signal_segment_list_target, time_segment_list_target, markers_placement_list_target, 'Target')
    #save other segements
    save_data(signal_segment_list_other, time_segment_list_other, markers_placement_list_other, 'Other')
    
    #plot target
    plot_segments(signal_segment_list_target, time_segment_list_target, markers_placement_list_target, 'Target')
    #plot other 
    plot_segments(signal_segment_list_other, time_segment_list_other, markers_placement_list_other, 'Other')

        
