import os
import numpy as np
import pyxdf
from filter_ import filter_sig as flt
from filter_ import cut_edges 
from filter_ import baseline_correction
import csv
from visualizer import plot_data
from visualizer import plot_each_segment_all_ch
from visualizer import plot_all_segments_raw_av_per_ch
from visualizer import plot_all_segments_scaled_av_per_ch
from sklearn import preprocessing as pr



#RECORDED_FILE = "EEG_5_12_1.xdf"
#RECORDED_FILE = "EEG_05_12_2.xdf"
#RECORDED_FILE = "EEG_12_12.xdf"
#RECORDED_FILE = "EEG_12_12_1.xdf"
#RECORDED_FILE = "eeg_26_12_2.xdf"
RECORDED_FILE = "eeg_26_12_1.xdf"
#RECORDED_FILE = "eeg_26_12.xdf"
RECORDED_FILE = "eeg_31_12.xdf"
#RECORDED_FILE = "eeg_31_12_1.xdf"
#RECORDED_FILE = "synt.xdf"
#RECORDED_FILE = "EEG_6_1.xdf"
#RECORDED_FILE = "EEG_6_1_1.xdf"
RECORDED_FILE = "EEG_7_1_1.xdf"
RECORDED_FILE = "EEG_7_1_2.xdf"
#RECORDED_FILE = "EEG_7_1_3.xdf"
#RECORDED_FILE = "EEG_7_1_4.xdf"
#RECORDED_FILE = "EEG_8_1_1_t.xdf"
#RECORDED_FILE = "EEG_8_1_2_t.xdf"

PRIO_MARKER = 0.200
POST_MARKER = 0.500
REMOVE_START_ARTIFACT_AFTER_FLT = 3000

CURRENT_FOLDER = os.getcwd()
RECORDING_FOLDER = os.path.join(CURRENT_FOLDER, "Recordings")
RECORDING_FILE = os.path.join(RECORDING_FOLDER, RECORDED_FILE)
DATA_FOLDER = os.path.join(CURRENT_FOLDER, "segmented_data")
OTHER_DATA_FOLDER = os.path.join(DATA_FOLDER, "other\\data")
TARGET_DATA_FOLDER = os.path.join(DATA_FOLDER, "target\\data")
N_SEG_PER_CH = os.path.join(DATA_FOLDER, "n_seg_per_ch")

def get_signal(recording_file):    
    data, header = pyxdf.load_xdf(recording_file)  
    line_0= data[0] 
    line_1= data[1]
    print(data)
    if((isinstance(line_0["time_series"],list))!=True): #data/GUI streem is first 
        line_0, line_1 = line_1, line_0
    
    #get markers
    markers_list = np.array(line_0["time_series"])
    markers_time_stamps = np.array(line_0["time_stamps"])
    
    #get data from GUI streem
    channels_data = np.array(line_1["time_series"])
    time_stamps_data = np.array(line_1["time_stamps"])
       
    return markers_list, markers_time_stamps, channels_data, time_stamps_data
        

def get_P300_segment(markers_list, markers_time_stamps, channels_data, time_stamps_data, target = 'Target'):
        
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
            signal_segment = channels_data[start_index:stop_index, :]
            signal_segment_list.append(signal_segment)
            time_segment = time_stamps_data[start_index:stop_index]
            time_segment_list.append(time_segment)
            markers_placement_list.append(markers_time_stamps[i])
                   
    return  signal_segment_list, time_segment_list, markers_placement_list

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
            for line, ts in zip(signal_segment, time_segment):
                writer.writerow(np.hstack(([ts], line)))
    
def read_data(recorded_file = RECORDED_FILE, jupyter_b = False):
    #Read *.xdf file 
    recored_file_name = recorded_file.split('\\')[len(recorded_file.split('\\'))-1].split('.')[0]
                                            
    markers_list, markers_time_stamps, channels_data, time_stamps_data = get_signal(recorded_file)
    #print(markers_list)
    num_of_channels = 10
    channels_data = channels_data[:,0:num_of_channels]

    # Visualize time series and frequencies 
    plot_data(markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, center = False, jupyter_b = jupyter_b)        
    return markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, num_of_channels 

def filter_normalize(markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, num_of_channels, jupyter_b = False ):
    #Filter 
    chs = []
    for i in range(0, num_of_channels):  
        ##filter channel data 
        sig = flt(channels_data[:,i])
        chs.append(sig)    
    channels_data = np.array(chs).transpose()
    
    #cut fitler artifact in the beginning of the data time series  
    markers_list, markers_time_stamps, channels_data, time_stamps_data = cut_edges(markers_list, markers_time_stamps, channels_data, time_stamps_data, cut_start = REMOVE_START_ARTIFACT_AFTER_FLT, cut_stop= 0)
    
    # Visualize time series and frequencies after filtering        
    if(jupyter_b == False): plot_data(markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name + " filtered", center = False, jupyter_b = jupyter_b)    

    #Normalize 
    channels_data = pr.normalize(channels_data, axis = 0)
    #visulaze time series and frequencies after normalization
    plot_data(markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name + " Normalized", center = False, jupyter_b = jupyter_b)    
    #read, filter, and segement target data 
    return markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, num_of_channels 

def cut_segments(markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, num_of_channels):
    #Cut target segments
    signal_segment_list_target, time_segment_list_target, markers_placement_list_target = get_P300_segment(markers_list, markers_time_stamps, channels_data, time_stamps_data, target='Circle-t')#'Target')#
    if(markers_placement_list_target==[]):
        signal_segment_list_target, time_segment_list_target, markers_placement_list_target = get_P300_segment(markers_list, markers_time_stamps, channels_data, time_stamps_data, target='triangle-t')#'Target')#                
        if(markers_placement_list_target==[]): 
            print("no target")
            exit()
        current_target = "triangle"
    else:
        current_target = "circle"
    if(current_target == "circle"):
        signal_segment_list_other, time_segment_list_other, markers_placement_list_other = get_P300_segment(markers_list, markers_time_stamps, channels_data, time_stamps_data, target='triangle')#'Other')#
    else:
        signal_segment_list_other, time_segment_list_other, markers_placement_list_other = get_P300_segment(markers_list, markers_time_stamps, channels_data, time_stamps_data, target='Circle')#'Other')#
        
    #Cut gapfiller segments
    markers_placement_list_gf = []
    signal_segment_list_gf = []
    signal_segment_list_gf, time_segment_list_gf, markers_placement_list_gf = get_P300_segment(markers_list, markers_time_stamps, channels_data, time_stamps_data, target='gap filler')
    
    #plot target
    plot_each_segment_all_ch(signal_segment_list_target, time_segment_list_target, markers_placement_list_target, 'Target', num_of_channels)
    #plot other 
    #lot_each_segment_all_ch(signal_segment_list_other, time_segment_list_other, markers_placement_list_other, 'Other', num_of_channels)
    #plot gap filler 
    #plot_each_segment_all_ch(signal_segment_list_gf, time_segment_list_gf, markers_placement_list_gf, 'gap filler', num_of_channels)

    #plot average Target and Average data on 
    plot_all_segments_scaled_av_per_ch(signal_segment_list_target, signal_segment_list_other, signal_segment_list_gf, time_segment_list_target[0], markers_placement_list_target[0])
    signal_segment_list_target = baseline_correction(signal_segment_list_target, time_segment_list_target, markers_placement_list_target, num_of_channels)
    signal_segment_list_other = baseline_correction(signal_segment_list_other, time_segment_list_other, markers_placement_list_other, num_of_channels)
    signal_segment_list_gf = baseline_correction(signal_segment_list_gf, time_segment_list_gf, markers_placement_list_gf, num_of_channels)
    plot_all_segments_raw_av_per_ch(signal_segment_list_target, signal_segment_list_other, signal_segment_list_gf, time_segment_list_target[0], markers_placement_list_target[0], recored_file_name, current_target)
    
    #save target segements data 
    save_data(signal_segment_list_target, time_segment_list_target, markers_placement_list_target, 'Target')
    #save other segements
    save_data(signal_segment_list_other, time_segment_list_other, markers_placement_list_other, 'Other')
    

    
#markernames = ['Target', 'Other', 'inter']
if __name__ == '__main__':

    markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, num_of_channels = read_data(RECORDING_FILE)
    markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, num_of_channels = filter_normalize(markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, num_of_channels )
    cut_segments(markers_list, markers_time_stamps, channels_data, time_stamps_data, recored_file_name, num_of_channels)


    

