import os
import numpy as np
import pyxdf

PRIO_MARKER = 0.200
POST_MARKER = 0.500

def get_P300_segment(recording_folder = os.path.join(os.getcwd(), "Recordings\\EGI.xdf")):
    data, header = pyxdf.load_xdf(recording_folder)
    
    #get data from markers streem
    line_= data[0] 
    markers_list = np.array(line_["time_series"])
    markers_time_stamps = np.array(line_["time_stamps"])
    
    #get data from GUI streem
    line_= data[1] 
    channels_data = np.array(line_["time_series"])
    time_stamps_data = np.array(line_["time_stamps"])
    
    #cut the relevant signal accroding to the markers
    start_marker_search_index = 0
    signal_segment_list = [] 
    
    for i in range(markers_list.shape[0]):
        #if(markers_list[i] == 'Target') #currently EGI.xdf has "Circle". Replace with Target once recored
        if(markers_list[i] == 'Circle'):
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
            signal_segment_list.append(signal_segment)
    return  signal_segment_list    


if __name__ == '__main__':
    get_P300_segment()
