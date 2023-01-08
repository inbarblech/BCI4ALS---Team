# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:39:33 2022

@author: marko
"""
PRIO_MARKER = 0.200
POST_MARKER = 0.500
from scipy import signal
import numpy as np

SAMP_FREQ = 125  # Sample frequency (Hz)
NOTCH_FREQ1 = 50.0  # Frequency to be removed from signal (Hz)
NOTCH_FREQ2 = 27.5  # Frequency to be removed from signal (Hz)
NOTCH_FREQ3 = 31.125  # Frequency to be removed from signal (Hz)
QUALITY_FACTOR = 25  # Quality factor
BP_WIND = [0.5,40] #window for the bandpass filter 

def filter_bp(sig):
    
    sos = signal.butter(10, BP_WIND, 'bandpass', fs=SAMP_FREQ, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered

def remove_fr(sig, stop_f):
    # Create/view notch filter
    b_notch, a_notch = signal.iirnotch(stop_f, QUALITY_FACTOR, SAMP_FREQ)
    outputSignal = signal.filtfilt(b_notch, a_notch, sig)
    return outputSignal
    

def filter_sig(sig):
    #Apply bandpass filter
    sig = filter_bp(sig)      
    #Aply Notch filter 
    sig = remove_fr(sig, NOTCH_FREQ1)
    #sig = remove_fr(sig, NOTCH_FREQ2)    
    sig = remove_fr(sig, NOTCH_FREQ3)  

  
    return sig

def cut_edges(markers_list, markers_time_stamps, channels_data, time_stamps_data, cut_start = 0, cut_stop= 0):
    if cut_start>0 or cut_stop >0:
        if(cut_start > (time_stamps_data.shape[0]-cut_stop)):
            print("Wrong cutting defintion")
            return -1        
        channels_data = channels_data[cut_start:time_stamps_data.shape[0]-cut_stop,:]
        time_stamps_data = time_stamps_data[cut_start:time_stamps_data.shape[0]-cut_stop]
        first_m = 0
        while markers_time_stamps[first_m]<time_stamps_data[0]:
            first_m +=1
        last_m = markers_time_stamps.shape[0]-1
        while markers_time_stamps[last_m]>time_stamps_data[time_stamps_data.shape[0] - 1]:
            last_m -=1
        markers_time_stamps = markers_time_stamps[first_m: last_m]
        markers_list = markers_list[first_m: last_m]
    return markers_list, markers_time_stamps, channels_data, time_stamps_data

def baseline_correction(signal_segment_list, time_segment_list, markers_placement_list, num_of_channels):
    for signal_segment, epoch, time_segment, markers_placement in zip(signal_segment_list, range(len(signal_segment_list)), time_segment_list, markers_placement_list):
        baseline_end = 0
        while time_segment[baseline_end]<markers_placement:
            baseline_end +=1
        for channel in range(num_of_channels):
            baseline = np.mean(signal_segment[:baseline_end,channel])
            signal_segment_list[epoch][:][channel] -= baseline
    return signal_segment_list
