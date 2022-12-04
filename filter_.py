# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:39:33 2022

@author: marko
"""

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
SAMP_FREQ = 1000  # Sample frequency (Hz)
NOTCH_FREQ = 50.0  # Frequency to be removed from signal (Hz)
QUALITY_FACTOR = 30.0  # Quality factor
BP_WIND = [0.5,40] #window for the bandpass filter 

def filter_sig(sig, freq = BP_WIND, btype = 'bandpass', a_notch = NOTCH_FREQ):
    #Apply bandpass filter
    sos = signal.butter(10, freq, btype, fs=SAMP_FREQ, output='sos')    
    sig = signal.sosfilt(sos, sig)
    #Aply Notch filter 
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, QUALITY_FACTOR, SAMP_FREQ)
    sig = signal.filtfilt(b_notch, a_notch, sig)
    return sig