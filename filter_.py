# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:39:33 2022

@author: marko
"""

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def filter_sig(sig, freq=10, btype = 'lp'):
    print(sig)
    sos = signal.butter(10, freq, btype, fs=1000, output='sos')
    sig = signal.sosfilt(sos, sig)
    return sig