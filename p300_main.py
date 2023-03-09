import os
import pyxdf
import mne
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy
from preprocessing import xdf2mne, remove_bad_channels, filtering, epochs_segmentation, erp_segmentation

path = os.path.join(os.getcwd(), "Recordings")
recordings = os.listdir(path)  # list of all recordings files
recordings = [fname for fname in recordings if fname[-3:] == "xdf"]
Plot_flag = True
Save_flag = True

# for fname in recordings:
Fname = recordings[1]
Fpath = path + '\\' + Fname

L_freq = 0.5
H_freq = 40
Target = 'triangle-t'
Inter = 'blank'  # TODO: make sure

if __name__ == '__main__':
    raw_data = xdf2mne(Fpath, plot_scale=1e-2, plot=Plot_flag)
    # TODO: find bad channels (maxwell/pyprep?)
    remove_bad_channels(raw_data)
    # filtering
    raw_filtered = filtering(raw_data, L_freq, H_freq, notch_th=1e-12,notch_dist=10, notch_qf=25, ica_exclude=[0, 1], plot=Plot_flag)
    # reject_criteria = dict(eeg=100e-6)  # 100 µV
    # flat_criteria = dict(eeg=1e-7)  # 1 µV
    epochs_data = epochs_segmentation(raw_filtered, reject_criteria=None, flat_criteria=None, plot=Plot_flag)
    target_erp, inter_erp, other_erp = erp_segmentation(epochs_data, Target, Inter, plot_epochs=Plot_flag, plot_erp=Plot_flag, save2csv=Save_flag)


