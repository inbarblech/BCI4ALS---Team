import os
import pyxdf
import mne
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy
from feature_extraction import extract_p300_features
from preprocessing import xdf2mne, remove_bad_channels, filtering, epochs_segmentation, erp_segmentation, data4eegnet

path = os.path.join(os.getcwd(), "Recordings")
recordings = os.listdir(path)  # list of all recordings files
recordings = [fname for fname in recordings if fname[-3:] == "xdf"]
Plot_flag = False
Save_flag = False

# for fname in recordings:
Fname = recordings[1]
Fpath = os.path.join(path, Fname)
Fname = Fname[:-4]   # remove .xdf

L_freq = 0.5
H_freq = 40
Target = 'triangle-t'

if __name__ == '__main__':
    raw_data = xdf2mne(Fpath, plot_scale=1e-2, plot=Plot_flag, fname_plot=Fname)
    remove_bad_channels(raw_data)
    # filtering
    raw_filtered = filtering(raw_data, L_freq, H_freq, notch_th=1e-12, notch_dist=10, notch_qf=25, ica_exclude=[0, 1], plot=Plot_flag, fname_plot=Fname)
    reject_criteria = dict(eeg=1000e-6)  # 1000 µV
    flat_criteria = dict(eeg=0.01e-6)  # 0.01 µV
    epochs_data = epochs_segmentation(raw_filtered, Target,
                                      reject_criteria=reject_criteria, flat_criteria=flat_criteria,
                                      plot=Plot_flag, fname=Fname, save2csv=Save_flag)

    target_x, other_x, gf_x = data4eegnet(epochs_data)

    target_erp, inter_erp, other_erp = erp_segmentation(epochs_data, plot=True, save2csv=Save_flag, fname=Fname)


    # Extract features from erp preprocessed data
    filler_features = extract_p300_features(inter_erp)
    target_features = extract_p300_features(target_erp)
    non_target_features = extract_p300_features(other_erp)
