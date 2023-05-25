import os
import pyxdf
import mne
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy
from feature_extraction import extract_p300_features
from preprocessing import xdf2mne, remove_bad_channels, filtering, epochs_segmentation, erp_segmentation, data4eegnet


Plot_flag = True
Save_flag = True

L_freq = 0.5
H_freq = 40

rec_path = os.path.join(os.path.join(os.getcwd(), os.pardir), "Recordings")
if  not os.path.isdir(rec_path):
            os.makedirs(rec_path)
            print(f'Created new data folder: {rec_path}\n'
                  f'no new recordings to preprocess.')
recordings = os.listdir(rec_path)  # list of all recordings files
recordings = [rec for rec in recordings if rec[-3:] == "xdf"]

if __name__ == '__main__':
    for rec in recordings:
        Fpath = os.path.join(rec_path, rec)
        f_name = rec[:-4]  # remove .xdf
        raw_data = xdf2mne(Fpath, f_name, plot=Plot_flag)
        remove_bad_channels(raw_data, interpolate=False)
        # filtering
        raw_filtered = filtering(raw_data, L_freq, H_freq, notch_dist=10, notch_qf=25, ica_exclude=[0, 1], plot=Plot_flag, fname_plot=f_name)
        epochs_data = epochs_segmentation(raw_filtered, plot=Plot_flag, fname=f_name, save2csv=Save_flag)

        target_x, other_x, gf_x = data4eegnet(epochs_data)
        target_erp, inter_erp, other_erp = erp_segmentation(epochs_data, plot=True, save2csv=Save_flag, fname=f_name)


        # Extract features from erp preprocessed data
        filler_features = extract_p300_features(inter_erp)
        target_features = extract_p300_features(target_erp)
        non_target_features = extract_p300_features(other_erp)

        move_to = os.path.join(rec_path, "processed")
        if not os.path.isdir(rec_path):
            os.makedirs(rec_path)
        os.rename(Fpath, os.path.join(move_to, f_name))
