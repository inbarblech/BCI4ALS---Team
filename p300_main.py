import os
from features.feature_extraction import extract_p300_features
from preprocessing import xdf2mne, remove_bad_channels, filtering, epochs_segmentation, erp_segmentation, data4eegnet

Plot_flag = True
Save_flag_csv = False
Save_flag_np = True
Calc_Features = False

L_freq = 0.5
H_freq = 40

rec_path = os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), "BCI_data"), "Recordings")
if not os.path.isdir(rec_path):
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
        # filtering
        raw_filtered = filtering(raw_data, L_freq, H_freq, notch_dist=10, notch_qf=25, ica_exclude=[0, 1],
                                 plot=Plot_flag, fname_plot=f_name)
        raw_clean = remove_bad_channels(raw_filtered, interpolate=True)

        epochs_data = epochs_segmentation(raw_clean, plot=Plot_flag, fname=f_name, save2csv=Save_flag_csv)

        gf_x, other_x, target_x = data4eegnet(epochs_data, f_name, to_save=Save_flag_np)
        gf_erp, other_erp, target_erp = erp_segmentation(epochs_data, plot=Plot_flag, save2csv=Save_flag_csv, fname=f_name)


        if Calc_Features: # Extract features from erp preprocessed data
            filler_features = extract_p300_features(gf_erp)
            target_features = extract_p300_features(target_erp)
            non_target_features = extract_p300_features(other_erp)

        move_to = os.path.join(rec_path, "processed")
        if not os.path.isdir(rec_path):
            os.makedirs(rec_path)
        os.rename(Fpath, os.path.join(move_to, f_name))
