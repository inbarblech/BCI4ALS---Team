import os
import pyxdf
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy
from autoreject import AutoReject, get_rejection_threshold
from pyprep.find_noisy_channels import NoisyChannels
from filter_ import filter_bp, find_notch_freq, filter_notch
from visualizer import plot_raw, plot_erp, plot_epochs, plot_epochs_by_event, plot_frequency_domain, plot_erp_compare

def create_data_folder(rec_name):
    global Data_Path, Plots_Path, rec_plots, Segmented_Data_Path, EEGnet_Path
    Data_Path = os.path.join(os.path.join(os.getcwd(), os.pardir), "BCI_data")
    Plots_Path = os.path.join(Data_Path, "plots")
    rec_plots = os.path.join(Plots_Path, rec_name)
    Segmented_Data_Path = os.path.join(Data_Path, "segmented_data")
    EEGnet_Path = os.path.join(Segmented_Data_Path, "for_EEGNET")
    dirs = [Data_Path, Plots_Path, Segmented_Data_Path,rec_plots, EEGnet_Path]
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)
            print(f'Created new data folder: {d}')


def data_validation(data):
    # check xdf file content
    if len(data) != 2:
        raise RuntimeError('Unknown data format')

    # check streams types
    first_stream_type = data[0]['info']['type']
    if first_stream_type[0] == 'Markers':
        markers_stream = data[0]
        signal_stream = data[1]
    elif first_stream_type[0] == 'EEG':
        markers_stream = data[1]
        signal_stream = data[0]
    else:
        raise RuntimeError('Unknown type for the first stream in data')

    # check number of channels
    n_chs = int(signal_stream['info']['channel_count'][0])
    if n_chs != 16:
        warnings.warn(f'there are {n_chs} channels in that recording')
    else:
        print(f'the recording is valid with {len(data)} streams and {n_chs} channels')

    return markers_stream, signal_stream


def scale_data(markers_stream, signal_stream):
    # set markers time [s]
    start_time = signal_stream['time_stamps'][0]
    markers_stream['time_stamps'] -= start_time

    # set EEG values & clean empty channels
    eeg_signal = signal_stream['time_series'].T
    eeg_signal = [ch * 1e-6 for ch in eeg_signal if ch.any()]  # change uV to V

    return markers_stream, eeg_signal

def cut_edges(raw):
    t_min = raw.annotations.onset[0]
    t_max = raw.annotations.onset[-1]
    raw.crop(tmin=t_min-0.5, tmax=t_max+0.5)


def xdf2mne(fpath, fname, plot=False, plot_scale=1e-3):
    """
    :param: fname (string) = file name including path
    :return: raw = MNE raw array with annotations
    """
    create_data_folder(fname)
    data, _ = pyxdf.load_xdf(fpath)
    print('\n--------------------------------\n'
          'current file is: ' + os.path.basename(fpath) +
          '\n--------------------------------')

    # data validation and scales
    markers, signal_rec = data_validation(data)
    markers, eeg_signal = scale_data(markers, signal_rec)

    # create mne raw object with annotations
    events = [e[0] for e in markers['time_series']]
    annotations = mne.Annotations(markers['time_stamps'],
                                  np.zeros(len(markers['time_stamps'])),
                                  events)
    sfreq = float(signal_rec['info']['nominal_srate'][0])
    ch_names = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'O1', 'O2']
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    info.set_montage('standard_1020')

    raw = mne.io.RawArray(eeg_signal, info)
    raw.set_annotations(annotations)
    mne.datasets.eegbci.standardize(raw)
    raw.set_eeg_reference(ref_channels="average")
    cut_edges(raw)
    raw._filenames = [fname]
    if plot:
        plot_raw(raw, fname, plot_scale=plot_scale, save=True)

    return raw


def remove_bad_channels(raw, interpolate=False, remove_o=True):
    print('\nSearching for bad channels using Pyprep...')
    while raw.info['bads']:
        nd = NoisyChannels(raw, do_detrend=True)
        nd.find_all_bads(ransac=False)
        bad_chs = list(set(nd.get_bads()))
        raw.info['bads'] = list(set(bad_chs + raw.info['bads']))
        print(f"recognized {raw.info['bads']} as bad")
        stat = 'removed'
        if interpolate:
            raw = raw.interpolate_bads(reset_bads=True)
            stat = 'interpolated'
        raw.plot(scalings=dict(eeg=1e-4))

    occipital_chs = [ch for ch in ['O1','O2'] if ch in raw.ch_names]
    if len(occipital_chs) != 0:
        if remove_o:
            raw.drop_channels(occipital_chs)
            print('Occipital channels where removed')

    print(raw.info)
    print(f'{bad_chs} was recognized as bad and {stat}')
    return raw

def filtering(raw, lfreq, hfreq, notch_dist=50, notch_qf=25, run_ica=True, ica_exclude=[0,1], plot=False, fname_plot=''):
    filtered_data = filter_bp(raw._data, l_freq=lfreq, h_freq=hfreq)
    raw_data_filtered = raw.copy()
    raw_data_filtered._data = filtered_data

    notch_freq = find_notch_freq(raw_data_filtered, dist=notch_dist)
    for f0 in notch_freq:
        filtered_data = filter_notch(filtered_data, f0, notch_qf)

    raw_data_filtered._data = filtered_data
    if run_ica:
        raw_data_filtered_ica = ica_processing(raw_data_filtered, ica_exclude, plot=plot,fname_plot=fname_plot)
        raw_data_filtered = raw_data_filtered_ica

    if plot:
        plot_frequency_domain(raw, fname_plot, f'{fname_plot}_original', save=True)
        plot_frequency_domain(raw_data_filtered, fname_plot,  f'{fname_plot}_after_filtering', save=True)

    return raw_data_filtered


def ica_processing(raw_data_filtered, ica_exclude, plot=False, fname_plot=''):
    n = len(raw_data_filtered.ch_names)-len(raw_data_filtered.info['bads'])
    if n > 8:
        n = 8
    ica = mne.preprocessing.ICA(n_components=n, max_iter='auto', random_state=97, method='fastica')
    raw_data_filtered_ica = raw_data_filtered.copy()
    ica.fit(raw_data_filtered_ica)
    if plot:
        dir = os.path.join(Plots_Path, fname_plot)
        fig1 = ica.plot_sources(raw_data_filtered_ica)
        fig2 = ica.plot_components()
        fig1.savefig(os.path.join(dir, f'{fname_plot}_ICA_sources.jpeg'), format='jpeg')
        fig2[0].savefig(os.path.join(dir, f'{fname_plot}_ICA_topo.jpeg'), format='jpeg')

    ica.exclude = ica_exclude  # 0 is blinking, 1 is heartbeats
    ica.apply(raw_data_filtered_ica)
    return raw_data_filtered_ica


def epochs_segmentation(raw, reject_criteria_p=0.80, flat_criteria_v=1e-6, t_min=-0.2, t_max=0.5,
                        detrend=1, baseline=(-0.2, 0), auto_reject=True, plot=False, fname='', save2csv=False):
    raw.plot(scalings=dict(eeg=5e-5))
    original_bads = deepcopy(raw.info['bads'])
    raw_drop = raw.copy().drop_channels(original_bads)
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    excludes = [event_dict[i] for i in event_dict.keys() if i in ['all done', 'blank', 'block end']]
    events_include = mne.pick_events(events_from_annot, exclude=excludes)
    target = [key for key in event_dict.keys() if key.endswith('-t')][0]

    events_include_dict = {key: value for key, value in event_dict.items() if value not in excludes and key != target}
    events_include_dict['target'] = event_dict[target]
    other = [e for e in events_include_dict if e not in ['target', 'gap filler']]
    if len(other) == 1:
        other_n = other[0]
        events_include_dict['other'] = events_include_dict[other_n]
        events_include_dict.pop(other_n)

    reject_criteria = dict(eeg=np.abs(raw_drop._data).max()*reject_criteria_p)  # 100 µV
    flat_criteria = dict(eeg=flat_criteria_v)  # 1 µV

    epochs = mne.Epochs(raw, events_include, tmin=t_min, tmax=t_max, event_id=events_include_dict, detrend=detrend,
                        baseline=baseline, reject=reject_criteria, preload=True, flat=flat_criteria)

    if len(other) > 1:  # in case of multiple other events
        mne.epochs.combine_event_ids(epochs, other, {'other': max(events_include_dict.values())+1}, copy=False)

    print(f"New Annotations descriptions: {epochs.event_id.keys()}")
    if auto_reject:
        print('\nStarting epochs AutoReject, might take a while...')
        ar = AutoReject()
        epochs_clean = ar.fit_transform(epochs)
        reject = get_rejection_threshold(epochs)
        print(f'\nReject epochs voltage:{reject}')
    else:
        epochs_clean = epochs
    print("Manual option was selected, please mark bad epochs on the following plot")
    epochs_clean.plot(scalings=dict(eeg=1e-4))
    epochs_clean.drop_bad()
    epochs_clean = epochs_clean.interpolate_bads(reset_bads=True)

    for event_name in epochs_clean.event_id:
        if save2csv:
            save_epochs_data(epochs_clean, event_name, fname)
        if plot:
            plot_epochs_by_event(epochs_clean, event_name, fname, save=True)

    if plot:
        plot_epochs(epochs_clean, fname, save=True)

    print(f'\n Dropped epochs:\n{epochs_clean.drop_log}')
    print(epochs_clean)

    return epochs_clean


def erp_segmentation(epochs, plot=False, fname='', save2csv=False):
    ERPs = {}
    for event in epochs.event_id:
        ERPs[event] = epochs[event].average()
    sorted(ERPs)
    if plot:
        plot_erp_compare(ERPs, fname, save=True)
        for event_name in ERPs.keys():
            plot_erp(ERPs, event_name, fname, save=True)

    if save2csv:
        for event_name in ERPs.keys():
            save_erp_data(ERPs[event_name], event_name, fname)

    object_list = list(ERPs.values())
    return object_list[0], object_list[1], object_list[2]


def save_erp_data(erp, erp_name, fname):
    data = erp.data
    data = np.insert(data, 0, erp.times, axis=0)
    header = erp.ch_names.copy()
    header.insert(0, 'Time')

    df = pd.DataFrame(data.T, columns=header)
    dir = os.path.join(Segmented_Data_Path, erp_name)
    file_path = os.path.join(dir, f'{fname}_{erp_name}_ERP.csv')
    df.to_csv(file_path, index=True, header=True)


def save_epochs_data(epochs, event_name, fname):
    dir = os.path.join(Segmented_Data_Path, event_name)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    df = epochs.to_data_frame()
    epochs_num = df.epoch.unique()
    for i in epochs_num:
        df_i = df[df.epoch==i].copy()
        df_i.drop(columns=['condition', 'epoch'], inplace=True)
        file_path = os.path.join(dir, f'{fname}_epoch_num_{i}.csv')
        df_i.to_csv(file_path, index=False, header=True)


def data4eegnet(epochs_data, fname, to_save=False):
    conds = list(epochs_data.event_id.keys())
    conds.sort()
    arrays = []
    for cond in conds:
        cond_x = epochs_data[cond]._data
        cond_x = cond_x[:, :, :, np.newaxis]
        cond_x = np.swapaxes(cond_x, 1, 3)
        if to_save:
            dir = os.path.join(EEGnet_Path, cond)
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print(f'Created new folder of {cond} data for EEGNET')
            save_name = os.path.join(dir, fname)
            np.save(save_name, cond_x)
        arrays.append(cond_x)

    return arrays


