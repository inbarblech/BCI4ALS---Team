import os
import pyxdf
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import scipy
from filter_ import filter_bp, find_notch_freq, filter_notch
from visualizer import plot_raw, plot_erp, plot_epochs, plot_epochs_by_event, plot_frequency_domain
Plot_Path = os.path.join(os.getcwd(), "plots")
Data_path = os.path.join(os.getcwd(), "segmented_data")

Recording_file_name = "debbi_31_1_2"

def data_validation(data):
    # check xdf file content
    if len(data) != 2:
        raise RuntimeError('Unknown data format')

    # check streams types
    y = data[0]['time_series']
    if isinstance(y, list):
        markers_stream = data[0]
        signal_stream = data[1]
    elif isinstance(y, np.ndarray):
        markers_stream = data[1]
        signal_stream = data[0]
    else:
        raise RuntimeError('Unknown format for time_series in data[0]')

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

    # signal_stream['time_series'] = eeg_signal # TODO: --- not sure if I should change it here?

    return markers_stream, eeg_signal


def xdf2mne(fpath, plot_scale=1e-6, plot=False, fname_plot=None):
    """
    :param: fname (string) = file name including path
    :return: raw = MNE raw array with annotations
    """
    data, _ = pyxdf.load_xdf(fpath)
    print('current file is: ' + fpath)

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

    if plot:
        plot_raw(raw, fname_plot, plot_scale=plot_scale, save=True)

    return raw


def remove_bad_channels(raw, bad_chs=[]):
    """
    :param: bad_chs (list) = channels to remove
    :return: raw (MNE raw) without bas channels
    """
    occipital_chs = ['O1', 'O2']
    bad_chs += occipital_chs
    raw.info['bads'] = bad_chs
    raw.drop_channels(raw.info['bads'])
    print(raw.info)
    print(f'{bad_chs} was recognized as bad and removed')


def filtering(raw, lfreq, hfreq, notch_th, notch_dist=50, notch_qf=25, ica_exclude=[0, 1], plot=False, fname_plot=None):
    filtered_data = filter_bp(raw._data, l_freq=lfreq, h_freq=hfreq)
    raw_data_filtered = raw.copy()
    raw_data_filtered._data = filtered_data

    notch_freq = find_notch_freq(raw_data_filtered, th=notch_th, dist=notch_dist)
    for f0 in notch_freq:
        filtered_data = filter_notch(filtered_data, f0, notch_qf)

    raw_data_filtered._data = filtered_data

    raw_data_filtered_ica = ica_processing(raw_data_filtered, ica_exclude, plot=plot)

    if plot:
        plot_frequency_domain(raw, fname_plot+'original', save=True)
        plot_frequency_domain(raw, fname_plot + 'after_filtering', save=True)

    return raw_data_filtered_ica


def ica_processing(raw_data_filtered, ica_exclude, plot=False):
    ica = mne.preprocessing.ICA(n_components=8, max_iter='auto', random_state=97,
                                method='infomax')  # TODO: try different n_components
    raw_data_filtered_ica = raw_data_filtered.copy()
    ica.fit(raw_data_filtered_ica)
    if plot: # # TODO: RINAT - what do we see?
        fig = ica.plot_sources(raw_data_filtered_ica, show_scrollbars=False)
        ica.plot_components()
        fig.savefig(f'{Plot_Path}\\{Recording_file_name}_ICA_sources.jpeg', format='jpeg')

    ica.exclude = ica_exclude  # 0 is blinking, 1 is heartbeats
    ica.apply(raw_data_filtered_ica)
    return raw_data_filtered_ica


def epochs_segmentation(raw, reject_criteria, flat_criteria, t_min=-0.2, t_max=0.5, detrend=1, baseline=(-0.2, 0), plot=False, fname_plot=None):
    """

    :param raw:
    :return:
    """
    # TODO: find reject_criteria, flat_criteria in a general smart way
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events_from_annot, tmin=t_min, tmax=t_max, event_id=event_dict, detrend=detrend,
                        baseline=baseline, reject=reject_criteria, flat=flat_criteria)


    if plot:
        plot_epochs(epochs, fname_plot, save=True)

    print(f'\n dropped epochs:\n{epochs.drop_log}')
    print(epochs)

    return epochs


def erp_segmentation(epochs, target_name, inter_name, plot_epochs=False, plot_erp=False,fname=None, save2csv=False):
    """
    :param epochs:
    :param target_name:
    :param inter_name:
    :param plot_epochs:
    :param plot_erp:
    :param save2csv:
    :return:
    """
    # TODO: make sure what to combine
    event = epochs.event_id
    other_event = [e for e in event if e != target_name and e != inter_name]
    mne.epochs.combine_event_ids(epochs, other_event, {'other': 5}, copy=False)

    ERP = {}
    for event in epochs.event_id:
        ERP[event] = epochs[target_name].average()

    if plot_epochs:
        for event_name in ERP.keys():
            plot_epochs_by_event(epochs, event_name, fname, save=True)

    if plot_erp:
        for event_name in ERP.keys():
            plot_erp(epochs, event_name, fname, save=True)

    if save2csv:
        for event_name in ERP.keys():
            save_erp_data(ERP[event_name], event_name,fname)

    object_list = list(ERP.values())

    return object_list[0], object_list[1], object_list[2]


def save_erp_data(erp, erp_name, fname):
    data = erp.data
    data = np.insert(data, 0, erp.times, axis=0)
    header = erp.ch_names.copy()
    header.insert(0, 'Time')
    #df = pd.DataFrame(data.T, index=erp.times, columns=header)

    df = pd.DataFrame(data.T, columns=header)
    df.to_csv(f'{Data_path}\\{fname}_{erp_name}.csv', index=True, header=True)
