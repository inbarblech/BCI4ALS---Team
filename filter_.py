import os
from scipy import signal
import numpy as np
import scipy

PRIO_MARKER = 0.200
POST_MARKER = 0.500

SAMP_FREQ = 125  # Sample frequency (Hz)
NOTCH_FREQ1 = 50.0  # Frequency to be removed from signal (Hz)
NOTCH_FREQ2 = 27.5  # Frequency to be removed from signal (Hz)
NOTCH_FREQ3 = 31.125  # Frequency to be removed from signal (Hz)
QUALITY_FACTOR = 25  # Quality factor
BP_WIND = [0.5, 40]  # window for the bandpass filter

Plot_Path = os.path.join(os.getcwd(), "plots")
Recording_file_name = "debbi_31_1_1"

def remove_fr(sig, stop_f):
    # Create/view notch filter
    b_notch, a_notch = signal.iirnotch(stop_f, QUALITY_FACTOR, SAMP_FREQ)
    outputSignal = signal.filtfilt(b_notch, a_notch, sig)
    return outputSignal


def filter_sig(sig):
    # Apply bandpass filter
    sig = filter_bp(sig)
    # Aply Notch filter
    sig = remove_fr(sig, NOTCH_FREQ1)
    # sig = remove_fr(sig, NOTCH_FREQ2)
    sig = remove_fr(sig, NOTCH_FREQ3)
    return sig


def cut_edges(markers_list, markers_time_stamps, channels_data, time_stamps_data, cut_start=0, cut_stop=0):
    if cut_start > 0 or cut_stop > 0:
        if (cut_start > (time_stamps_data.shape[0] - cut_stop)):
            print("Wrong cutting defintion")
            return -1
        channels_data = channels_data[cut_start:time_stamps_data.shape[0] - cut_stop, :]
        time_stamps_data = time_stamps_data[cut_start:time_stamps_data.shape[0] - cut_stop]
        first_m = 0
        while markers_time_stamps[first_m] < time_stamps_data[0]:
            first_m += 1
        last_m = markers_time_stamps.shape[0] - 1
        while markers_time_stamps[last_m] > time_stamps_data[time_stamps_data.shape[0] - 1]:
            last_m -= 1
        markers_time_stamps = markers_time_stamps[first_m: last_m]
        markers_list = markers_list[first_m: last_m]
    return markers_list, markers_time_stamps, channels_data, time_stamps_data


def baseline_correction(signal_segment_list, time_segment_list, markers_placement_list, num_of_channels):
    for signal_segment, epoch, time_segment, markers_placement in zip(signal_segment_list,
                                                                      range(len(signal_segment_list)),
                                                                      time_segment_list, markers_placement_list):
        baseline_end = 0
        while time_segment[baseline_end] < markers_placement:
            baseline_end += 1
        for channel in range(num_of_channels):
            baseline = np.mean(signal_segment[:baseline_end, channel])
            signal_segment_list[epoch][:][channel] -= baseline
    return signal_segment_list



def filter_bp(sig, l_freq=0.5, h_freq=40, fs=125):
    sos = scipy.signal.butter(10, [l_freq, h_freq], 'bandpass', fs=fs, output='sos')
    filteredSignal = scipy.signal.sosfiltfilt(sos, sig)
    return filteredSignal


def filter_notch(sig, f0, quality_factor, fs=125):
    b_notch, a_notch = scipy.signal.iirnotch(f0, quality_factor, fs)
    filteredSignal = scipy.signal.filtfilt(b_notch, a_notch, sig)
    return filteredSignal


def find_notch_freq(raw_data, th, dist):
    """

    :param:
     raw_data:
     th: peaks' threshold
    :return:
    """
    spectrum = raw_data.compute_psd()
    f = spectrum.freqs
    p = spectrum.get_data()
    peaks_inx, _ = scipy.signal.find_peaks(p.mean(axis=0), threshold=th, distance=dist)
    notch_freq = f[peaks_inx]
    notch_freq = notch_freq[notch_freq > 20]

    print(f'the frequencies chosen for notch filtering are {notch_freq}')
    return notch_freq

