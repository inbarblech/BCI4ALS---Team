import numpy as np
import mne
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.signal import detrend
import matplotlib.pyplot as plt

def katz_fd(signal):
    """Calculate Katz fractal dimension"""
    L = np.abs(np.diff(signal)).mean()
    n = len(signal)
    d = np.abs(signal - signal[0]).max()
    return np.log(n) / (np.log(d / L) + np.log(n))

def hurst_exponent(signal):
    """Calculate Hurst exponent"""
    signal = detrend(signal)
    H, c, data = compute_Hc(signal, kind='change')
    return H

def extract_p300_features(target, non_target, filler):
    """
    Extracts features relevant to the P300 paradigm using MNE.
    Features: mean, ptp, rms, kurtosis, peak amplitude, latency, katz fd, hurst, variance

    Args:
        data (mne evoked): 3 files of EEG data in the shape of (timestamps * num_channels).
        For Target, no-target and filler.

    Returns:
        features (numpy array): 3 Arrays of extracted features in the shape of (num_trials, num_features).
    """

    # load evoked data
    evoked_data_target = mne.read_evokeds(target)[0]
    evoked_data_non_target = mne.read_evokeds(non_target)[0]
    evoked_data_filler = mne.read_evokeds(filler)[0]

    # calculate mean across all channels
    """not sure if '.data' is necessary, it depends on the data structure of the files.
    same goes for all the code"""
    mean_data_target = np.mean(evoked_data_target.data, axis=1)
    mean_data_non_target = np.mean(evoked_data_non_target.data, axis=1)
    mean_data_filler = np.mean(evoked_data_filler.data, axis=1)

    # calculate PTP amplitude across all channels
    ptp_data_target = np.ptp(evoked_data_target.data, axis=1)
    ptp_data_non_target = np.ptp(evoked_data_non_target.data, axis=1)
    ptp_data_filler = np.ptp(evoked_data_filler.data, axis=1)

    # calculate RMS across all channels
    squared_data_target = np.square(evoked_data_target.data)
    rms_data_target = np.sqrt(np.mean(squared_data_target, axis=1))
    squared_data_non_target = np.square(evoked_data_non_target.data)
    rms_data_non_target = np.sqrt(np.mean(squared_data_non_target, axis=1))
    squared_data_filler = np.square(evoked_data_filler.data)
    rms_data_filler = np.sqrt(np.mean(squared_data_filler, axis=1))

    # calculate kurtosis across all channels
    kurtosis_data_target = stats.kurtosis(evoked_data_target.data, axis=1)
    kurtosis_data_non_target = stats.kurtosis(evoked_data_non_target.data, axis=1)
    kurtosis_data_filler = stats.kurtosis(evoked_data_filler.data, axis=1)

    # calculate peak of signal
    peaks_target, _ = find_peaks(evoked_data_target.data, height=0)
    peak_amplitude_target = evoked_data_target[peaks_target].max()
    peaks_non_target, _ = find_peaks(evoked_data_non_target.data, height=0)
    peak_amplitude_non_target = evoked_data_non_target[peaks_non_target].max()
    peaks_filler, _ = find_peaks(evoked_data_filler.data, height=0)
    peak_amplitude_filler = evoked_data_target[peaks_filler].max()

    # calculate latency (time to peak from onset of stimulus)
    latency_target = evoked_data_target.times[np.argmax(evoked_data_target.data)]
    latency_non_target = evoked_data_non_target.times[np.argmax(evoked_data_non_target.data)]
    latency_filler = evoked_data_filler.times[np.argmax(evoked_data_filler.data)]

    # Define the time window of interest and extract data to signal
    tmin, tmax = 0.2, 0.6
    p300_signal_target = evoked_data_target.copy().crop(tmin=tmin, tmax=tmax).data[0]
    p300_signal_non_target = evoked_data_non_target.copy().crop(tmin=tmin, tmax=tmax).data[0]
    p300_signal_filler = evoked_data_filler.copy().crop(tmin=tmin, tmax=tmax).data[0]

    # Calculate Katz fractal dimension
    katz_fd_target = katz_fd(p300_signal_target)
    katz_fd_non_target = katz_fd(p300_signal_non_target)
    katz_fd_filler = katz_fd(p300_signal_filler)

    # Calculate Hurst exponent
    hurst_target = hurst_exponent(p300_signal_target)
    hurst_non_target = hurst_exponent(p300_signal_non_target)
    hurst_filler = hurst_exponent(p300_signal_filler)

    # Calculate variance
    variance_target = np.var(p300_signal_target)
    variance_non_target = np.var(p300_signal_non_target)
    variance_filler = np.var(p300_signal_filler)

    # stack all the feature arrays into a single numpy array
    feature_array_target = np.stack((mean_data_target, ptp_data_target, rms_data_target, kurtosis_data_target,
                                     peak_amplitude_target, latency_target, katz_fd_target, hurst_target, variance_target))
    feature_array_non_target = np.stack((mean_data_non_target, ptp_data_non_target, rms_data_non_target, kurtosis_data_non_target,
                                     peak_amplitude_non_target, latency_non_target, katz_fd_non_target, hurst_non_target, variance_non_target))
    feature_array_filler = np.stack((mean_data_filler, ptp_data_filler, rms_data_filler, kurtosis_data_filler,
                                     peak_amplitude_filler, latency_filler, katz_fd_filler, hurst_filler, variance_filler))

    # transpose the feature array to make electrodes the rows and features the columns
    feature_array_target = feature_array_target.T
    feature_array_non_target = feature_array_non_target.T
    feature_array_filler = feature_array_filler.T

    plot_features(mean_data_target, ptp_data_target, rms_data_target, kurtosis_data_target, peaks_target,latency_target,katz_fd_target,hurst_target,variance_target)

    return feature_array_target, feature_array_non_target, feature_array_filler

def plot_features(mean, ptp, rms, kurtosis, peak_amp, latency, katz_fd, hurst_exp, variance):
    """
    Plots the extracted features on a single figure.

    Parameters
    ----------
    mean : float
        The mean amplitude of the P300 signal.
    ptp : float
        The peak-to-peak amplitude of the P300 signal.
    rms : float
        The root-mean-square amplitude of the P300 signal.
    kurtosis : float
        The kurtosis of the P300 signal.
    peak_amp : float
        The peak amplitude of the P300 signal.
    latency : float
        The latency of the P300 signal.
    katz_fd : float
        The Katz fractal dimension of the P300 signal.
    hurst_exp : float
        The Hurst exponent of the P300 signal.
    variance : float
        The variance of the P300 signal.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(["Mean", "Peak-to-Peak", "RMS", "Kurtosis", "Peak Amp", "Latency", "Katz FD", "Hurst Exp", "Variance"],
           [mean, ptp, rms, kurtosis, peak_amp, latency, katz_fd, hurst_exp, variance])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Feature Value")
    ax.set_title("P300 Feature Extraction Results")
    plt.show()

if __name__ == '__main__':
    # Load segmented EEG data
    'Must change name of file to match the output of the preprocessing function'
    eeg_data_target = np.load('eeg_file')
    eeg_data_non_target = np.load('eeg_file')
    eeg_data_filler = np.load('eeg_file')
    features = extract_p300_features(eeg_data_target, eeg_data_non_target, eeg_data_filler)