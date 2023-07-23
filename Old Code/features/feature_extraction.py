from preprocessing import *
import numpy as np
import mne
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.signal import detrend
import matplotlib.pyplot as plt
from mne_features.feature_extraction import FeatureExtractor


def get_katz_fd(signal):
    """Calculate Katz fractal dimension"""
    L = np.abs(np.diff(signal)).mean()
    n = len(signal)
    d = np.abs(signal - signal[0]).max()
    return np.log(n) / (np.log(d / L) + np.log(n))


def get_hurst_exponent(signal, max_lag=20):
    """Returns the Hurst Exponent of the time series"""

    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def extract_p300_features(ep):
    """
    Extracts features relevant to the P300 paradigm using MNE.
    Features: mean, ptp, rms, kurtosis, peak amplitude, latency, katz fd, hurst, variance

    Args:
        data (mne epoches): Preprocessed EEG data in the shape of (timestamps * num_channels).
        For Target, no-target and filler.

    Returns:
        epochs_features (numpy array): Arrays of extracted features in the shape of (num_trials, num_features).
        features_list
    """
    features_list = ['mean', 'variance', 'ptp_amp', 'kurtosis', 'rms', 'hurst_exp', 'katz_fd']
    Fs = ep.info['sfreq']
    ep_data = ep.get_data()
    feature_extractor = FeatureExtractor(sfreq=Fs, selected_funcs=features_list)

    epochs_features = feature_extractor.fit_transform(ep_data)
    return epochs_features, features_list


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