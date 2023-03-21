import numpy as np
from pylsl import StreamInlet, resolve_stream, local_clock
import mne
from collections import defaultdict

def read_from_lsl():
    # resolve an EEG stream on the lab network
    print("looking for online data streams...")
    streams = resolve_stream()
    # create a new inlet to read from the streams
    if len(streams) == 2:
        inlet0 = StreamInlet(streams[0])
        inlet1 = StreamInlet(streams[1])
    else:
        raise RuntimeError(f'found {len(streams)} streams instead of 2, check LabRecorder and start again')

    # create a dict inorder to save the data
    keys = ['info', 'time_series', 'time_stamps']

    eeg_stream = dict.fromkeys(keys, [])
    markers_stream = dict.fromkeys(keys, [])

    eeg_data = list()
    markers_data = list()
    eeg_time = list()
    markers_time = list()
    cur_marker = ''

    while cur_marker != 'block start':
        data0, timestamp0 = inlet0.pull_sample()
        data1, timestamp1 = inlet1.pull_sample()
        print(f'time:{timestamp0} \n data0={data0}\n time:{timestamp1} \n data1={data1}\n')
        if len(data0) > 1:
            eeg_data.append(data0)
            eeg_time.append(timestamp0)
            markers_data.append(data1)
            markers_time.append(timestamp1)
        else:
            eeg_data.append(data1)
            eeg_time.append(timestamp1)
            markers_data.append(data0)
            markers_time.append(timestamp0)

        eeg_stream['time_series'] = np.array(eeg_data)
        eeg_stream['time_stamps'] = np.array(eeg_time)
        markers_stream['time_series'] = markers_data
        markers_stream['time_stamps'] = np.array(markers_time)

        cur_marker = markers_stream['time_series'][-1][0]

    info_generate(eeg_stream, markers_stream)
    raw_stream = create_raw_from_data_streams(eeg_stream, markers_stream)

    return raw_stream

def info_generate(eeg_stream, markers_stream):
    eeg_info = dict(name=['obci_eeg1'], type=['EEG'], channel_count=[eeg_stream['time_series'].shape[1]],
                    nominal_srate=[round(1 / (eeg_stream['time_stamps'][1:] - eeg_stream['time_stamps'][:-1]).mean())],
                    channel_format=['float32'], source_id=['openbcigui'], created_at=[eeg_stream['time_stamps'][0]])
    markers_info = dict(name=['MyMarkerStream'], type=['Markers'], markers_count=[len(markers_stream['time_series'])],
                        first_marker=markers_stream['time_stamps'][0], end_block=markers_stream['time_stamps'][-1],
                        channel_format=['string'], source_id=['myuidw43536'])

    eeg_stream['info'] = defaultdict(list, eeg_info)
    markers_stream['info'] = defaultdict(list, markers_info)


def create_raw_from_data_streams(eeg_stream, markers_stream):
    # create mne raw object with annotations
    events = [e[0] for e in markers_stream['time_series']]
    ''' start_time = eeg_stream['time_stamps'][0]
        markers_stream['time_stamps'] -= start_time'''
    annotations = mne.Annotations(markers_stream['time_stamps'], np.zeros(len(markers_stream['time_stamps'])), events)

    # set EEG values & clean empty channels
    eeg_signal = eeg_stream['time_series'].T
    eeg_signal = [ch * 1e-6 for ch in eeg_signal if ch.any()]  # change uV to V

    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    if eeg_stream['info']['channel_count'] == 13:
        ch_names = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'O1', 'O2']
    else: # synthetic data
        ch_names = [str(i) for i in range(eeg_stream['info']['channel_count'][0])]

    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    if eeg_stream['info']['channel_count'] == 13:
        info.set_montage('standard_1020')

    raw = mne.io.RawArray(eeg_signal, info)
    raw.set_annotations(annotations)

    return raw


if __name__ == '__main__':
    raw = read_from_lsl()
    raw.plot(scalings=dict(eeg=100e-6))
