import numpy as np
from pylsl import StreamInlet, resolve_streams
import mne
from collections import defaultdict
from multiprocessing import Process, Pipe

import p300_light_on_off as paradigm
from preprocessing import remove_bad_channels, filtering, epochs_segmentation_online, erp_segmentation, data4eegnet
from EEGNETprediction import EEGNET_predict_target

def read_from_lsl(multiprocessing_, conn1, conn3):
    if(multiprocessing_): conn1.recv()
    
    # resolve an EEG stream on the lab network
    print("looking for online data streams...")
    streams = resolve_streams(wait_time=0.004)
    # create a new inlet to read from the streams
    if len(streams) == 2:
        inlet0 = StreamInlet(streams[0])
        inlet1 = StreamInlet(streams[1])
    else:
        raise RuntimeError(f'found {len(streams)} streams instead of 2, check LabRecorder and start again')

    # create a dict inorder to save the data
    keys = ['info', 'time_series', 'time_stamps']

    cur_marker = ''
    while(cur_marker != 'all done'):
        eeg_stream = dict.fromkeys(keys, [])
        markers_stream = dict.fromkeys(keys, [])

        eeg_data = list()
        markers_data = list()
        eeg_time = list()
        markers_time = list()
        cur_marker = ''
        while cur_marker != 'block end' and cur_marker != 'all done':
            data0, timestamp0 = inlet0.pull_sample(timeout=0.004) #timout is mandatory for not loosing EEG data between markers
            data1, timestamp1 = inlet1.pull_sample(timeout=0.004)
            if(data0!=None and data1!=None): print(f'time:{timestamp0} \n data0={data0}\n time:{timestamp1} \n data1={data1}\n')
            
            if(data0!=None and data1!=None):
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
            elif(data0!=None):
                if len(data0) > 1:
                    eeg_data.append(data0)
                    eeg_time.append(timestamp0)
                else:
                    markers_data.append(data0)
                    markers_time.append(timestamp0)
            elif(data1!=None):
                if len(data1) > 1:
                    eeg_data.append(data1)
                    eeg_time.append(timestamp1)
                else:
                    markers_data.append(data1)
                    markers_time.append(timestamp1)
            else:
                print("Empty stream")   
            if(len(markers_data)>0): 
                cur_marker =   markers_data[len(markers_data)-1][0]   

        if(cur_marker == 'all done'): break
        eeg_stream['time_series'] = np.array(eeg_data)
        eeg_stream['time_stamps'] = np.array(eeg_time)
        markers_stream['time_series'] = markers_data
        markers_stream['time_stamps'] = np.array(markers_time)

        print(eeg_stream['time_series'].shape, eeg_stream['time_stamps'].shape, len(markers_stream['time_series']), len(markers_stream['time_stamps']))
        print(markers_stream['time_series'])
        info_generate(eeg_stream, markers_stream)
        raw_stream = create_raw_from_data_streams(eeg_stream, markers_stream)
        if(multiprocessing_ ): 
            conn3.send("next block")
        
    if(multiprocessing_): 
        conn1.recv()
        print("Recieved")
        conn1.send(raw_stream)
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
    conn1, conn2 = Pipe(duplex=True)
    conn3, conn4 = Pipe(duplex=True)
    multiprocessing_ = True
    
    if(multiprocessing_ == False):
        raw =read_from_lsl(False, conn1)
        raw.plot(scalings=dict(eeg=100e-6))
    else:
    
        width,height = paradigm.get_screen_param()
        training_set, targets = paradigm.create_online_set()
        
        read_from_lsl_process = Process(target=read_from_lsl, args=(True, conn1, conn3))
        read_from_lsl_process.start()
        
        outlet = paradigm.set_outlet()
        conn2.send("start reading streams")
        
        paradigm.present_paradigm(training_set, targets, width, height, outlet, conn4)
        
        conn2.send("stop reading streams")
        raw_stream = conn2.recv()
        raw_stream.plot(scalings=dict(eeg=100e-6))
        # preprocessing
        raw_filtered = filtering(raw_stream, lfreq=0.5, hfreq=40, notch_dist=10, notch_qf=25, ica_exclude=[0, 1])
        raw_clean = remove_bad_channels(raw_filtered, interpolate=True)
        epochs_data = epochs_segmentation_online(raw_clean)
        # Feature extraction
        gf_x, off_x, on_x = data4eegnet(epochs_data)
        # Predict
        predict = EEGNET_predict_target(on_x, off_x)
        print("==========================================\n"
              f"      Detect {predict} as target\n"
              f"             {predict}!\n"
              "==========================================")

        read_from_lsl_process.terminate()
        read_from_lsl_process.join()
