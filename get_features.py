# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:39:22 2022

@author: marko
"""
import os
import pandas as pd
import tsfel
import glob
import matplotlib.pyplot as plt
import numpy as np
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

DATA_FOLDER = os.path.join(os.getcwd(), "segmented_data")
OTHER_DATA_FOLDER = os.path.join(DATA_FOLDER, "other\\data")
TARGET_DATA_FOLDER = os.path.join(DATA_FOLDER, "target\\data")
OTHER_FEATUERS_FOLDER = os.path.join(DATA_FOLDER, "other\\features")
TARGET_FEATUERS_FOLDER = os.path.join(DATA_FOLDER, "target\\features")
FSFEL_OTHER_FEATUERS_FOLDER = os.path.join(OTHER_FEATUERS_FOLDER, "FSFEL")
FSFEL_TARGET_FEATUERS_FOLDER = os.path.join(TARGET_FEATUERS_FOLDER, "FSFEL")
TSFRESH_OTHER_FEATUERS_FOLDER = os.path.join(OTHER_FEATUERS_FOLDER, "TSFRESH")
TSFRESH_TARGET_FEATUERS_FOLDER = os.path.join(TARGET_FEATUERS_FOLDER, "TSFRESH")

PLOTS_FOLDER = os.path.join(DATA_FOLDER, "plots")
FSFEL_PLOTS_FOLDER = os.path.join(PLOTS_FOLDER, "FSFEL")
TSFRESH_PLOTS_FOLDER = os.path.join(PLOTS_FOLDER, "TSFRESH")

FILE_NAME = "EEG_05_12_2_0.csv"
DATA_FILE = os.path.join(TARGET_DATA_FOLDER, FILE_NAME)
FEATURES_FILE = os.path.join(TARGET_FEATUERS_FOLDER, FILE_NAME)
TSFEL_FEATURES_FOLDER = os.path.join(DATA_FOLDER, "plots\\FSFEL")

def plot_features(x_range, y_t, y_o, title, color_t, color_o, folder, x_lable):
        plt.plot(x_range, y_t)
        plt.plot(x_range, y_o, color = color_o, alpha=0.4)
        try:
            plt.ylim(min(y_t.min(),y_o.min()) , max(y_t.max(),y_o.max()))
        except:
            print("limit except", title)
            return 
        if(title.find('.')!=-1)            :
            title = title.replace(".", "_")           

        plt.title(title)
        plt.xlabel(x_lable) 
        plt.legend()
        try:
            plt.savefig(os.path.join(folder, title))
        except:
            print("failed to plot", title)            
        plt.show()
        
def TSFEL_features(data_file, features_file):
    segment = pd.read_csv(data_file)
    cfg_file = tsfel.get_features_by_domain()# If no argument is passed retrieves all available features
    X_train = tsfel.time_series_features_extractor(cfg_file, segment, window_size=20,  window_spliter=True)
    X_train.to_csv(features_file)
    return X_train    

def TSFRESH_features(data_file, features_file):
    print("TSFRESH_features")
    segment = pd.read_csv(data_file)
    segment.insert(0, 'id', '1')

    extracted_features = extract_features(segment, column_id="id", column_sort="ts")
    extracted_features = impute(extracted_features)
    df_filt = extracted_features.columns[(extracted_features == 0).all()]
    extracted_features = extracted_features.drop(df_filt, axis=1)
    extracted_features.to_csv(features_file)
    print("TSFRESH_features done")

    return extracted_features

def extract_all_features():
    TSFEL_other_feat_list = []
    TSFEL_target_feat_list = []
    TSFRESH_other_feat_list = []
    TSFRESH_target_feat_list = []
    
    target_list = glob.glob(os.path.join(TARGET_DATA_FOLDER,"*.csv"))
    for target_data in target_list:
        TSFEL_feat = TSFEL_features(target_data, target_data.replace("\\data\\", "\\features\\FSFEL\\"))
        TSFEL_target_feat_list.append(TSFEL_feat)
        
        TSFRESH_feat = TSFRESH_features(target_data, target_data.replace("\\data\\", "\\features\\TSFRESH\\"))
        TSFRESH_feat_all_chann = combine_channels_data(TSFRESH_feat)
        TSFRESH_target_feat_list.append(TSFRESH_feat_all_chann)
    other_list = glob.glob(os.path.join(OTHER_DATA_FOLDER,"*.csv"))
    for other_data in other_list:
        TSFEL_feat = TSFEL_features(other_data, other_data.replace("\\data\\", "\\features\\FSFEL\\"))
        TSFEL_other_feat_list.append(TSFEL_feat)

        TSFRESH_feat = TSFRESH_features(other_data, other_data.replace("\\data\\", "\\features\\TSFRESH\\"))
        TSFRESH_feat_all_chann = combine_channels_data(TSFRESH_feat)
        TSFRESH_other_feat_list.append(TSFRESH_feat_all_chann)
        
    TSFRESH_x = [0,1,2,3,4,5,6,7,8]
    print(TSFRESH_other_feat_list)
    vesualize_features(TSFRESH_target_feat_list,TSFRESH_other_feat_list, TSFRESH_x, TSFRESH_PLOTS_FOLDER, 'channel #')
    
    TSFEL_x = [1,2,3,4]
    vesualize_features(TSFEL_target_feat_list,TSFEL_other_feat_list, TSFEL_x, FSFEL_PLOTS_FOLDER, "window #")

def combine_channels_data(split_channels_features):
    print("combine_channels_data")
    number_of_channels = 9
    merged_channels_features = pd.DataFrame()
    for i in range(number_of_channels):   
        prefix_rm = str(i)+"__"
        i_ch_features = split_channels_features.loc[:, split_channels_features.columns.str.startswith(prefix_rm)]
        i_ch_features.columns =  i_ch_features.columns.str.lstrip(prefix_rm)
        merged_channels_features = pd.concat([merged_channels_features, i_ch_features])
    return merged_channels_features

def vesualize_features(target_feat_list, other_feat_list, x, graphs_folder, x_lable):
    if(len(target_feat_list)==0):
        print("No target data features")
    else:
        list_of_features = list(target_feat_list[0].columns.values)
        target_features = {}
        for target_feat in target_feat_list:
            for feature in list_of_features:
                try:
                    temp = target_feat[feature]
                except:
                    continue 
                try:
                    feature_values = target_features[feature]
                except:
                    target_features[feature] = np.array(target_feat[feature].tolist())
                else:
                    target_features[feature] = np.vstack((target_features[feature],np.array(target_feat[feature].tolist())))
        other_features = {}
        for other_feat in other_feat_list:
            for feature in list_of_features:
                try:
                    temp = other_feat[feature]
                except:
                    continue 
                try:
                    feature_values = other_features[feature]
                except:
                    other_features[feature] = np.array(other_feat[feature].tolist())
                else:
                    other_features[feature] = np.vstack((other_features[feature],np.array(other_feat[feature].tolist())))
        
        for feature in list_of_features:
            try:
                y_t = np.transpose(target_features[feature])
            except:
                print("except on target")
            else:
                try:
                    y_o = np.transpose(other_features[feature])
                except:
                    print("except on other")
                else:
                    
                    if(y_t[0][0] != y_t[1][0] or y_t[1][0]!=y_t[2][0]):
                        plot_features(x, y_t, y_o, feature, 'g', 'grey', graphs_folder, x_lable)

if __name__ == '__main__':
    extract_all_features()
 