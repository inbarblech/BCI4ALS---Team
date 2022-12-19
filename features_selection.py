# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:55:13 2022

@author: marko
"""
import pandas as pd
import glob
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import array 



DATA_FOLDER = os.path.join(os.getcwd(), "segmented_data")
OTHER_FEATUERS_FOLDER = os.path.join(DATA_FOLDER, "other\\features")
TARGET_FEATUERS_FOLDER = os.path.join(DATA_FOLDER, "target\\features")
FSFEL_OTHER_FEATUERS_FOLDER = os.path.join(OTHER_FEATUERS_FOLDER, "FSFEL")
FSFEL_TARGET_FEATUERS_FOLDER = os.path.join(TARGET_FEATUERS_FOLDER, "FSFEL")
TSFRESH_OTHER_FEATUERS_FOLDER = os.path.join(OTHER_FEATUERS_FOLDER, "TSFRESH")
TSFRESH_TARGET_FEATUERS_FOLDER = os.path.join(TARGET_FEATUERS_FOLDER, "TSFRESH")
TRAINING_FOLDER = os.path.join(os.getcwd(), "training_data")
TSFRESH_TRAINING_FOLDER = os.path.join(TRAINING_FOLDER, "TSFRESH")

def get_all_features(folder):
    files_list = glob.glob(os.path.join(folder,"*.csv"))
    all_feat = pd.DataFrame()
    for f in files_list:
        temp_df = [all_feat, pd.read_csv(f)]
        all_feat = pd.concat(temp_df, ignore_index=True)
    return all_feat
        

def load_TSFRESH_features(target_folder = TSFRESH_TARGET_FEATUERS_FOLDER, other_folder = TSFRESH_OTHER_FEATUERS_FOLDER):
    #load all features for target    
    all_targ_features = get_all_features(target_folder)
    len_t = all_targ_features.shape[0]
    all_other_features = get_all_features(other_folder)
    len_o = all_other_features.shape[0]
    all_features = pd.concat([all_targ_features, all_other_features], ignore_index=True)
    lables = [1]*len_t +  [0]*len_o
    return all_features, lables

def select_TSFRESH_features():
    all_features, lables = load_TSFRESH_features()
    #selection per channel
    for i in range(1, 9, 1):
        #filna and abs due to sklearn limitation 
        chan_feat = all_features.loc[:, all_features.columns.str.startswith(str(i) + '__')].fillna(0).abs()
        print("i: ", i, "number of feat: ", chan_feat.shape)
        select = SelectKBest(score_func=chi2, k=50)
        z = select.fit_transform(chan_feat, lables)
        filt = select.get_support()
        all_chan_feat = array(chan_feat.columns)
        selected_chan_feat_names = all_chan_feat[filt]
        selected_chan_feat = chan_feat[selected_chan_feat_names]
        selected_chan_feat.columns = selected_chan_feat.columns.str.lstrip(str(i) + '__')   
        selected_chan_feat["Target"] = lables
        file_name = "Chan_" + str(i) + ".csv"
        selected_chan_feat.to_csv(os.path.join(TSFRESH_TRAINING_FOLDER, file_name))


if __name__ == '__main__':
    select_TSFRESH_features()