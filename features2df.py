import pandas as pd
import numpy as np
import os

def features2df():
    DATA_FOLDER = os.path.join(os.getcwd(), "segmented_data")
    OTHER_FEATUERS_FOLDER_TSFRESH = os.path.join(DATA_FOLDER, "other", "features", "TSFRESH")
    TARGET_FEATUERS_FOLDER_TSFRESH = os.path.join(DATA_FOLDER, "target", "features", "TSFRESH")
    other_list = os.listdir(OTHER_FEATUERS_FOLDER_TSFRESH)
    target_list = os.listdir(TARGET_FEATUERS_FOLDER_TSFRESH)
    X_list = []
    for i in other_list:
        cur_file = pd.read_csv(os.path.join(OTHER_FEATUERS_FOLDER_TSFRESH, i))
        X_list.append(cur_file)
    X = pd.concat(X_list)
    other_labels = np.zeros(len(other_list))
    Y = pd.DataFrame(other_labels.tolist())

    X_list = []
    for i in target_list:
        cur_file = pd.read_csv(os.path.join(OTHER_FEATUERS_FOLDER_TSFRESH, i))
        X_list.append(cur_file)
    X_list.append(X)
    X = pd.concat(X_list)
    target_labels = np.ones(len(target_list))
    df_target_labels = pd.DataFrame(target_labels)
    Y_list = [df_target_labels, Y]
    Y = pd.concat(Y_list)
    return X, Y


if __name__ == '__main__':
    X, Y = features2df()
    path = os.path.join(os.getcwd(), "segmented_data")
    X.to_csv(os.path.join(path, 'X.csv'))
    Y.to_csv(os.path.join(path, 'Y.csv'))
