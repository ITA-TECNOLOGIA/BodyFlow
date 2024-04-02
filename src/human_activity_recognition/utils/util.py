# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnologico de Aragon (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@itainnova.es
#          Angel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Maranes Nueno                            cmaranes@itainnova.es
#          Irene Lopez Bosque                                ilopez@itainnova.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibanez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocio Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import logging
from mlflow import MlflowClient
import os
import mlflow

def get_df(args):
    logging.info("Loading dataset...") 

    # Import csv file 
    df = pd.read_csv(args.path_dataset)#, nrows = 20000)

    # Columns names to lowercase
    df.columns = df.columns.str.lower()

    # Drop unnecesary info (columns containing "name", "type" and "hallucinated")
    df = df[df.columns.drop(df.filter(regex='.name|.type|.hallucinated'))]

    # Fill missing values with zeros
    df = df.fillna(0)

    if args.dataset == 'harup':
        info_range = range(0, 5)
        imus_range = range(5, 35)
        ankle_range = range(5, 11)
        pose_c1_3d_range = range(35, 146)
        pose_c1_2d_range = range(146, 220)
        pose_c2_3d_range = range(220, 331)
        pose_c2_2d_range = range(331, 405)

    elif args.dataset == 'itainnova':
        # ITAINNOVA_DATASET
        info_range = range(0, 4)
        imus_range = range(4, 82)
        # instep_left_range = range(34, 40)
        # instep_right_range = range(40, 46)
        ankle_range = range(34, 46) # both insteps

        # Drop bonus activity from the dataset
        df = df[df.activity != 12]
    
    elif args.dataset == 'vision':
        landmarks = ["hips", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "spine", 
        "chest", "jaw", "nose", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]

        df_pose = df.drop(df.columns[~df.columns.str.contains('|'.join(landmarks))], axis=1)
        df = pd.concat([ df[['timestamp']], df_pose], axis=1)

    if args.dataset == 'harup' or args.dataset == 'itainnova':
        # Get number of classes
        num_classes = len(df.activity.unique())
        logging.info(f"Total number of classes: %s", len(df.activity.unique()))
        logging.info(f"Classes IDs: %s", df.activity.unique())
    else:
        num_classes = 0


    if args.features[0] == "2d" or args.features[0] == "all":
        logging.info("Applying min_max_scaler to camera 1 and camera 2, 2D pose (in pixels)")  
        min_max_scaler = MinMaxScaler()
        min_max_scaler = MinMaxScaler()
        df.iloc[:, np.r_[pose_c1_2d_range, pose_c2_2d_range]] = min_max_scaler.fit_transform(df.iloc[:, np.r_[pose_c1_2d_range, pose_c2_2d_range]])


    # Depending on the type of feature you want to train the model, select the appropiate columns
    if args.features[0] == "all":
        df = df
    elif args.features[0] == "imus":
        df = df.iloc[:, np.r_[info_range, imus_range]]
    elif args.features[0] == "2d":
        df = df.iloc[:, np.r_[info_range, pose_c1_2d_range, pose_c2_2d_range]]
    elif args.features[0] == "3d":
        df = df.iloc[:, np.r_[info_range, pose_c1_3d_range, pose_c2_3d_range]]
    elif args.features[0] == "ankle":
        df = df.iloc[:, np.r_[info_range, ankle_range]]


    # The 37 body landmarks are reduced to the 17 keypoints commonly used for HAR
    if args.features[0] == "2d" or args.features[0] == "3d" or args.features[0] == "all":
        landmarks = ["hips", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "spine", 
        "chest", "jaw", "nose", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
        
        df_pose = df.drop(df.columns[~df.columns.str.contains('|'.join(landmarks))], axis=1)
        df_info  = df.iloc[:, np.r_[info_range]]
        df = pd.concat([df_info, df_pose], axis=1)

        if args.features[0] == "all":
            df_imus  = df.iloc[:, np.r_[imus_range]]
            df = pd.concat([df_info, df_imus, df_pose], axis=1)

    # Divide the resulting dataframe in two, one for training and the other one for testing/validating 
    df_train = df[(df["trial"] == 1) | (df["trial"] == 2)]
    df_test = df[df["trial"] == 3]

    return num_classes, df_train, df_test



def slide_window_over_dataset(X_list, y_list, sub_df_features, sub_df_label, window_size_past, window_size_future, step):
    # At the beginning and at the end, add the first and last value n times
    sub_df_features = pd.concat([pd.concat([sub_df_features.head(1)]*window_size_past), sub_df_features , pd.concat([sub_df_features.tail(1)]*window_size_future)])
    sub_df_label = pd.concat([pd.concat([sub_df_label.head(1)]*window_size_past), sub_df_label , pd.concat([sub_df_label.tail(1)]*window_size_future)])


    for i in range(window_size_past, len(sub_df_features) - window_size_future, step): 
        # "1" takes into account the current sample
        x = sub_df_features.iloc[(i - window_size_past):(i + 1 + window_size_future)].values.tolist()
        labels = sub_df_label.iloc[(i - window_size_past): i + window_size_future]

        X_list.append(np.array(x))
        y_list.append(np.array(labels.values[0]))
    return


def get_X_y_datasets(df_train, df_test, args):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Transform the original dataframe in "subdataframes" grouped by user, trial and activity
    df_train_grouped = df_train.groupby(["subject", "activity", "trial"])
    df_test_grouped = df_test.groupby(["subject", "activity", "trial"])
    logging.info("Sliding window over dataset...")  
 

    df_grouped_list = [df_train_grouped, df_test_grouped]

    for i, df_grouped in enumerate(df_grouped_list):
        # Iterate over each group of the train dataframe
        for info , df_group in df_grouped:
      
            if args.dataset == "itainnova":
                df_group_X = df_group.drop(["trial", "subject", "activity", "step"], axis=1)
            else:
                df_group_X = df_group.drop(["Tag", "Trial", "Subject", "Activity", "TimeStamps"], axis=1)
            df_group_y = df_group[args.label]

            if i == 0:
                slide_window_over_dataset(X_train, y_train, df_group_X, df_group_y, args.window_size_past, args.window_size_future, args.window_step)
            elif i == 1:
                slide_window_over_dataset(X_test, y_test, df_group_X, df_group_y, args.window_size_past, args.window_size_future, args.window_step)
  
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    # Encode labels -> integer [1, 2, 3 ... ] to [1, 0, 0], [0, 1, 0], [0, 0, 1] 
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    return X_train, y_train, X_test, y_test




def plot_confusion_matrix(y_true, y_pred, args):
    if args.dataset == 'harup':
        classes = [ 'Fhands', 
                    'Fknees', 
                    'Fback', 
                    'Fside', 
                    'Fchair', 
                    'Walk', 
                    'Stand', 
                    'Sit', 
                    'Object', 
                    'Jump', 
                    'Lay']
    elif args.dataset == "itainnova":
        # The current version of dataset Itainnova does not have "laying" activity
        classes = [ 'walk',
                'jog',
                'steps',
                'sit',
                'stand',
                'jump',
                'object',
                'Ffor',
                'Fback',
                'Fside']
    
    cf_matrix_norm = confusion_matrix(y_true, y_pred, normalize='true')
    df_cm_norm = pd.DataFrame(cf_matrix_norm, index = [i for i in classes], columns = [i for i in classes])

    fig1 = plt.figure(figsize = (8,8))
    ax = sns.heatmap(df_cm_norm, fmt = '.2f', annot = True, annot_kws = {'fontsize': 8})
    ax.set(xlabel = 'Predicted Label', ylabel = 'True Label')
    matrix_title =  f'Modelo {args.har_model}, features: {args.features[0]}'
    ax.set(title = matrix_title)

    plt.xticks(fontsize = 8) 
    plt.yticks(fontsize = 8) 
    temp_name = "confusion-matrix.png"
    plt.savefig(temp_name)
    plt.close(fig1)

    # Log the confusion matrix to mlflow
    mlflow.log_artifact(temp_name, "confusion-matrix-plots")

    # Delete confusion matrix .png as it is already saved in the 
    try:
        os.remove(temp_name)
    except FileNotFoundError:
        logging.warning(f"{temp_name} file is not found")

    return

def auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]

    logging.info("run_id: {}".format(r.info.run_id))
    logging.info("artifacts: {}".format(artifacts))
    logging.info("params: {}".format(r.data.params))
    logging.info("metrics: {}".format(r.data.metrics))
    logging.info("tags: {}".format(tags))

    return


