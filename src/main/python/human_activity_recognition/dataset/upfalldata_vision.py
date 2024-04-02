# --------------------------------------------------------------------------------
# BodyFlow
# Version: 2.0
# Copyright (c) 2024 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: March 2024
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@ita.es
#          Angel Gimeno Valero                              agimeno@ita.es
#          Carlos Maranes Nueno                            cmaranes@ita.es
#          Irene Lopez Bosque                                ilopez@ita.es
#          Jose Ignacio Calvo Callejo                       jicalvo@ita.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@ita.es
#          Pilar Salvo Ibanez                                psalvo@ita.es
#          Rafael del Hoyo Alonso                          rdelhoyo@ita.es
#          Rocio Aznar Gimeno                                raznar@ita.es
#          Pablo Perez Lazaro                               plazaro@ita.es
#          Marcos Marina Castello                           mmarina@ita.es
# All rights reserved 
# --------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns
import logging

import numpy as np
from sklearn.preprocessing import OneHotEncoder

"""
NOTE: IT ONLY CAN BE TRAINED WITH 1 CAMERA. IDEALLY WE SHOULD HAVE TWO SAMPLES (CAMERA 1 AND CAMERA 2) PER USER
"""

class UPFalldataVision():
    def __init__(self, args):
        """
        This class loads UPFall data according to the input type and returns the 
        number of classes,  input_features and pre-processed data 
        """
        self.features = args.input_data
        self.path_dataset = args.path_dataset 
        self.dataset = args.dataset
        self.label = args.label
        self.window_size = args.window_size
        self.window_step = args.window_step
        self.train = args.train
        self.test = args.test
        self.camera = 'c2'
        

    def fetch_data(self):
        self.load_data_from_csv()
        self.data_labels()
        self.normalization()
        self.featureSelection()
        df_train, df_test = self.dataSplit()
        if self.train == True:
            X_train, y_train, X_test, y_test =  self.get_train_test(df_train, df_test)
        elif self.train == False:
            X_train = np.empty
            y_train = np.empty
            X_test, y_test =  self.get_test(df_test)

        return X_train, y_train, X_test, y_test, df_test, self.har_in_features, self.num_classes, self.classes


    def load_data_from_csv(self):
        #df = pd.read_csv(self.path_dataset, low_memory=False, nrows=1000)
        df = pd.read_csv(self.path_dataset, low_memory=False)
        df.columns = df.columns.str.lower()
        df = df[df.columns.drop(df.filter(regex = '.name|.type|.hallucinated'))]
        self.df = df.fillna(0)
        self.num_classes = len(df.activity.unique())
        logging.info(f"Total number of classes: %s" %len(df.activity.unique()))
        logging.info(f"Classes IDs: %s" %df.activity.unique())
        
    def data_labels(self):
        """"
        UP-Fall dataset dicts for selection the appropiate data
        """
        self.har_dict_csv = {'labels':      range(0, 5),
                            'pose_c1_3d':   range(35, 146),
                            'pose_c1_2d':   range(146, 220),
                            'pose_c2_3d':   range(220, 331),
                            'pose_c2_2d':   range(331, 405),
                            'all':          range(35, 405),
                            }
        self.har_dict_csv['all'] = list(self.har_dict_csv[f'pose_{self.camera}_2d']) + list(self.har_dict_csv[f'pose_{self.camera}_3d'])

        self.landmarks = ["hips", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "spine", 
                          "chest", "jaw", "nose", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow",
                          "right_wrist"]
        
        self.info_columns = ["tag", "trial", "subject", "activity", "timestamps"]
        
        self.data_groups = ["subject", "activity", "trial"]

        self.classes = ['Fhands', 
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

    def normalization(self):
        logging.info("Applying min_max_scaler to camera 1 and camera 2, 2D pose (in pixels)")  
        min_max_scaler = MinMaxScaler()
        self.df.iloc[:, np.r_[self.har_dict_csv[f'pose_{self.camera}_2d']]] = min_max_scaler.fit_transform(self.df.iloc[:, np.r_[self.har_dict_csv[f'pose_{self.camera}_2d']]])
        self.df.iloc[:, np.r_[self.har_dict_csv[f'pose_{self.camera}_3d']]] = min_max_scaler.fit_transform(self.df.iloc[:, np.r_[self.har_dict_csv[f'pose_{self.camera}_3d']]])

    def featureSelection(self):
        df_info = self.df.iloc[:, np.r_[self.har_dict_csv['labels']]]
        if self.features == 'all':
            df_f = self.df.iloc[:, np.r_[self.har_dict_csv[self.features]]]     
        elif self.features == '2d':
            df_f = self.df.iloc[:, np.r_[self.har_dict_csv[f'pose_{self.camera}_2d']]]
        elif self.features == '3d':
            df_f = self.df.iloc[:, np.r_[self.har_dict_csv[f'pose_{self.camera}_3d']]]
        self.df = pd.concat([df_info, df_f], axis = 1)

        if self.features == '2d' or self.features == '3d' or self.features == 'all':
            df_pose = self.df.drop(self.df.columns[~self.df.columns.str.contains('|'.join(self.landmarks))], axis=1)
            df_info = self.df.iloc[:, np.r_[self.har_dict_csv['labels']]]
            
            self.df = pd.concat([df_info, df_pose], axis=1)
            if self.features == 'all':
                self.df = pd.concat([df_info, df_pose], axis=1)

        self.har_in_features = len(self.df.columns) - len(df_info.columns)

    def dataSplit(self):
        # Divide the resulting dataframe in two, one for training and the other one for testing/validating 
        df_train = self.df[(self.df['trial'] == 1) | (self.df['trial'] == 2)]
        df_test = self.df[self.df['trial'] == 3]
        return df_train, df_test
        

    def slide_window_over_dataset(self, X_list, y_list, sub_df_features, sub_df_label, window_pad, step):
        # At the beginning and at the end, add the first and last value n times
        sub_df_features = pd.concat([pd.concat([sub_df_features.head(1)]*window_pad), sub_df_features , pd.concat([sub_df_features.tail(1)]*window_pad)])
        sub_df_label = pd.concat([pd.concat([sub_df_label.head(1)]*window_pad), sub_df_label , pd.concat([sub_df_label.tail(1)]*window_pad)])

        for i in range(window_pad, len(sub_df_features) - window_pad, step): 
            # "1" takes into account the current sample
            x = sub_df_features.iloc[(i - window_pad):(i + 1 + window_pad)].values.tolist()
            labels = sub_df_label.iloc[(i - window_pad): i + window_pad]

            X_list.append(np.array(x))
            y_list.append(np.array(labels.values[0]))
        return


    def get_train_test(self, df_train, df_test):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        window_pad = int((self.window_size-1)/2)

        # Transform the original dataframe in "subdataframes" grouped by user, trial and activity
        df_train_grouped = df_train.groupby(self.data_groups)
        df_test_grouped = df_test.groupby(self.data_groups)
        logging.info("Sliding window over dataset...")  
    
        df_grouped_list = [df_train_grouped, df_test_grouped]

        for i, df_grouped in enumerate(df_grouped_list):
            # Iterate over each group of the train dataframe
            for info , df_group in df_grouped:
                df_group_X = df_group.drop(self.info_columns, axis=1)
                df_group_y = df_group[self.label]

                if i == 0:
                    self.slide_window_over_dataset(X_train, y_train, df_group_X, df_group_y, window_pad, self.window_step)
                elif i == 1:
                    self.slide_window_over_dataset(X_test, y_test, df_group_X, df_group_y, window_pad, self.window_step)
    
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

        # Encode labels -> integer [1, 2, 3 ... ] to [1, 0, 0], [0, 1, 0], [0, 0, 1] 
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(y_train)
        y_train = enc.transform(y_train)
        y_test = enc.transform(y_test)
        return X_train, y_train, X_test, y_test
    

    def get_test(self, df_test):
        X_test = []
        y_test = []

        window_pad = int((self.window_size-1)/2)

        # Transform the original dataframe in "subdataframes" grouped by user, trial and activity
        df_t_grouped = df_test.groupby(self.data_groups)
        logging.info("Sliding window over dataset...")  
    
        for info , df_group in df_t_grouped:
            if self.dataset == "itainnova":
                df_group_X = df_group.drop(self.info_columns, axis=1)
            else:
                df_group_X = df_group.drop(self.info_columns, axis=1)
            df_group_y = df_group[self.label]
            self.slide_window_over_dataset(X_test, y_test, df_group_X, df_group_y, window_pad, self.window_step)
    
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        y_test = y_test.reshape(y_test.shape[0], 1)

        # Encode labels -> integer [1, 2, 3 ... ] to [1, 0, 0], [0, 1, 0], [0, 0, 1] 
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(y_test)
        y_test = enc.transform(y_test)

        return X_test, y_test


