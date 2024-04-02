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
import logging
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class LogsReader():
    def __init__(self, log_filename, person_id, input_data, window_size, window_step):
        """
        This class loads UPFall data according to the input type and returns the 
        number of classes,  input_features and pre-processed data 
        """
        self.log_filename = log_filename
        self.person_id = person_id
        self.features = input_data
        self.window_size = window_size
        self.window_step = window_step
        
        

    def fetch_data(self):
        self.load_data_from_csv()
        self.data_labels()
        self.normalization()
        self.featureSelection()
        df_predict = self.df
        X_pred = self.get_predict(df_predict)

        return X_pred, self.har_in_features


    def load_data_from_csv(self):
        df = pd.read_csv(os.path.join('logs', self.log_filename))
        df = df[df['id'] == self.person_id]
        df.columns = df.columns.str.lower()
        df = df[df.columns.drop(df.filter(regex = '.name|.type|.hallucinated'))]
        self.df = df.fillna(0)
        return
        
    
    def data_labels(self):
        """"
        Logger data from HPE dicts for selection the appropiate data
        """
        self.har_dict_csv = {'timestamp':      range(0, 1),
                            'id':              range(1, 2),
                            'pose_c1_3d':    range(2, 113),
                            'pose_c1_2d':   range(113, 191)
                            }

        self.landmarks = ["hips", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "spine", 
                          "chest", "jaw", "nose", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow",
                          "right_wrist"]
        
    
    def normalization(self):
        logging.info("Applying min_max_scaler to camera 1 and camera 2, 2D pose (in pixels)")  
        min_max_scaler = MinMaxScaler()
        self.df.iloc[:, np.r_[self.har_dict_csv['pose_c1_2d']]] = min_max_scaler.fit_transform(self.df.iloc[:, np.r_[self.har_dict_csv['pose_c1_2d']]])

    def featureSelection(self):
        if self.features == '2d':
            self.df = self.df.iloc[:, np.r_[self.har_dict_csv['pose_c1_2d']]]
        elif self.features == '3d':
            self.df = self.df.iloc[:, np.r_[self.har_dict_csv['pose_c1_3d']]]

        if self.features == '2d' or self.features == '3d':
            self.df = self.df.drop(self.df.columns[~self.df.columns.str.contains('|'.join(self.landmarks))], axis=1)
    
        self.har_in_features = len(self.df.columns)



    def slide_window_over_dataset(self, X_predict, sub_df_features, window_pad, step):
        # At the beginning and at the end, add the first and last value n times
        sub_df_features = pd.concat([pd.concat([sub_df_features.head(1)]*window_pad), sub_df_features, pd.concat([sub_df_features.tail(1)]*window_pad)])
     
        for i in range(window_pad, len(sub_df_features) - window_pad, step): 
            # "1" takes into account the current sample
            x = sub_df_features.iloc[(i - window_pad):(i + 1 + window_pad)].values.tolist()
            X_predict.append(np.array(x))
        return


    def get_predict(self, df_predict):
        window_pad = int((self.window_size-1)/2)
        logging.info("Sliding window over data...")  
        X_predict = []
        self.slide_window_over_dataset(X_predict, df_predict, window_pad, self.window_step)   
        X_predict = np.array(X_predict)
        
        C_X_predict =  np.concatenate((X_predict, X_predict), axis=2)
        # Temporal fix for bug
        
        return C_X_predict
    