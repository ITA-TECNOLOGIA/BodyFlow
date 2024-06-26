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

import os
import zipfile
import glob
import cv2
import pandas as pd
import pandas.io.common
import numpy as np
import logging
import sys

sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation'))
sys.path.append(os.path.join('src', 'main', 'python'))

from human_pose_estimation.inference_server import inference
from human_pose_estimation.video_capture.VideoFromImages import VideoFromImages
from human_pose_estimation.models.predictors_2d.CPN import CPN
from human_pose_estimation.models.predictors_3d.MHFormer import MHFormer ## from models.predictors_3d.videopose import videopose
from human_pose_estimation.models.HPE import HPE
from human_pose_estimation.common_pose.PoseLogger import PoseLogger
from human_pose_estimation.common_pose.visualization_from_images import Visualization

from human_pose_estimation.models.person_detector.YoloV3 import YoloV3
from human_pose_estimation.models.tracking.Sort import Sort
from human_pose_estimation.models.tracking.Single import Single

from human_pose_estimation.models.predictors_3d.ExPose import ExPose  

net2d = 'cpn'
net3d = 'expose'
persons = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8', 
            'Subject9', 'Subject10', 'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17']
actions = ['Activity1', 'Activity2', 'Activity3', 'Activity4', 'Activity5', 'Activity6', 'Activity7', 'Activity8', 'Activity9', 
           'Activity10', 'Activity11']
trials = ['Trial1', 'Trial2', 'Trial3']
cameras = ['Camera1', 'Camera2']



def extractZipFiles():
    # Extract all zip files
    for person in persons:
        for action in actions:
            for trial in trials:
                for camera in cameras:
                    try:
                        read_zip = zipfile.ZipFile(os.path.join(dataset_folder, person, action, trial, f'{person}{action}{trial}{camera}.zip'), 'r')
                        # uncompress_size = sum((file.file_size for file in read_zip.infolist()))
                        extracted_size = 0
                        if os.path.isdir(os.path.join(dataset_folder, person, action, trial, f'{person}{action}{trial}{camera}')) == False:
                            for file in read_zip.infolist():
                                extracted_size += file.file_size
                                #print ("%.02f %%" % (extracted_size * 100/uncompress_size))
                                read_zip.extract(file, os.path.join(dataset_folder, person, action, trial, f'{person}{action}{trial}{camera}'))
                    except FileNotFoundError:
                        logging.info(f'User warning... Passing: {person}{action} {trial} {camera}')
                        pass
                    logging.info( f'Uncompressed: {person} {action} {trial} {camera} ')
    return

def inferOnData():
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    for person in persons:
        for action in actions:
            for trial in trials:
                for camera in cameras:
                    try: 
                        video_file = f'{dataset_folder}/{person}/{action}/{trial}/{person}{action}{trial}{camera}'
                        prospect = f'Log_{net2d}_{net3d}_{video_file.split("/")[-1]}.csv'
                        log_list = os.listdir(logs_dir)
                        filename = 'processing_list.txt'
                        writeOnFile(video_file, filename)
                        if prospect not in log_list:
                            images_path = glob.glob(os.path.join(video_file, "*.png"))
                            images_path.sort(key=os.path.getctime)
                            flag = True
                            for img_path in images_path:
                                try:
                                    cv2.imread(img_path)
                                except:
                                    flag = False
                                    pass
                            if flag == True:
                                logging.info(f'Processing {prospect}')
                                videoFromImages = VideoFromImages(video_file, False)
                                predictor_2d = CPN()
                                predictor_3d = ExPose(window_length = 1, video_filename=video_file.split('/')[-1].split('.')[0]) #MHFormer(window_length=81) #
                                person_detector = person_detector = YoloV3()
                                tracking = Single() #Sort(max_age=10)
                                human_pose_estimator = HPE(predictor_2d, predictor_3d, window_length=1, person_detector=person_detector, tracking=tracking)
                                pose_logger = PoseLogger(-1, filename=prospect)
                                inference(videoFromImages, human_pose_estimator, pose_logger)
                                #Visualization(video_file, prospect, person_id = 1)
                            elif flag == False:
                                pass
                            pass
                        else:
                            logging.info(f'Passing {prospect} (all ready on folder)')
                    except IndexError:
                        pass
                    except FileNotFoundError:
                        pass
    return


# Ignore warning about DataFrame overwriting
pd.options.mode.chained_assignment = None

def syncData():
    # Get IMU data
    imu_data = pd.read_csv(os.path.join(dataset_folder,'CompleteDataSet.csv'),  skiprows=[1],
                usecols = ['TimeStamps', 
                'AnkleAccelerometer', 'Unnamed: 2', 'Unnamed: 3',
                'AnkleAngularVelocity', 'Unnamed: 5', 'Unnamed: 6',
                'RightPocketAccelerometer',  'Unnamed: 9', 'Unnamed: 10',
                'RightPocketAngularVelocity', 'Unnamed: 12', 'Unnamed: 13',
                'BeltAccelerometer', 'Unnamed: 16', 'Unnamed: 17',
                'BeltAngularVelocity', 'Unnamed: 19', 'Unnamed: 20',
                'NeckAccelerometer', 'Unnamed: 23', 'Unnamed: 24',
                'NeckAngularVelocity', 'Unnamed: 26', 'Unnamed: 27',
                'WristAccelerometer', 'Unnamed: 30', 'Unnamed: 31',
                'WristAngularVelocity', 'Unnamed: 33', 'Unnamed: 34',
                'Subject', 'Activity','Trial', 'Tag'] )

    # Reoder data
    imu_data = imu_data[ ['TimeStamps',  'Subject', 'Activity','Trial', 'Tag',
                'AnkleAccelerometer', 'Unnamed: 2', 'Unnamed: 3',
                'AnkleAngularVelocity', 'Unnamed: 5', 'Unnamed: 6',
                'RightPocketAccelerometer',  'Unnamed: 9', 'Unnamed: 10',
                'RightPocketAngularVelocity', 'Unnamed: 12', 'Unnamed: 13',
                'BeltAccelerometer', 'Unnamed: 16', 'Unnamed: 17',
                'BeltAngularVelocity', 'Unnamed: 19', 'Unnamed: 20',
                'NeckAccelerometer', 'Unnamed: 23', 'Unnamed: 24',
                'NeckAngularVelocity', 'Unnamed: 26', 'Unnamed: 27',
                'WristAccelerometer', 'Unnamed: 30', 'Unnamed: 31',
                'WristAngularVelocity', 'Unnamed: 33', 'Unnamed: 34']]

    new_data_frame = pd.DataFrame()
    for person in persons:
        for action in actions:
            for trial in trials:
                print(f'Processing {person} {action} {trial}')
                # Get Log file with 2D and 3D info.
                try:
                    camera = 'Camera1'
                    prospect = f'Log_{net2d}_{net3d}_{person}{action}{trial}{camera}.csv'
                    vision_data = pd.read_csv(f'logs/{prospect}')
                    
                    camera = 'Camera2'
                    prospect = f'Log_{net2d}_{net3d}_{person}{action}{trial}{camera}.csv'
                    vision_datac2 = pd.read_csv(f'logs/{prospect}')

                    # Get timestamp from vision file
                    timestamp = list(vision_data['timestamp'])
                    timestampc2 = list(vision_datac2['timestamp'])

                    # Reformat to match
                    if timestamp == timestampc2:
                        timestamp_nf = []
                        for i in timestamp:
                            timestamp_nf.append(i)
                    else:
                        timestamp_match = list(set(timestamp) & set(timestampc2))
                        timestamp_nf = []
                        for i in timestamp_match:
                            timestamp_nf.append(i)
                        vision_data= vision_data[vision_data['timestamp'].isin(timestamp_match)]
                        vision_data= vision_data.drop_duplicates(subset=['timestamp'])
                        vision_datac2 = vision_datac2[vision_datac2['timestamp'].isin(timestamp_match)]
                        
                    # Get the current activity, subject and trial
                    imu_data_actual = imu_data[(imu_data['Subject'] == int(person.split('Subject')[-1]) ) &
                                                (imu_data['Activity'] == int(action.split('Activity')[-1])) & 
                                                (imu_data['Trial'] == int(trial.split('Trial')[-1]) )]
                    

                    # Set timestamp to 0, 1, 2...
                    imu_data_actual['TimeStamps'] =np.arange(0, len(imu_data_actual),dtype="int" )
                    
                    imu_data_actual=imu_data_actual[imu_data_actual['TimeStamps'].isin(timestamp_nf)]
                    
                    imu_data_actual['Subject'] = imu_data_actual['Subject'].astype(int)
                    imu_data_actual['Activity'] = imu_data_actual['Activity'].astype(int)
                    imu_data_actual['Trial'] = imu_data_actual['Trial'].astype(int)
                    imu_data_actual['Tag'] = imu_data_actual['Tag'].astype(int)
                    
                    imu_data_actual['TimeStamps'] =np.arange(0, len(imu_data_actual),dtype="int" )

                    # Change imu data column names
                    imu_data_actual.columns = ['TimeStamps', 'Subject', 'Activity','Trial', 'Tag',
                                'ankle_accelerometer_x', 'ankle_accelerometer_y', 'ankle_accelerometer_z',
                                'ankle_angular_velocity_x', 'ankle_angular_velocity_y', 'ankle_angular_velocity_z',
                                'right_pocket_accelerometer_x',  'right_pocket_accelerometer_y', 'right_pocket_accelerometer_z',
                                'right_pocket_angular_velocity_x', 'right_pocket_angular_velocity_y', 'right_pocket_angular_velocity_z',
                                'belt_accelerometer_x', 'belt_accelerometer_y', 'belt_accelerometer_z',
                                'belt_angular_velocity_x', 'belt_angular_velocity_y', 'belt_angular_velocity_z',
                                'neck_accelerometer_x', 'neck_accelerometer_y', 'neck_accelerometer_z',
                                'neck_angular_velocity_x', 'neck_angular_velocity_y', 'neck_angular_velocity_z',
                                'wrist_accelerometer_x', 'wrist_accelerometer_y', 'wrist_accelerometer_z',
                                'wrist_angular_velocity_x', 'wrist_angular_velocity_y', 'wrist_angular_velocity_z']
                    
                    vision_data.drop('timestamp', inplace=True, axis=1)
                    vision_datac2.drop('timestamp', inplace=True, axis=1)
                    
                    vision_data = vision_data.add_prefix('c1.' )
                    vision_datac2 = vision_datac2.add_prefix('c2.')
                    
                    imu_data_actual.reset_index(drop=True, inplace=True)
                    vision_data.reset_index(drop=True, inplace=True)
                    vision_datac2.reset_index(drop=True, inplace=True)
                    out_data = pd.concat([imu_data_actual, vision_data, vision_datac2], axis=1)

                    # Append data to csv
                    new_data_frame = pd.concat([new_data_frame, out_data],  ignore_index=True)

                except pd.errors.EmptyDataError:
                    prospect = f'Skipped: {person}{action}{trial}'
                    logging.info(prospect)
                    data_to_file= f'{person}/{action}/{trial}'
                    filename = 'missing_files.txt'
                    writeOnFile(data_to_file, filename)

                except FileNotFoundError:
                    prospect = f'Skipped: {person}{action}{trial}'
                    logging.info(prospect)
                    data_to_file= f'{person}/{action}/{trial}'
                    writeOnFile(data_to_file, filename)
    print('Done')
    # Eliminar filas con valores NaN en la columna 'TimeStamps'
    new_data_frame.dropna(subset=['TimeStamps'], inplace=True)
    new_data_frame['TimeStamps'] = new_data_frame['TimeStamps'].astype(int)                
    new_data_frame['Subject'] = new_data_frame['Subject'].astype(int)
    new_data_frame['Activity'] = new_data_frame['Activity'].astype(int)
    new_data_frame['Trial'] = new_data_frame['Trial'].astype(int)
    new_data_frame['Tag'] = new_data_frame['Tag'].astype(int)
    
    new_data_frame.to_csv(os.path.join(dataset_folder, f'{net3d}_processed_upfall.csv'),index=False )
    return

def writeOnFile(data_to_file, filename):
    current_txt = open(os.path.join(dataset_folder, filename),'a')
    current_txt.write("%s \n"%(data_to_file))
    current_txt.close()
    return



if __name__ == "__main__":
    dataset_folder = 'upfall'

    # Create txt files for missing files and processing list
    open(os.path.join(dataset_folder, 'missing_files.txt'), 'w')
    open(os.path.join(dataset_folder,'processing_list.txt'), 'w')

    # Set logger for info and warnings
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    logging.basicConfig(filename=f"logger_harup_processing.log", level=logging.INFO)

    # Main fuctions
    extractZipFiles()
    inferOnData()
    syncData()

    