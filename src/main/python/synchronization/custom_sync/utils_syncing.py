## --------------------------------------------------------------------------------
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
import numpy as np
import cv2
from moviepy.editor import *
from scipy.signal import find_peaks
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

max_seconds_reach_peak = 10 # The for sync must be found in the first max_seconds_reach_peak seconds

def sync_cameras(cameras_paths: dict):
    cameras_starting_frame = {}
    for camera, video_path in cameras_paths.items():
        frame_start = get_cam_start(video_path["video_path_raw"])
        cameras_starting_frame[camera] = frame_start
    return cameras_starting_frame

def get_cam_start(video_path):
    # Check video fps
    vidcap = cv2.VideoCapture(video_path)
    fps = int(os.environ.get("SENSORS_CAM_FREQUENCY")) #vidcap.get(cv2.CAP_PROP_FPS)


    audio_path = os.path.join("aux.wav")
    clip = AudioFileClip(video_path)
    clip.write_audiofile(audio_path, logger=None)
    audio, sr = librosa.load(audio_path, sr = None)
    os.remove(audio_path)

    audio = audio[:sr*max_seconds_reach_peak] # Cut audio
    if len(audio) > 0:
        img = librosa.display.waveshow(audio, sr=sr)
        plt.savefig("librosa.png")
        peaks, dict_ret = find_peaks(audio, height = 0)

        num = np.where(audio == np.max(dict_ret['peak_heights']))
        cam_start = float(num[0]*fps)/sr
        return int(cam_start)
    else:
        return None


def sync_sensors(sample, sensors_df_full):
    subject_id, activity_id, trial_id = sample['subject_id'], sample['activity_id'], sample['trial_id']
    sensors_df = sensors_df_full[(sensors_df_full['subject'] == subject_id) & (sensors_df_full['activity'] == activity_id) & (sensors_df_full['trial'] == trial_id)]
    sensors_start = get_imu_start(sensors_df)
    sensors_df = sensors_df.iloc[sensors_start:, :]
    return sensors_df

def get_imu_start(imus_df):
    axes = ['Acc_X_chest_D', 'Acc_X_chest_D', 'Acc_X_chest_D']
    sensors_frequency = int(os.environ.get("SENSORS_CAM_FREQUENCY"))
    acc_axes = imus_df[axes].to_numpy()[:sensors_frequency*max_seconds_reach_peak]
    acc = np.linalg.norm(acc_axes, axis=1)
    peaks, dict_ret = find_peaks(acc, height = 0)
    peak_max = np.where(acc == np.max(dict_ret['peak_heights']))
    starting_timestamp = peak_max[0][0]

    return starting_timestamp
