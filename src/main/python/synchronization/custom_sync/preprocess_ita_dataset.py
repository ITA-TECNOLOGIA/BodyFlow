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

# Script that takes the raw data and preprocess it for training
import os
from utils.utils_processing import get_samples_itainnova, blur_video_face, compute_poses
from utils.utils_syncing import sync_cameras, sync_sensors
from dotenv import load_dotenv, find_dotenv
import pandas as pd

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

def preprocess_dataset(dataset_raw_path, dataset_processed_path):

    sensors_csv_path = os.environ.get("SENSORS_CSV_PATH")
    sensors_df_full = pd.read_csv(sensors_csv_path)
    
    samples, logs = get_samples_itainnova(dataset_raw_path, dataset_processed_path)
    for log in logs:
        print(log)
    
    for sample in samples:

        ##################################################
        ####        Syncing video and sensors         ####
        ##################################################
        cameras_starting_frame = sync_cameras(sample["videos_path"])
        sensors_df = sync_sensors(sample, sensors_df_full)


        ##################################################
        ####        Save blurred camera frames        ####
        ##################################################
        #blur_video_face(sample)

        ##################################################
        ####               Compute poses              ####
        ##################################################
        compute_poses(sample)  # Needed the blurred face frames

    pass


if __name__ == "__main__":
    dataset_raw_path = os.path.join("data", "ita", "raw")
    dataset_processed_path = os.path.join("data", "ita", "processed")      
    preprocess_dataset(dataset_raw_path, dataset_processed_path)
