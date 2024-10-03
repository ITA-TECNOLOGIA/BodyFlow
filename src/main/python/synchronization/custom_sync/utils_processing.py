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
from dotenv import load_dotenv, find_dotenv
from deepface import DeepFace
import cv2

import sys
sys.path.append("src/pose-estimation")
from inference_server import inference
from video_capture.VideoFromImages import VideoFromImages
from models.predictors_2d.CPN import CPN
from models.predictors_3d.MHFormer import MHFormer
from models.HPE import HPE
from common.PoseLogger import PoseLogger
from unimatch.compute_flow import compute_optical_flow



dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def get_samples_itainnova(dataset_raw_path, dataset_processed_path):
    """
    This returns a list of dicts which the necessary info to read
    the samples paths, etc.
    """
    logs = []
    n_trials = int(os.environ.get("N_TRIALS"))
    n_cameras = int(os.environ.get("N_CAMERAS"))
    samples = []

    for subject in os.listdir(dataset_raw_path):
        subject_path_raw = os.path.join(dataset_raw_path, subject)
        if not os.path.isdir(subject_path_raw):
            continue
        subject_id = int(subject.split("_")[0]) # {id}_sujeto
        subject_path_processed = os.path.join(dataset_processed_path, f"Subject{subject_id}")

        for activity in os.listdir(subject_path_raw):
            activity_path_raw = os.path.join(subject_path_raw, activity)
            if not os.path.isdir(activity_path_raw):
                continue
            activity_id, activity_name = activity.split("_") # {id}_{activity}
            activity_id = int(activity_id)
            activity_path_processed = os.path.join(subject_path_processed, f"Activity{activity_id}")

            for trial_id in range(1, n_trials+1):
                trial = f"{trial_id}_trial"
                trial_path_raw = os.path.join(activity_path_raw, trial)
                trial_path_processed = os.path.join(activity_path_processed, f"Trial{trial_id}")
                if not os.path.isdir(trial_path_raw):
                    logs.append(f"Trial {trial_path_raw} not found")
                    continue

                videos_path = {}
                for camera_id in range(1, n_cameras+1):
                    camera_folder = f"camara_{camera_id}"
                    video_folder_path = os.path.join(trial_path_raw, "videos", camera_folder)
                    video_name = None
                    for video_name_aux in os.listdir(video_folder_path):
                        if ".MP4" in video_name_aux:
                            video_name = video_name_aux

                    video_path_raw = os.path.join(video_folder_path, video_name)
                    if video_path_raw is None:
                        logs.append(f"Video {video_folder_path} not found")
                        continue
                    videos_path[camera_folder] = {
                        "video_path_raw" : video_path_raw,
                        "video_path_processed" : os.path.join(trial_path_processed, f"Camera{camera_id}")
                    }

                
                sample = {
                    "subject_id" : subject_id,
                    "subject_path_raw" : subject_path_raw,
                    "subject_path_processed" : subject_path_processed,
                    "activity_id" : activity_id,
                    "activity_name" : activity_name,
                    "activity_path_raw" : activity_path_raw,
                    "activity_path_processed" : activity_path_processed,
                    "trial_id" : trial_id,
                    "trial_path_raw" : trial_path_raw,
                    "trial_path_processed" : trial_path_processed,
                    "videos_path" : videos_path
                }
                samples.append(sample)
    return samples, logs


def blur_video_face(sample):
    for video_path_dict in sample['videos_path'].values():
        video_path_raw = video_path_dict["video_path_raw"]
        video_path_processed = video_path_dict["video_path_processed"]

        # Create folder
        os.makedirs(video_path_processed, exist_ok=True)
        
        vidcap = cv2.VideoCapture(video_path_raw)
        success,image = vidcap.read()
        count = 0
        while success:
            output_aux_frame = os.path.join("frame.png")
            cv2.imwrite(output_aux_frame, image)
            face_blurred = blur_face_frames(output_aux_frame)
            os.remove(output_aux_frame)
            cv2.imwrite(os.path.join(video_path_processed, "frame%d.png" % count), face_blurred)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame: ', count)
            if count > 100:
                break
            count += 1

        
    pass

def blur_face_frames(frame_path: str):
    frame = cv2.imread(frame_path)
    try:
        obj = DeepFace.analyze(img_path = frame_path, actions=['age'], detector_backend="retinaface", prog_bar=False)
        x, y, w, h = obj['region']['x'], obj['region']['y'], obj['region']['w'], obj['region']['h']
        roi = frame[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(roi, (51,51), 0)
        frame[y:y+h, x:x+w] = blur
    except ValueError: # Face not detected
        pass
    
    return frame

def compute_poses(sample):
    poses_csv_paths = []
    for video_path_dict in sample['videos_path'].values():
        video_path_raw = video_path_dict["video_path_raw"]
        video_path_processed = video_path_dict["video_path_processed"]

        videoFromImages = VideoFromImages(video_path_processed, False)
        predictor_2d = CPN()
        predictor_3d = MHFormer()
        human_pose_estimator = HPE(predictor_2d, predictor_3d)
        pose_logger = PoseLogger(-1, filename=log_filename)

    
    for video_folder in cameras_folder:
        images_video_path = os.path.join(aux_folder, video_folder)
        
        predictor_2d = CPN()
        predictor_3d = MHFormer()
        human_pose_estimator = HPE(predictor_2d, predictor_3d)
        pose_logger = PoseLogger(-1)
        pose_ok = inference(videoFromImages, human_pose_estimator, pose_logger, export_csv=False)
        poses = pose_logger.get_poses()[cameras_frame_start:]
        poses_3d.append(poses)
        min_poses = min(min_poses, len(poses))

