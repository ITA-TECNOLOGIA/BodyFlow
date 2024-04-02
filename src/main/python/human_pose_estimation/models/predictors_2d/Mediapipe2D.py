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

"""
Mediapipe Pose Estimation

# https://google.github.io/mediapipe/solutions/pose.html
"""
from human_pose_estimation.models.HPE2D import HPE2D
import numpy as np
import cv2
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks2d, Landmark

import mediapipe as mp

class Mediapipe2D(HPE2D):
    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def get_frame_keypoints(self, frame_full, cropped_frame, bbox):
        RGB = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        # process the RGB frame to get the result
        results = self._pose.process(RGB)
        if results.pose_landmarks is None:
            return None
        landmarks = results.pose_landmarks.landmark

        # https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
        frame_keypoints_2d = {
            'nose' : [landmarks[self._mp_pose.PoseLandmark.NOSE].x, landmarks[self._mp_pose.PoseLandmark.NOSE].y],
            'left_eye_inner' : [landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_INNER].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_INNER].y],
            'left_eye' : [landmarks[self._mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[self._mp_pose.PoseLandmark.LEFT_EYE].y],
            'left_eye_outer' : [landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_OUTER].x, landmarks[self._mp_pose.PoseLandmark.LEFT_EYE_OUTER].y],
            'right_eye_inner' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_INNER].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_INNER].y],
            'right_eye' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE].y],
            'right_eye_outer' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_OUTER].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y],
            'left_ear' : [landmarks[self._mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[self._mp_pose.PoseLandmark.LEFT_EAR].y],
            'right_ear' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_EAR].y],
            'mouth_left' : [landmarks[self._mp_pose.PoseLandmark.MOUTH_LEFT].x, landmarks[self._mp_pose.PoseLandmark.MOUTH_LEFT].y],
            'mouth_right' : [landmarks[self._mp_pose.PoseLandmark.MOUTH_RIGHT].x, landmarks[self._mp_pose.PoseLandmark.MOUTH_RIGHT].y],
            'left_shoulder' : [landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER].y],
            'right_shoulder' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
            'left_elbow' : [landmarks[self._mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[self._mp_pose.PoseLandmark.LEFT_ELBOW].y],
            'right_elbow' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_ELBOW].y],
            'left_wrist' : [landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST].y],
            'right_wrist' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST].y],
            'left_pinky' : [landmarks[self._mp_pose.PoseLandmark.LEFT_PINKY].x, landmarks[self._mp_pose.PoseLandmark.LEFT_PINKY].y],
            'right_pinky' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_PINKY].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_PINKY].y],
            'left_index' : [landmarks[self._mp_pose.PoseLandmark.LEFT_INDEX].x, landmarks[self._mp_pose.PoseLandmark.LEFT_INDEX].y],
            'right_index' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_INDEX].y],
            'left_thumb' : [landmarks[self._mp_pose.PoseLandmark.LEFT_THUMB].x, landmarks[self._mp_pose.PoseLandmark.LEFT_THUMB].y],
            'right_thumb' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_THUMB].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_THUMB].y],
            'left_hip' : [landmarks[self._mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[self._mp_pose.PoseLandmark.LEFT_HIP].y],
            'right_hip' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP].y],
            'left_knee' : [landmarks[self._mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[self._mp_pose.PoseLandmark.LEFT_KNEE].y],
            'right_knee' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_KNEE].y],
            'left_ankle' : [landmarks[self._mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[self._mp_pose.PoseLandmark.LEFT_ANKLE].y],
            'right_ankle' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_ANKLE].y],
            'left_heel' : [landmarks[self._mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[self._mp_pose.PoseLandmark.LEFT_HEEL].y],
            'right_heel' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_HEEL].y],
            'left_foot_index' : [landmarks[self._mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[self._mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y],
            'right_foot_index' : [landmarks[self._mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks[self._mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
        }

        # Additional keypoints
        hips_x = (landmarks[self._mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
        hips_y = (landmarks[self._mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
        frame_keypoints_2d['hips'] = [hips_x, hips_y]

        chest_x = (landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
        chest_y = (landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        frame_keypoints_2d['chest'] = [chest_x, chest_y]

        spine_x = (hips_x + chest_x) / 2
        spine_y = (hips_y + chest_y) / 2
        frame_keypoints_2d['spine'] = [spine_x, spine_y]

        jaw_x = (chest_x + landmarks[self._mp_pose.PoseLandmark.NOSE].x) / 2
        jaw_y = (chest_y + landmarks[self._mp_pose.PoseLandmark.NOSE].y) / 2
        frame_keypoints_2d['jaw'] = [jaw_x, jaw_y]

        for keypoint in frame_keypoints_2d.keys():
            value = frame_keypoints_2d[keypoint]
            value = np.array(value)
            value[0] *= cropped_frame.shape[1]
            value[1] *= cropped_frame.shape[0]
            frame_keypoints_2d[keypoint] = value
        

        frame_keypoints_2d =  BodyLandmarks2d(
            nose = Landmark("nose", "2d", frame_keypoints_2d['nose'], False, bbox),
            left_eye_inner = Landmark("left_eye_inner", "2d", frame_keypoints_2d['left_eye_inner'], False, bbox),
            left_eye = Landmark("left_eye", "2d", frame_keypoints_2d['left_eye'], False, bbox),
            left_eye_outer = Landmark("left_eye_outer", "2d", frame_keypoints_2d['left_eye_outer'], False, bbox),
            right_eye_inner = Landmark("right_eye_inner", "2d", frame_keypoints_2d['right_eye_inner'], False, bbox),
            right_eye = Landmark("right_eye", "2d", frame_keypoints_2d['right_eye'], False, bbox),
            right_eye_outer = Landmark("right_eye_outer", "2d", frame_keypoints_2d['right_eye_outer'], False, bbox),
            left_ear = Landmark("left_ear", "2d", frame_keypoints_2d['left_ear'], False, bbox),
            right_ear = Landmark("right_ear", "2d", frame_keypoints_2d['right_ear'], False, bbox),
            mouth_left = Landmark("mouth_left", "2d", frame_keypoints_2d['mouth_left'], False, bbox),
            mouth_right = Landmark("mouth_right", "2d", frame_keypoints_2d['mouth_right'], False, bbox),
            left_shoulder = Landmark("left_shoulder", "2d", frame_keypoints_2d['left_shoulder'], False, bbox),
            right_shoulder = Landmark("right_shoulder", "2d", frame_keypoints_2d['right_shoulder'], False, bbox),
            left_elbow = Landmark("left_elbow", "2d", frame_keypoints_2d['left_elbow'], False, bbox),
            right_elbow = Landmark("right_elbow", "2d", frame_keypoints_2d['right_elbow'], False, bbox),
            left_wrist = Landmark("left_wrist", "2d", frame_keypoints_2d['left_wrist'], False, bbox),
            right_wrist = Landmark("right_wrist", "2d", frame_keypoints_2d['right_wrist'], False, bbox),
            left_pinky = Landmark("left_pinky", "2d", frame_keypoints_2d['left_pinky'], False, bbox),
            right_pinky = Landmark("right_pinky", "2d", frame_keypoints_2d['right_pinky'], False, bbox),
            left_index = Landmark("left_index", "2d", frame_keypoints_2d['left_index'], False, bbox),
            right_index = Landmark("right_index", "2d", frame_keypoints_2d['right_index'], False, bbox),
            left_thumb = Landmark("left_thumb", "2d", frame_keypoints_2d['left_thumb'], False, bbox),
            right_thumb = Landmark("right_thumb", "2d", frame_keypoints_2d['right_thumb'], False, bbox),
            left_hip = Landmark("left_hip", "2d", frame_keypoints_2d['left_hip'], False, bbox),
            right_hip = Landmark("right_hip", "2d", frame_keypoints_2d['right_hip'], False, bbox),
            left_knee = Landmark("left_knee", "2d", frame_keypoints_2d['left_knee'], False, bbox),
            right_knee = Landmark("right_knee", "2d", frame_keypoints_2d['right_knee'], False, bbox),
            left_ankle = Landmark("left_ankle", "2d", frame_keypoints_2d['left_ankle'], False, bbox),
            right_ankle = Landmark("right_ankle", "2d", frame_keypoints_2d['right_ankle'], False, bbox),
            left_heel = Landmark("left_heel", "2d", frame_keypoints_2d['left_heel'], False, bbox),
            right_heel = Landmark("right_heel", "2d", frame_keypoints_2d['right_heel'], False, bbox),
            left_foot_index = Landmark("left_foot_index", "2d", frame_keypoints_2d['left_foot_index'], False, bbox),
            right_foot_index = Landmark("right_foot_index", "2d", frame_keypoints_2d['right_foot_index'], False, bbox),
            jaw = Landmark("jaw", "2d", frame_keypoints_2d['jaw'], True, bbox),
            chest = Landmark("chest", "2d", frame_keypoints_2d['chest'], True, bbox),
            spine = Landmark("spine", "2d", frame_keypoints_2d['spine'], True, bbox),
            hips = Landmark("hips", "2d", frame_keypoints_2d['hips'], True, bbox),
            bbox = bbox,
        )

        return frame_keypoints_2d
