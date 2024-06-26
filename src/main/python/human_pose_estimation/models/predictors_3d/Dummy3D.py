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

from human_pose_estimation.models.HPE3D import HPE3D
import numpy as np
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks3d, Landmark


class Dummy3D(HPE3D):
    def __init__(self, window_length):
        super().__init__(window_length)

    def translate_keypoints_2d(self, keypoints_2d):
        return np.zeros(shape=(17, 2))

    def get_3d_keypoints(self, img_w, img_h, bodyLandmarks2d, timestamp, input_2D_no, frame, bbox):
        body_landmarks = BodyLandmarks3d(
            raw=np.zeros(shape=(3, 17)),
            timestamp=timestamp,
            nose = Landmark("nose", "3d",np.array([0., 0., 0.]), True),
            left_eye_inner = Landmark("left_eye_inner", "3d", np.array([0., 0., 0.]), True),
            left_eye = Landmark("left_eye", "3d", np.array([0., 0., 0.]), True),
            left_eye_outer = Landmark("left_eye_outer", "3d", np.array([0., 0., 0.]), True),
            right_eye_inner = Landmark("right_eye_inner", "3d", np.array([0., 0., 0.]), True),
            right_eye = Landmark("right_eye", "3d", np.array([0., 0., 0.]), True),
            right_eye_outer = Landmark("right_eye_outer", "3d", np.array([0., 0., 0.]), True),
            left_ear = Landmark("left_ear", "3d", np.array([0., 0., 0.]), True),
            right_ear = Landmark("right_ear", "3d", np.array([0., 0., 0.]), True),
            mouth_left = Landmark("mouth_left", "3d", np.array([0., 0., 0.]), True),
            mouth_right = Landmark("mouth_right", "3d", np.array([0., 0., 0.]), True),
            left_shoulder = Landmark("left_shoulder", "3d", np.array([0., 0., 0.]), True),
            right_shoulder = Landmark("right_shoulder", "3d", np.array([0., 0., 0.]), True),
            left_elbow = Landmark("left_elbow", "3d", np.array([0., 0., 0.]), True),
            right_elbow = Landmark("right_elbow", "3d", np.array([0., 0., 0.]), True),
            left_wrist = Landmark("left_wrist", "3d", np.array([0., 0., 0.]), True),
            right_wrist = Landmark("right_wrist", "3d", np.array([0., 0., 0.]), True),
            left_pinky = Landmark("left_pinky", "3d", np.array([0., 0., 0.]), True),
            right_pinky = Landmark("right_pinky", "3d", np.array([0., 0., 0.]), True),
            left_index = Landmark("left_index", "3d", np.array([0., 0., 0.]), True),
            right_index = Landmark("right_index", "3d", np.array([0., 0., 0.]), True),
            left_thumb = Landmark("left_thumb", "3d", np.array([0., 0., 0.]), True),
            right_thumb = Landmark("right_thumb", "3d", np.array([0., 0., 0.]), True),
            left_hip = Landmark("left_hip", "3d", np.array([0., 0., 0.]), True),
            right_hip = Landmark("right_hip", "3d", np.array([0., 0., 0.]), True),
            left_knee = Landmark("left_knee", "3d", np.array([0., 0., 0.]), True),
            right_knee = Landmark("right_knee", "3d", np.array([0., 0., 0.]), True),
            left_ankle = Landmark("left_ankle", "3d", np.array([0., 0., 0.]), True),
            right_ankle = Landmark("right_ankle", "3d", np.array([0., 0., 0.]), True),
            left_heel = Landmark("left_heel", "3d", np.array([0., 0., 0.]), True),
            right_heel = Landmark("right_heel", "3d", np.array([0., 0., 0.]), True),
            left_foot_index = Landmark("left_foot_index", "3d", np.array([0., 0., 0.]), True),
            right_foot_index = Landmark("right_foot_index", "3d", np.array([0., 0., 0.]), True),
            jaw = Landmark("jaw", "3d", np.array([0., 0., 0.]), True),
            chest = Landmark("chest", "3d", np.array([0., 0., 0.]), True),
            spine = Landmark("spine", "3d", np.array([0., 0., 0.]), True),
            hips = Landmark("hips", "3d", np.array([0., 0., 0.]), True),
            # 2D Keypoints
            bodyLandmarks2d=bodyLandmarks2d
        )

        return body_landmarks