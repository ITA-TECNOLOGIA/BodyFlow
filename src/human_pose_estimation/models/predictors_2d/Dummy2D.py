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

from models.HPE2D import HPE2D
import numpy as np
from common_pose.BodyLandmarks import BodyLandmarks2d, Landmark

class Dummy2D(HPE2D):
    def __init__(self):
        pass

    def get_frame_keypoints(self, frame):
        frame_keypoints_2d = BodyLandmarks2d(
            nose = Landmark("nose", "2d",np.array([0., 0.]), True),
            left_eye_inner = Landmark("left_eye_inner", "2d", np.array([0., 0.]), True),
            left_eye = Landmark("left_eye", "2d", np.array([0., 0.]), True),
            left_eye_outer = Landmark("left_eye_outer", "2d", np.array([0., 0.]), True),
            right_eye_inner = Landmark("right_eye_inner", "2d", np.array([0., 0.]), True),
            right_eye = Landmark("right_eye", "2d", np.array([0., 0.]), True),
            right_eye_outer = Landmark("right_eye_outer", "2d", np.array([0., 0.]), True),
            left_ear = Landmark("left_ear", "2d", np.array([0., 0.]), True),
            right_ear = Landmark("right_ear", "2d", np.array([0., 0.]), True),
            mouth_left = Landmark("mouth_left", "2d", np.array([0., 0.]), True),
            mouth_right = Landmark("mouth_right", "2d", np.array([0., 0.]), True),
            left_shoulder = Landmark("left_shoulder", "2d", np.array([0., 0.]), True),
            right_shoulder = Landmark("right_shoulder", "2d", np.array([0., 0.]), True),
            left_elbow = Landmark("left_elbow", "2d", np.array([0., 0.]), True),
            right_elbow = Landmark("right_elbow", "2d", np.array([0., 0.]), True),
            left_wrist = Landmark("left_wrist", "2d", np.array([0., 0.]), True),
            right_wrist = Landmark("right_wrist", "2d", np.array([0., 0.]), True),
            left_pinky = Landmark("left_pinky", "2d", np.array([0., 0.]), True),
            right_pinky = Landmark("right_pinky", "2d", np.array([0., 0.]), True),
            left_index = Landmark("left_index", "2d", np.array([0., 0.]), True),
            right_index = Landmark("right_index", "2d", np.array([0., 0.]), True),
            left_thumb = Landmark("left_thumb", "2d", np.array([0., 0.]), True),
            right_thumb = Landmark("right_thumb", "2d", np.array([0., 0.]), True),
            left_hip = Landmark("left_hip", "2d", np.array([0., 0.]), True),
            right_hip = Landmark("right_hip", "2d", np.array([0., 0.]), True),
            left_knee = Landmark("left_knee", "2d", np.array([0., 0.]), True),
            right_knee = Landmark("right_knee", "2d", np.array([0., 0.]), True),
            left_ankle = Landmark("left_ankle", "2d", np.array([0., 0.]), True),
            right_ankle = Landmark("right_ankle", "2d", np.array([0., 0.]), True),
            left_heel = Landmark("left_heel", "2d", np.array([0., 0.]), True),
            right_heel = Landmark("right_heel", "2d", np.array([0., 0.]), True),
            left_foot_index = Landmark("left_foot_index", "2d", np.array([0., 0.]), True),
            right_foot_index = Landmark("right_foot_index", "2d", np.array([0., 0.]), True),
            jaw = Landmark("jaw", "2d", np.array([0., 0.]), True),
            chest = Landmark("chest", "2d", np.array([0., 0.]), True),
            spine = Landmark("spine", "2d", np.array([0., 0.]), True),
            hips = Landmark("hips", "2d", np.array([0., 0.]), True)
        )

        return frame_keypoints_2d