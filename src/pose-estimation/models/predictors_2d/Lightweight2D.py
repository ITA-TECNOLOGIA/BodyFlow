# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnológico de Aragón (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernández Ruiz                      ahernandez@itainnova.es
#          Ángel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Marañes Nueno                            cmaranes@itainnova.es
#          Irene López Bosque                                ilopez@itainnova.es
#          María de la Vega Rodrigálvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibáñez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocío Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------

"""
Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose

https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
"""
from models.HPE2D import HPE2D
import numpy as np
import torch
import os
from common_pose.BodyLandmarks import BodyLandmarks2d, Landmark

# LightWeight
from models.predictors_2d.lightweight.models.with_mobilenet import PoseEstimationWithMobileNet
from models.predictors_2d.lightweight.modules.keypoints import extract_keypoints, group_keypoints
from models.predictors_2d.lightweight.modules.pose import Pose
from models.predictors_2d.lightweight.modules.load_state import load_state
from models.predictors_2d.lightweight.utils import infer_fast

# https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
class Lightweight2D(HPE2D):
    def __init__(self):
        self._stride = 8
        self._upsample_ratio = 4
        self._num_keypoints = Pose.num_kpts
        self._delay = 1
        self._height_size = 256

        self._lightweight_net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(os.path.join('models', 'lightweight_checkpoint_iter_370000.pth'), map_location='cpu')
        load_state(self._lightweight_net, checkpoint)
        self._lightweight_net = self._lightweight_net.eval()
        if torch.cuda.is_available():
            self._lightweight_net = self._lightweight_net.cuda()

    def get_frame_keypoints(self, frame):
        cpu = not torch.cuda.is_available()

        heatmaps, pafs, scale, pad = infer_fast(self._lightweight_net, frame, self._height_size, self._stride, self._upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self._num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self._stride / self._upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self._stride / self._upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self._num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self._num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        # -> Keypoints order
        # https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/TRAIN-ON-CUSTOM-DATASET.md
        # https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/data/shake_it_off.jpg
        if len(current_poses) == 0:
            return None
        keypoints = current_poses[0].keypoints
        frame_keypoints_2d = {
            'nose' : list(keypoints[0]),
            'left_eye_inner' : list(keypoints[0]),
            'left_eye' : list(keypoints[0]),
            'left_eye_outer' : list(keypoints[0]),
            'right_eye_inner' : list(keypoints[0]),
            'right_eye' : list(keypoints[0]),
            'right_eye_outer' : list(keypoints[0]),
            'left_ear' : list(keypoints[0]),
            'right_ear' :list(keypoints[0]),
            'mouth_left' : list(keypoints[0]),
            'mouth_right' : list(keypoints[0]),
            'left_shoulder' : list(keypoints[5]),
            'right_shoulder' : list(keypoints[2]),
            'left_elbow' : list(keypoints[6]),
            'right_elbow' : list(keypoints[3]),
            'left_wrist' : list(keypoints[7]),
            'right_wrist' : list(keypoints[4]),
            'left_pinky' : list(keypoints[7]),
            'right_pinky' : list(keypoints[4]),
            'left_index' : list(keypoints[7]),
            'right_index' : list(keypoints[4]),
            'left_thumb' : list(keypoints[7]),
            'right_thumb' : list(keypoints[4]),
            'left_hip' : list(keypoints[11]),
            'right_hip' : list(keypoints[8]),
            'left_knee' : list(keypoints[12]),
            'right_knee' : list(keypoints[9]),
            'left_ankle' : list(keypoints[13]),
            'right_ankle' : list(keypoints[10]),
            'left_heel' : list(keypoints[13]),
            'right_heel' : list(keypoints[10]),
            'left_foot_index' : list(keypoints[13]),
            'right_foot_index' : list(keypoints[10])
        }
        
        # Additional keypoints
        hips = (keypoints[11] + keypoints[8]) / 2
        frame_keypoints_2d['hips'] = hips
        chest = (keypoints[5] + keypoints[2]) / 2
        frame_keypoints_2d['chest'] = chest
        frame_keypoints_2d['spine'] = (hips + chest) / 2
        frame_keypoints_2d['jaw'] = list(keypoints[1]) # neck 

        for keypoint in frame_keypoints_2d.keys():
            value = frame_keypoints_2d[keypoint]
            value = np.array(value)
            frame_keypoints_2d[keypoint] = value

        frame_keypoints_2d =  BodyLandmarks2d(
            nose = Landmark("nose", "2d", frame_keypoints_2d['nose'], False),
            left_eye_inner = Landmark("left_eye_inner", "2d", frame_keypoints_2d['left_eye_inner'], True),
            left_eye = Landmark("left_eye", "2d", frame_keypoints_2d['left_eye'], False),
            left_eye_outer = Landmark("left_eye_outer", "2d", frame_keypoints_2d['left_eye_outer'], True),
            right_eye_inner = Landmark("right_eye_inner", "2d", frame_keypoints_2d['right_eye_inner'], True),
            right_eye = Landmark("right_eye", "2d", frame_keypoints_2d['right_eye'], False),
            right_eye_outer = Landmark("right_eye_outer", "2d", frame_keypoints_2d['right_eye_outer'], True),
            left_ear = Landmark("left_ear", "2d", frame_keypoints_2d['left_ear'], False),
            right_ear = Landmark("right_ear", "2d", frame_keypoints_2d['right_ear'], False),
            mouth_left = Landmark("mouth_left", "2d", frame_keypoints_2d['mouth_left'], True),
            mouth_right = Landmark("mouth_right", "2d", frame_keypoints_2d['mouth_right'], True),
            left_shoulder = Landmark("left_shoulder", "2d", frame_keypoints_2d['left_shoulder'], False),
            right_shoulder = Landmark("right_shoulder", "2d", frame_keypoints_2d['right_shoulder'], False),
            left_elbow = Landmark("left_elbow", "2d", frame_keypoints_2d['left_elbow'], False),
            right_elbow = Landmark("right_elbow", "2d", frame_keypoints_2d['right_elbow'], False),
            left_wrist = Landmark("left_wrist", "2d", frame_keypoints_2d['left_wrist'], False),
            right_wrist = Landmark("right_wrist", "2d", frame_keypoints_2d['right_wrist'], False),
            left_pinky = Landmark("left_pinky", "2d", frame_keypoints_2d['left_pinky'], True),
            right_pinky = Landmark("right_pinky", "2d", frame_keypoints_2d['right_pinky'], True),
            left_index = Landmark("left_index", "2d", frame_keypoints_2d['left_index'], True),
            right_index = Landmark("right_index", "2d", frame_keypoints_2d['right_index'], True),
            left_thumb = Landmark("left_thumb", "2d", frame_keypoints_2d['left_thumb'], True),
            right_thumb = Landmark("right_thumb", "2d", frame_keypoints_2d['right_thumb'], True),
            left_hip = Landmark("left_hip", "2d", frame_keypoints_2d['left_hip'], False),
            right_hip = Landmark("right_hip", "2d", frame_keypoints_2d['right_hip'], False),
            left_knee = Landmark("left_knee", "2d", frame_keypoints_2d['left_knee'], False),
            right_knee = Landmark("right_knee", "2d", frame_keypoints_2d['right_knee'], False),
            left_ankle = Landmark("left_ankle", "2d", frame_keypoints_2d['left_ankle'], False),
            right_ankle = Landmark("right_ankle", "2d", frame_keypoints_2d['right_ankle'], False),
            left_heel = Landmark("left_heel", "2d", frame_keypoints_2d['left_heel'], True),
            right_heel = Landmark("right_heel", "2d", frame_keypoints_2d['right_heel'], True),
            left_foot_index = Landmark("left_foot_index", "2d", frame_keypoints_2d['left_foot_index'], True),
            right_foot_index = Landmark("right_foot_index", "2d", frame_keypoints_2d['right_foot_index'], True),
            jaw = Landmark("jaw", "2d", frame_keypoints_2d['jaw'], True),
            chest = Landmark("chest", "2d", frame_keypoints_2d['chest'], False),
            spine = Landmark("spine", "2d", frame_keypoints_2d['spine'], True),
            hips = Landmark("hips", "2d", frame_keypoints_2d['hips'], True),
        )

        return frame_keypoints_2d