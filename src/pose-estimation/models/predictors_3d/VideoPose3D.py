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

"""
3D human pose estimation in video with temporal convolutions and semi-supervised training

https://github.com/facebookresearch/VideoPose3D
"""
import torch
import numpy as np
from models.predictors_3d.videopose.model import TemporalModel
import os
from common_pose.BodyLandmarks import BodyLandmarks3d, Landmark
from common_pose.utils import normalize_screen_coordinates
import logging


class VideoPose3D:
    def __init__(self, window_length=243):
        self._window_length = window_length
        self._kps_left = [1, 3, 5, 7, 9, 11, 13, 15]
        self._joints_left = [4, 5, 6, 11, 12, 13]
        self._kps_right = [2, 4, 6, 8, 10, 12, 14, 16]
        self._joints_right = [1, 2, 3, 14, 15, 16]
        joints_number = 17
        joints_coordinate_space = 2
        filter_widths = [3, 3, 3, 3, 3]
        causal = False
        dropout = 0.25
        channels = 1024
        dense = False
        self._model_pos = TemporalModel(joints_number, joints_coordinate_space, joints_number,
                            filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                            dense=dense)
        checkpoint = torch.load(os.path.join('models', 'pretrained_h36m_detectron_coco.bin'), map_location=lambda storage, loc: storage)
        logging.debug('This model was trained for {} epochs'.format(checkpoint['epoch']))
        self._model_pos.load_state_dict(checkpoint['model_pos'])
        if torch.cuda.is_available():
            self._model_pos = self._model_pos.cuda()
        with torch.no_grad():
            self._model_pos.eval()

        self._buffer_frames = [] # buffer of positions, since it predicts from video
        self._buffer_keypoints_2d = []
        self._buffer_timestamps = []
        self._buffer_bodyLandmarks2d = []

    def buffer_full(self):
        return self._window_length == len(self._buffer_keypoints_2d)
    
    def buffer_empty(self):
        return self._buffer_keypoints_2d == []

    def translate_keypoints_2d(self, keypoints_2d):
        if keypoints_2d is None:
            return None

        frame_keypoints_2d = [  # Order matters
            keypoints_2d._nose.coordinates, # 0
            keypoints_2d._left_eye.coordinates, # 1
            keypoints_2d._right_eye.coordinates, # 2
            keypoints_2d._left_ear.coordinates, # 3
            keypoints_2d._right_ear.coordinates, # 4
            keypoints_2d._left_shoulder.coordinates, # 5
            keypoints_2d._right_shoulder.coordinates, # 6
            keypoints_2d._left_elbow.coordinates, # 7
            keypoints_2d._right_elbow.coordinates, # 8
            keypoints_2d._left_wrist.coordinates, # 9
            keypoints_2d._right_wrist.coordinates, # 10
            keypoints_2d._left_hip.coordinates, # 11
            keypoints_2d._right_hip.coordinates, # 12
            keypoints_2d._left_knee.coordinates, # 13
            keypoints_2d._right_knee.coordinates, # 14
            keypoints_2d._left_ankle.coordinates, # 15
            keypoints_2d._right_ankle.coordinates, # 16
        ]
        frame_keypoints_2d = np.array(frame_keypoints_2d)
        return frame_keypoints_2d

    def add_frame(self, frame, keypoints_2d, timestamp, bodyLandmarks2d):
        if keypoints_2d is None: # No keypoints in this frame
            if self.buffer_empty():
                 # Skip this frame
                 return False
            else:
                # Repeat frame
                keypoints_2d = self._buffer_keypoints_2d[0]
                bodyLandmarks2d = self._buffer_bodyLandmarks2d[0]

        if self.buffer_full(): # Drop last frame
            self._buffer_frames.pop()
            self._buffer_keypoints_2d.pop()
            self._buffer_timestamps.pop()
            self._buffer_bodyLandmarks2d.pop()

        self._buffer_frames.insert(0, frame)
        self._buffer_keypoints_2d.insert(0, keypoints_2d)
        self._buffer_timestamps.insert(0, timestamp)
        self._buffer_bodyLandmarks2d.insert(0, bodyLandmarks2d)
        return self.buffer_full()

    def init_buffers(self, frame, keypoints_2d, timestamp, bodyLandmarks2d):
        if self.buffer_empty():  # Buffer not initialized
            # Replicate the first frame until complete half of the window (padding)
            for _ in range(self._window_length // 2): # TODO is -1?
                self.add_frame(frame, keypoints_2d, timestamp, bodyLandmarks2d)

        return self.add_frame(frame, keypoints_2d, timestamp, bodyLandmarks2d)
    
    def destroy_buffer(self):
        half_window = (self._window_length // 2) + 2
        timestamp_no = self._buffer_timestamps[:half_window].count(self._buffer_timestamps[0])
        expected_timestamps = len(self._buffer_timestamps[:half_window])
        buffer_ended = timestamp_no == expected_timestamps
        if buffer_ended:
            return None
        else:
            body_landmarks = self.get_3d_keypoints()
            self.add_frame(self._buffer_frames[0], self._buffer_keypoints_2d[0], self._buffer_timestamps[0], self._buffer_bodyLandmarks2d[0])
            return body_landmarks

    def get_3d_keypoints(self):
        frame_no = self._window_length // 2
        img = self._buffer_frames[frame_no]
        input_2D_src = self._buffer_keypoints_2d[frame_no]
        timestamp = self._buffer_timestamps[frame_no]
        bodyLandmarks2d = self._buffer_bodyLandmarks2d[frame_no]

        img_size = img.shape
        inputs_2d = np.array(self._buffer_keypoints_2d)
        inputs_2d = normalize_screen_coordinates(inputs_2d, w=img_size[1], h=img_size[0])
        inputs_2d = np.expand_dims(inputs_2d, axis=0)
        inputs_2d = np.concatenate((inputs_2d, inputs_2d), axis=0)

        inputs_2d[1, :, :, 0] *= -1
        inputs_2d[1, :, self._kps_left + self._kps_right] = inputs_2d[1, :, self._kps_right + self._kps_left]
        inputs_2d = torch.from_numpy(inputs_2d.astype('float32'))

        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()

        predicted_3d_pos = self._model_pos(inputs_2d)

        predicted_3d_pos[1, :, :, 0] *= -1
        predicted_3d_pos[1, :, self._joints_left + self._joints_right] = predicted_3d_pos[1, :, self._joints_right + self._joints_left]
        predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
        predicted_3d_pos = predicted_3d_pos.squeeze(0).squeeze(0).cpu().detach().numpy()

        return self.pose_to_landmarks(predicted_3d_pos, bodyLandmarks2d, timestamp)


    def pose_to_landmarks(self, pose, bodyLandmarks2d, timestamp):
        raw = pose
        # I had to infer them from here: https://github.com/Vegetebird/MHFormer/blob/main/demo/vis.py 
        hips = pose[0,:].tolist()
        right_hip = pose[1,:].tolist()
        right_knee = pose[2,:].tolist()
        right_ankle = pose[3,:].tolist()
        left_hip = pose[4,:].tolist()
        left_knee = pose[5,:].tolist()
        left_ankle = pose[6,:].tolist()
        spine = pose[7,:].tolist()
        chest = pose[8,:].tolist()
        jaw = pose[9,:].tolist()
        nose = pose[10,:].tolist()
        left_shoulder = pose[11,:].tolist()
        left_elbow = pose[12,:].tolist()
        left_wrist = pose[13,:].tolist()
        right_shoulder = pose[14,:].tolist()
        right_elbow = pose[15,:].tolist()
        right_wrist = pose[16,:].tolist()

        ######### Interpolate rest 

        # Head
        left_eye_inner = nose
        left_eye = nose
        left_eye_outer = nose
        right_eye_inner = nose
        right_eye = nose
        right_eye_outer = nose
        left_ear = nose
        right_ear = nose
        mouth_left = nose
        mouth_right = nose

        # Arms
        left_pinky = left_wrist
        right_pinky = right_wrist
        left_index = left_wrist
        right_index = right_wrist
        left_thumb = left_wrist
        right_thumb = right_wrist

        # Legs
        left_heel = left_ankle
        right_heel = right_ankle
        left_foot_index = left_ankle
        right_foot_index = right_ankle

        body_landmarks = BodyLandmarks3d(raw=raw,
                                       timestamp=timestamp,
                                       nose=Landmark("nose","3d",nose,False),
                                       left_eye_inner=Landmark("left_eye_inner","3d",left_eye_inner,True),
                                       left_eye=Landmark("left_eye","3d",left_eye,True),
                                       left_eye_outer=Landmark("left_eye_outer","3d",left_eye_outer,True),
                                       right_eye_inner=Landmark("right_eye_inner","3d",right_eye_inner,True),
                                       right_eye=Landmark("right_eye","3d",right_eye,True),
                                       right_eye_outer=Landmark("right_eye_outer","3d",right_eye_outer,True),
                                       left_ear=Landmark("left_ear","3d",left_ear,True),
                                       right_ear=Landmark("right_ear","3d",right_ear,True),
                                       mouth_left=Landmark("mouth_left","3d",mouth_left,True),
                                       mouth_right=Landmark("mouth_right","3d",mouth_right,True),
                                       left_shoulder=Landmark("left_shoulder","3d",left_shoulder,False),
                                       right_shoulder=Landmark("right_shoulder","3d",right_shoulder,False),
                                       left_elbow=Landmark("left_elbow","3d",left_elbow,False),
                                       right_elbow=Landmark("right_elbow","3d",right_elbow,False),
                                       left_wrist=Landmark("left_wrist","3d",left_wrist,False),
                                       right_wrist=Landmark("right_wrist","3d",right_wrist,False),
                                       left_pinky=Landmark("left_pinky","3d",left_pinky,True),
                                       right_pinky=Landmark("right_pinky","3d",right_pinky,True),
                                       left_index=Landmark("left_index","3d",left_index,True),
                                       right_index=Landmark("right_index","3d",right_index,True),
                                       left_thumb=Landmark("left_thumb","3d",left_thumb,True),
                                       right_thumb=Landmark("right_thumb","3d",right_thumb,True),
                                       left_hip=Landmark("left_hip","3d",left_hip,False),
                                       right_hip=Landmark("right_hip","3d",right_hip,False),
                                       left_knee=Landmark("left_knee","3d",left_knee,False),
                                       right_knee=Landmark("right_knee","3d",right_knee,False),
                                       left_ankle=Landmark("left_ankle","3d",left_ankle,False),
                                       right_ankle=Landmark("right_ankle","3d",right_ankle,False),
                                       left_heel=Landmark("left_heel","3d",left_heel,True),
                                       right_heel=Landmark("right_heel","3d",right_heel,True),
                                       left_foot_index=Landmark("left_foot_index","3d",left_foot_index,True),
                                       right_foot_index=Landmark("right_foot_index","3d",right_foot_index,True),
                                       jaw=Landmark("jaw","3d",jaw,False),
                                       chest=Landmark("chest","3d",chest,False),
                                       spine=Landmark("spine","3d",spine,False),
                                       hips=Landmark("hips","3d",hips,False),
                                       # 2D Keypoints
                                       bodyLandmarks2d=bodyLandmarks2d
                                       )
        return body_landmarks