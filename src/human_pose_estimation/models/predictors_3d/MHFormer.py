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
MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation [CVPR 2022]

https://github.com/Vegetebird/MHFormer
"""
from models.HPE3D import HPE3D
import numpy as np
from common_pose.utils import normalize_screen_coordinates
import torch
import copy
import os
from models.predictors_3d.mhformer.mhformer_model import Model
from common_pose.BodyLandmarks import BodyLandmarks3d, Landmark

class MHFormer(HPE3D):
    def __init__(self, window_length=(243)):
        # Load MHFormer model
        args_3d = {
            'layers' : 3,
            'channel' : 512,
            'd_hid' : 1024,
            'frames' : window_length,
            'n_joints' : 17,
            'out_joints' : 17
        }
        args_3d['pad'] = (args_3d['frames'] - 1) // 2
        model_path = os.path.join('models', f"mhformer_model_{window_length}.pth")
       
        self.model = Model(args_3d).cuda()
        model_dict = self.model.state_dict()
        pre_dict = torch.load(model_path)
        for name, _ in model_dict.items():
            model_dict[name] = pre_dict[name]
        self.model.load_state_dict(model_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        self._args_3d = args_3d

        self._buffer_frames = [] # buffer of positions, since it predicts from video
        self._buffer_keypoints_2d = []
        self._buffer_timestamps = []
        self._buffer_bodyLandmarks2d = []

    def buffer_full(self):
        return self._args_3d['frames'] == len(self._buffer_keypoints_2d)
    
    def buffer_empty(self):
        return self._buffer_keypoints_2d == []

    def translate_keypoints_2d(self, keypoints_2d):
        if keypoints_2d is None:
            return None
        frame_keypoints_2d = [  # Order matters
            keypoints_2d._hips.coordinates, # 0
            keypoints_2d._right_hip.coordinates, # 1
            keypoints_2d._right_knee.coordinates, # 2
            keypoints_2d._right_ankle.coordinates, # 3
            keypoints_2d._left_hip.coordinates, # 4
            keypoints_2d._left_knee.coordinates, # 5
            keypoints_2d._left_ankle.coordinates, # 6
            keypoints_2d._spine.coordinates, # 7
            keypoints_2d._chest.coordinates, # 8
            keypoints_2d._jaw.coordinates, # 9
            keypoints_2d._nose.coordinates, # 10
            keypoints_2d._left_shoulder.coordinates, # 11
            keypoints_2d._left_elbow.coordinates, # 12
            keypoints_2d._left_wrist.coordinates, # 13
            keypoints_2d._right_shoulder.coordinates, # 14
            keypoints_2d._right_elbow.coordinates, # 15
            keypoints_2d._right_wrist.coordinates, # 16
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
            for _ in range(self._args_3d['frames'] // 2): # TODO is -1?
                self.add_frame(frame, keypoints_2d, timestamp, bodyLandmarks2d)

        return self.add_frame(frame, keypoints_2d, timestamp, bodyLandmarks2d)
    
    def destroy_buffer(self):
        if self.buffer_empty():
            return None
        while not self.buffer_full():
            self.add_frame(self._buffer_frames[0], self._buffer_keypoints_2d[0], self._buffer_timestamps[0], self._buffer_bodyLandmarks2d[0])
        half_window = (self._args_3d['frames'] // 2) + 2
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
        # Frame to predict 
        frame_no = self._args_3d['frames'] // 2
        img = self._buffer_frames[frame_no]
        timestamp = self._buffer_timestamps[frame_no]
        bodyLandmarks2d = self._buffer_bodyLandmarks2d[frame_no]

        img_size = img.shape

        input_2D_no = np.array(self._buffer_keypoints_2d)

        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  
        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        if torch.cuda.is_available():
            input_2D = input_2D.cuda()

        ## estimation
        output_3D_non_flip = self.model(input_2D[:, 0])
        output_3D_flip     = self.model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D = output_3D[0:, self._args_3d['pad']].unsqueeze(1) 
        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()
        return self.pose_to_landmarks(post_out, bodyLandmarks2d, timestamp)
    

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
        