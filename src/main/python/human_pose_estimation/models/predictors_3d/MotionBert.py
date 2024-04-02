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
MotionBERT: Unified Pretraining for Human Motion Analysis

https://github.com/Walter0807/MotionBERT
"""
from human_pose_estimation.models.HPE3D import HPE3D
import os
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import logging
from functools import partial
from human_pose_estimation.common_pose.utils import normalize_screen_coordinates_bboxes
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks3d, Landmark
from human_pose_estimation.models.predictors_3d.motionbert.DSTformer import DSTformer

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
import logging

class MotionBert(HPE3D):
    """
    This class imports and executes the model called MotionBert. The model was downloaded from the 
    source. It works with a receptive field of 243 frames.
    """
    def __init__(self, window_length, developer_parameters):
        super().__init__(window_length)
        # Load MotionBert model with MotionBert arguments for network inference
        # self.model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, 
        #                        depth=5, num_heads=8, mlp_ratio=4, 
        #                        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        #                        maxlen=self.window_length, num_joints=17)

        self.model = DSTformer(dim_in = 3, dim_out = 3, dim_feat = 512, dim_rep =512, 
                               depth = 5, num_heads = 8, mlp_ratio = 2, 
                               norm_layer = partial(nn.LayerNorm, eps=1e-6), 
                               maxlen=self.window_length, num_joints = 17, att_fuse = True)
        model_params = 0
        for parameter in self.model.parameters():
            model_params = model_params + parameter.numel()
        logging.info(f'Trainable parameter count: {model_params}')
        self.model= nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model =  self.model.cuda()
        
        if developer_parameters is None:
            chk_filename='/models/best_epoch.bin'
        else:
            chk_filename  =os.path.join("models", "best_epoch.bin")
        logging.info(f'Loading checkpoint ´{chk_filename}')
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['model_pos'], strict=True)
        
        logging.info('Testing')
        self.model.eval()         
        self._args_3d = {'frames': 243}

        self._buffer_frames = [] # buffer of positions, since it predicts from video
        self._buffer_keypoints_2d = []
        self._buffer_timestamps = []       
        self._buffer_bodyLandmarks2d = []

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

    def get_3d_keypoints(self, img_w, img_h, bodyLandmarks2d, timestamp, input_2D_no, frame, bboxes):

        # scale = min(img_w,img_h) / 2.0
        # kpts_all = copy.deepcopy(input_2D_no)
        # kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([img_w, img_h]) / 2.0
        # kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        # input_2D = normalize_screen_coordinates_bboxes(input_2D_no, bboxes)  
        # input_2D = copy.deepcopy(kpts_all)
        # input_2D_aug = copy.deepcopy(kpts_all)
        
        input_2D = normalize_screen_coordinates_bboxes(input_2D_no, bboxes)  
        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        test_confidence = np.ones(input_2D.astype('float32').shape)[:,:,:,:,0:1] #MOTIONBERT CUSTOM ( we need confidence from 2d inference )
        input_2D = np.concatenate((input_2D.astype('float32'), test_confidence), axis=4)  # [BS, N, 17, 3]        
        
        input_2D = torch.from_numpy(input_2D.astype('float32'))
        
        if torch.cuda.is_available():
            input_2D = input_2D.cuda()

        ## estimation
        output_3D_non_flip = self.model(input_2D[:, 0])
        output_3D_flip     = self.model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2
      
        output_3D = output_3D[0:,(self._args_3d['frames'] - 1) // 2].unsqueeze(1) 
        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()
        
        
        # Convert to pixel coordinates
        # post_out = post_out * (min(img_w, img_h) / 2.0)
        # post_out[:,:2] = post_out[:,:2] + np.array([img_w, img_h]) / 2.0        
        
        return self.pose_to_landmarks(post_out,  bodyLandmarks2d, timestamp)
    

    def pose_to_landmarks(self, pose, bodyLandmarks2d, timestamp):
        raw = pose
        
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
        