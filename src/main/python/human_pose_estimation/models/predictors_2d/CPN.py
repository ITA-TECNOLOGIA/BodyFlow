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
Cascaded Pyramid Network (2D HPE) used in MHFormer

https://github.com/Vegetebird/MHFormer
"""
from human_pose_estimation.models.HPE2D import HPE2D
import numpy as np
import copy
import torch
import torch.backends.cudnn as cudnn
import os
import logging

from human_pose_estimation.models.predictors_2d.cpn.hrnet.lib.config import cfg, update_config
from human_pose_estimation.models.predictors_2d.cpn.hrnet.lib.models import pose_hrnet

from human_pose_estimation.models.predictors_2d.cpn.hrnet.lib.utils.utilitys import PreProcess
from human_pose_estimation.models.predictors_2d.cpn.hrnet.lib.utils.inference import get_final_preds
from human_pose_estimation.common_pose.BodyLandmarks import Landmark, BodyLandmarks2d

from human_pose_estimation.models.predictors_2d.cpn.preprocess import h36m_coco_format
import sys
import pkg_resources
import human_pose_estimation
import inspect


# Cascaded Pyramid Network used in MHFormer (Yolo + HRNet)
# https://github.com/Vegetebird/MHFormer
class CPN(HPE2D):
    def __init__(self, developer_parameters):
        self._det_dim = 416
        self._thread_score = 0.30
        self._nms_thresh = 0.4 # NMS Threshold
        self._num_person = 1 # Starting id is 1

        # --------- Load Pose Model (HRNet) ---------
        #cfg_file = os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_2d', 'cpn', 'hrnet', 'experiments','w48_384x288_adam_lr1e-3.yaml')
        
        archivo_fuente  = inspect.getfile(human_pose_estimation)
        library_path = os.path.dirname(archivo_fuente)
        logging.info(f"Path de la librería:{ library_path}")
        
               
   
        cfg_file = os.path.join(library_path,'models','predictors_2d','cpn', 'hrnet', 'experiments','w48_384x288_adam_lr1e-3.yaml')
        if developer_parameters is None:
            model_dir = os.path.join('/models', 'pose_hrnet_w48_384x288.pth')
        else:
            model_dir = os.path.join(developer_parameters['models_path'], 'pose_hrnet_w48_384x288.pth')
        self._pose_model = pose_model_load(cfg_file, model_dir)

    def get_frame_keypoints(self, frame_full, cropped_frame, bbox):
        
        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame_full, [bbox], cfg, self._num_person)

            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = self._pose_model(inputs)

            # compute coordinate
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        kpts = np.zeros((self._num_person, 17, 2), dtype=np.float32)
        scores = np.zeros((self._num_person, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        keypoints = np.array([kpts])
        scores = np.array([scores])

        keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
        scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

        # Convert to coco format
        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        keypoints = np.squeeze(keypoints)
        # Substract offset
        keypoints[:, 0] -= bbox[0]
        keypoints[:, 1] -= bbox[1]

        # Keypoints order???
        frame_keypoints_2d = BodyLandmarks2d(
            nose = Landmark("nose", "2d", keypoints[10], False, bbox),
            left_eye_inner = Landmark("left_eye_inner", "2d", keypoints[10], True, bbox),
            left_eye = Landmark("left_eye", "2d", keypoints[10], True, bbox),
            left_eye_outer = Landmark("left_eye_outer", "2d", keypoints[10], True, bbox),
            right_eye_inner = Landmark("right_eye_inner", "2d", keypoints[10], True, bbox),
            right_eye = Landmark("right_eye", "2d", keypoints[10], True, bbox),
            right_eye_outer = Landmark("right_eye_outer", "2d", keypoints[10], True, bbox),
            left_ear = Landmark("left_ear", "2d", keypoints[10], True, bbox),
            right_ear = Landmark("right_ear", "2d", keypoints[10], True, bbox),
            mouth_left = Landmark("mouth_left", "2d", keypoints[10], True, bbox),
            mouth_right = Landmark("mouth_right", "2d", keypoints[10], True),
            left_shoulder = Landmark("left_shoulder", "2d", keypoints[11], False, bbox),
            right_shoulder = Landmark("right_shoulder", "2d", keypoints[14], False, bbox),
            left_elbow = Landmark("left_elbow", "2d", keypoints[12], False, bbox),
            right_elbow = Landmark("right_elbow", "2d", keypoints[15], False, bbox),
            left_wrist = Landmark("left_wrist", "2d", keypoints[13], False, bbox),
            right_wrist = Landmark("right_wrist", "2d", keypoints[16], False, bbox),
            left_pinky = Landmark("left_pinky", "2d", keypoints[13], True, bbox),
            right_pinky = Landmark("right_pinky", "2d", keypoints[16], True, bbox),
            left_index = Landmark("left_index", "2d", keypoints[13], True, bbox),
            right_index = Landmark("right_index", "2d", keypoints[16], True, bbox),
            left_thumb = Landmark("left_thumb", "2d", keypoints[13], True, bbox),
            right_thumb = Landmark("right_thumb", "2d", keypoints[16], True, bbox),
            left_hip = Landmark("left_hip", "2d", keypoints[4], False, bbox),
            right_hip = Landmark("right_hip", "2d", keypoints[1], False, bbox),
            left_knee = Landmark("left_knee", "2d", keypoints[5], False, bbox),
            right_knee = Landmark("right_knee", "2d", keypoints[2], False, bbox),
            left_ankle = Landmark("left_ankle", "2d", keypoints[6], False, bbox),
            right_ankle = Landmark("right_ankle", "2d", keypoints[3], False, bbox),
            left_heel = Landmark("left_heel", "2d", keypoints[6], True, bbox),
            right_heel = Landmark("right_heel", "2d", keypoints[3], True, bbox),
            left_foot_index = Landmark("left_foot_index", "2d", keypoints[6], True, bbox),
            right_foot_index = Landmark("right_foot_index", "2d", keypoints[3], True, bbox),
            jaw = Landmark("jaw", "2d", keypoints[9], False, bbox),
            chest = Landmark("chest", "2d", keypoints[8], False, bbox),
            spine = Landmark("spine", "2d", keypoints[7], False, bbox),
            hips = Landmark("hips", "2d", keypoints[0], False, bbox),
            bbox = bbox
        )

        return frame_keypoints_2d


def pose_model_load(cfg_file, model_dir):
    update_config(cfg, cfg_file, model_dir)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = pose_hrnet.get_pose_net(cfg, is_train=False)
    logging.info(f"CPN pose estimator 2d loaded in in GPU: %r", torch.cuda.is_available())
    if torch.cuda.is_available():
        model = model.cuda()
        state_dict = torch.load(cfg.OUTPUT_DIR)
    else:
        state_dict = torch.load(cfg.OUTPUT_DIR, map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    logging.debug('CPN network successfully loaded')
    
    return model