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
Cascaded Pyramid Network (2D HPE) used in MHFormer

https://github.com/Vegetebird/MHFormer
"""
from models.HPE2D import HPE2D
import numpy as np
import copy
import torch
import torch.backends.cudnn as cudnn
import os
import logging

from models.predictors_2d.cpn.yolov3.human_detector import load_model as yolo_model

from models.predictors_2d.cpn.hrnet.lib.config import cfg, update_config
from models.predictors_2d.cpn.hrnet.lib.models import pose_hrnet

from models.predictors_2d.cpn.yolov3.human_detector import yolo_human_det as yolo_det
from models.predictors_2d.cpn.hrnet.lib.utils.utilitys import PreProcess
from models.predictors_2d.cpn.hrnet.lib.utils.inference import get_final_preds
from common_pose.BodyLandmarks import Landmark, BodyLandmarks2d

from models.predictors_2d.cpn.preprocess import h36m_coco_format
from models.predictors_2d.cpn.sort import Sort


# Cascaded Pyramid Network used in MHFormer (Yolo + HRNet)
# https://github.com/Vegetebird/MHFormer
class CPN(HPE2D):
    def __init__(self):
        self._det_dim = 416
        self._thread_score = 0.30
        self._nms_thresh = 0.4 # NMS Threshold
        self._num_person = 1
        self._people_sort = Sort(min_hits=0)

        # --------- Load Person Detector (Yolo) ---------
        self._human_model = yolo_model(inp_dim=self._det_dim)

        # --------- Load Pose Model (HRNet) ---------
        cfg_file = os.path.join('src', 'pose-estimation', 'models', 'predictors_2d', 'cpn', 'hrnet', 'experiments','w48_384x288_adam_lr1e-3.yaml')
        model_dir = os.path.join('models', 'pose_hrnet_w48_384x288.pth')
        self._pose_model = pose_model_load(cfg_file, model_dir)

    def get_frame_keypoints(self, frame):
        bboxs, scores = yolo_det(frame, self._human_model, reso=self._det_dim, confidence=self._thread_score, nms_thresh=self._nms_thresh)

        if bboxs is None or not bboxs.any():
            logging.debug("No person detected!")
            return None
        else:
            bboxs_pre = copy.deepcopy(bboxs) 
            scores_pre = copy.deepcopy(scores) 

        # Using Sort to track people
        people_track = self._people_sort.update(bboxs)
        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-self._num_person:, :-1].reshape(self._num_person, 4)
            people_track_ = people_track_[::-1]
        else:
            return None

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, self._num_person)

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

        # Keypoints order???
        frame_keypoints_2d = BodyLandmarks2d(
            nose = Landmark("nose", "2d", keypoints[10], False),
            left_eye_inner = Landmark("left_eye_inner", "2d", keypoints[10], True),
            left_eye = Landmark("left_eye", "2d", keypoints[10], True),
            left_eye_outer = Landmark("left_eye_outer", "2d", keypoints[10], True),
            right_eye_inner = Landmark("right_eye_inner", "2d", keypoints[10], True),
            right_eye = Landmark("right_eye", "2d", keypoints[10], True),
            right_eye_outer = Landmark("right_eye_outer", "2d", keypoints[10], True),
            left_ear = Landmark("left_ear", "2d", keypoints[10], True),
            right_ear = Landmark("right_ear", "2d", keypoints[10], True),
            mouth_left = Landmark("mouth_left", "2d", keypoints[10], True),
            mouth_right = Landmark("mouth_right", "2d", keypoints[10], True),
            left_shoulder = Landmark("left_shoulder", "2d", keypoints[11], False),
            right_shoulder = Landmark("right_shoulder", "2d", keypoints[14], False),
            left_elbow = Landmark("left_elbow", "2d", keypoints[12], False),
            right_elbow = Landmark("right_elbow", "2d", keypoints[15], False),
            left_wrist = Landmark("left_wrist", "2d", keypoints[13], False),
            right_wrist = Landmark("right_wrist", "2d", keypoints[16], False),
            left_pinky = Landmark("left_pinky", "2d", keypoints[13], True),
            right_pinky = Landmark("right_pinky", "2d", keypoints[16], True),
            left_index = Landmark("left_index", "2d", keypoints[13], True),
            right_index = Landmark("right_index", "2d", keypoints[16], True),
            left_thumb = Landmark("left_thumb", "2d", keypoints[13], True),
            right_thumb = Landmark("right_thumb", "2d", keypoints[16], True),
            left_hip = Landmark("left_hip", "2d", keypoints[4], False),
            right_hip = Landmark("right_hip", "2d", keypoints[1], False),
            left_knee = Landmark("left_knee", "2d", keypoints[5], False),
            right_knee = Landmark("right_knee", "2d", keypoints[2], False),
            left_ankle = Landmark("left_ankle", "2d", keypoints[6], False),
            right_ankle = Landmark("right_ankle", "2d", keypoints[3], False),
            left_heel = Landmark("left_heel", "2d", keypoints[6], True),
            right_heel = Landmark("right_heel", "2d", keypoints[3], True),
            left_foot_index = Landmark("left_foot_index", "2d", keypoints[6], True),
            right_foot_index = Landmark("right_foot_index", "2d", keypoints[3], True),
            jaw = Landmark("jaw", "2d", keypoints[9], False),
            chest = Landmark("chest", "2d", keypoints[8], False),
            spine = Landmark("spine", "2d", keypoints[7], False),
            hips = Landmark("hips", "2d", keypoints[0], False),
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