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
Expose: EXpressive POse and Shape rEgression 

https://github.com/vchoutas/expose
"""

import logging
import torch
import os

import numpy as np
from common_pose.utils import *

from models.predictors_3d.expose_lib.expose.data.utils.bbox import bbox_to_center_scale
from models.HPE3D import HPE3D
from common_pose.BodyLandmarks import BodyLandmarks3d, Landmark
from models.predictors_3d.expose_lib.expose.models.smplx_net import SMPLXNet
from models.predictors_3d.expose_lib.expose.config import cfg as cfgreal
from models.predictors_3d.expose_lib.expose.config.cmd_parser import set_face_contour
from models.predictors_3d.expose_lib.expose.data.targets import BoundingBox
from models.predictors_3d.expose_lib.expose.data.transforms import build_transforms
from models.predictors_3d.expose_lib.expose.utils.checkpointer import Checkpointer
from models.predictors_3d.expose_lib.expose.utils.plot_utils import HDRenderer

import PIL.Image as pil_img
from expose.data.targets.image_list import to_image_list


import open3d as o3d
import cv2

from models.predictors_3d.expose_lib.expose_utils import save_expose_outputs


EXT = ['.jpg', '.jpeg', '.png']


class ExPose(HPE3D):
    
    def __init__(self, window_length, video_filename, developer_parameters, render=True, save_vis=True, save_mesh=True, save_params=True):
        """_summary_

        Args:
            window_length (_type_): lenght of the window
            video_filename (_type_): input video path
            render (bool, optional): Render the overlays. Defaults to True.
            save_vis (bool, optional): Save the visualization. Defaults to True.
            save_mesh (bool, optional):  Store the mesh predicted by the body-crop network. Defaults to True.
            save_params (bool, optional): Save params in a file. Defaults to True.
        """
        super().__init__(window_length)  
        #load ExPose model
        
        self.video_filename = video_filename
        self.device = torch.device('cuda')
        if not torch.cuda.is_available():
            logging.error('CUDA is not available! ExPose cannot be used.')
            sys.exit(3)
        
        if developer_parameters is None:
            exp_cfg = 'models/data/conf.yaml'
        else:
            exp_cfg = os.path.join(developer_parameters["models_path"], "data", "conf.yaml")
        
        
        cfg= cfgreal.clone()
        
        cfg.merge_from_file(exp_cfg)
        cfg.datasets.body.batch_size = 1

        cfg.is_training = False
        cfg.datasets.body.splits.test = "openpose"
        use_face_contour = cfg.datasets.use_face_contour
        set_face_contour(cfg, use_face_contour=use_face_contour)
        
        self.model = SMPLXNet(cfg)
        
        try:
            self.model = self.model.to(device=self.device)
        except RuntimeError:
            logging.error('CUDA is not available! ExPose cannot be used.')
            sys.exit(3)
            

        output_folder = cfg.output_folder
        checkpoint_folder = os.path.join(output_folder, cfg.checkpoint_folder)
        checkpoint_folder = 'models/data/checkpoints' #Path to checkpoint folder
        checkpointer = Checkpointer(
            self.model, save_dir=checkpoint_folder, pretrained=cfg.pretrained)

        arguments = {'iteration': 0, 'epoch_number': 0}
        extra_checkpoint_data = checkpointer.load_checkpoint()
        for key in arguments:
            if key in extra_checkpoint_data:
                arguments[key] = extra_checkpoint_data[key]
            
        self.model = self.model.eval()
        
        self.means = np.array(cfg.datasets.body.transforms.mean)
        self.std = np.array(cfg.datasets.body.transforms.std)
        body_crop_size = cfg.get('datasets', {}).get('body', {}).get(
        'transforms').get('crop_size', 256)
        
        self.hd_renderer = HDRenderer(img_size=body_crop_size)
        self.hd_renderer.renderer.delete()              
        self.hd_renderer = HDRenderer(img_size=body_crop_size)
        
        dataset_cfg = cfg.get('datasets', {})
        body_dsets_cfg = dataset_cfg.get('body', {})
        body_transfs_cfg = body_dsets_cfg.get('transforms', {})
        transforms = build_transforms(body_transfs_cfg, is_train=False)

        self.transforms = transforms
        
        self.render=render
        self.save_vis=save_vis
        self.save_mesh=save_mesh
        self.save_params=save_params
        
     
        
    def translate_keypoints_2d(self, keypoints_2d):
        return None
    
    @torch.no_grad()
    def get_3d_keypoints(self, img_w, img_h, bodyLandMarks2d, timestamp, input_2D_no, frame, bboxes):
        assert len(bboxes) == 1, f"Expose only works with window time = 1"
        frameoriginal = frame.copy()
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB
        img = img/255.0
        img = img.astype(np.float32)          

        
        bbox = np.array(bboxes[0], dtype=np.float32)

        target = BoundingBox(bbox, size=frameoriginal.shape)

        center, scale, bbox_size = bbox_to_center_scale(
            bbox, dset_scale_factor=1.2)
        target.add_field('bbox_size', bbox_size)
        target.add_field('orig_bbox_size', bbox_size)
        target.add_field('orig_center', center)
        target.add_field('center', center)
        target.add_field('scale', scale)

        target.add_field('fname', f'timestamp_{timestamp}')
        
        if self.transforms is not None:
            full_img, cropped_image, targets = self.transforms(img, target)

        full_img = to_image_list([full_img])
        cropped_image = torch.unsqueeze(cropped_image, 0)
        cropped_image = cropped_image.to(device= self.device)
        body_targets = [targets.to(self.device)]

        full_img = full_img.to(device=self.device)
      
        self.model = self.model.eval()

        model_output = self.model(cropped_image, body_targets, full_imgs=full_img,
                             device= self.device)
        
        stage_n_out = model_output.get('body', {}).get('final', {})
              
        pose= stage_n_out['joints']
        
        #-------- 3D pose --------
        
        save_expose_outputs(full_img, cropped_image, model_output, self, body_targets)
        
         #-------------- END 3D pose --------------------
        #Clean memory
        frame=None
        model_output= None
        targets= None
        target= None
        stage_n_out= None
        torch.cuda.empty_cache()
       
        if torch.cuda.is_available():
            # Get the current CUDA device
            device = torch.cuda.current_device()

            # Get the total memory available on the current device
            total_memory = torch.cuda.get_device_properties(device).total_memory

            # Get the current memory allocated on the device
            current_memory = torch.cuda.memory_allocated(device)
            maxcurrent=torch.cuda.max_memory_allocated()

            # Convert the memory values to human-readable format
            total_memory_gb = total_memory / (1024 ** 3)
            current_memory_gb = current_memory / (1024 ** 3)
            maxcurrent=maxcurrent / (1024 ** 3)

            logging.info(f"Total GPU Memory: {total_memory_gb:.2f} GB")
            logging.info(f"Current GPU Memory Usage: {current_memory_gb:.2f} GB")
            logging.info(f"Max GPU Memory Allocated: {maxcurrent:.2f} GB")
        else:
            logging.info("GPU is not available.")

        
        return self.pose_to_landmarks(pose, bodyLandMarks2d, timestamp)
    
    def pose_to_landmarks(self, pose, Landmarks2d, timestamp):
        
        KEYPOINT_NAMES = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle',
            'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplx',
            'right_eye_smplx', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3',
            'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2',
            'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3',
            'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1',
            'right_thumb2', 'right_thumb3', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe',
            'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_thumb', 'left_index',
            'left_middle', 'left_ring', 'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky',
            'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5',
            'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4',
            'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2',
            'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1',
            'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3',
            'left_mouth_2', 'left_mouth_1', 'left_mouth_5', 'left_mouth_4', 'mouth_bottom', 'right_mouth_4', 'right_mouth_5',
            'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3',
            'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6',
            'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6',
            'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1'
        ]
        
        num_joints = 144 # Get the number of joints
        joint_to_keypoint = {}  # Create an empty dictionary for mapping joints to keypoints
        
        data= pose
    
        num_joints = 144
        joint_to_keypoint = {}

        for i in range(num_joints):
            if i < len(KEYPOINT_NAMES):
                keypoint_name = KEYPOINT_NAMES[i]
                joint_to_keypoint[keypoint_name] = data[0, i].tolist()

            
        raw= data
       
        nose= joint_to_keypoint['nose']
        left_eye_inner= joint_to_keypoint['left_eye2']
        left_eye= joint_to_keypoint['left_eye']
        left_eye_outer= joint_to_keypoint['left_eye3']
        right_eye_inner= joint_to_keypoint['right_eye2']
        right_eye= joint_to_keypoint['right_eye']
        right_eye_outer= joint_to_keypoint['right_eye3']
        left_ear= joint_to_keypoint['left_ear']
        right_ear= joint_to_keypoint['right_ear']
        mouth_left= joint_to_keypoint['left_mouth_3']
        mouth_right= joint_to_keypoint['right_mouth_3']
        left_shoulder= joint_to_keypoint['left_shoulder']
        right_shoulder= joint_to_keypoint['right_shoulder']
        left_elbow= joint_to_keypoint['left_elbow']
        right_elbow= joint_to_keypoint['right_elbow']
        left_wrist= joint_to_keypoint['left_wrist']
        right_wrist= joint_to_keypoint['right_wrist']
        left_pinky= joint_to_keypoint['left_pinky']
        right_pinky= joint_to_keypoint['right_pinky']
        left_index= joint_to_keypoint['left_index']
        right_index= joint_to_keypoint['right_index']
        left_thumb= joint_to_keypoint['left_thumb']
        right_thumb= joint_to_keypoint['right_thumb']
        left_hip= joint_to_keypoint['left_hip']
        right_hip= joint_to_keypoint['right_hip']
        left_knee= joint_to_keypoint['left_knee']
        right_knee= joint_to_keypoint['right_knee']
        left_ankle= joint_to_keypoint['left_ankle']
        right_ankle= joint_to_keypoint['right_ankle']
        left_heel= joint_to_keypoint['left_heel']
        right_heel= joint_to_keypoint['right_heel']
        left_foot_index= joint_to_keypoint['left_foot']
        right_foot_index= joint_to_keypoint['right_foot']
        jaw= joint_to_keypoint['jaw']
        chest= joint_to_keypoint['spine3']
        spine= joint_to_keypoint['spine1']
        hips= joint_to_keypoint['pelvis']    
       
        
        
        body_landmarks3d = BodyLandmarks3d(raw=raw,timestamp=timestamp,
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
                                            bodyLandmarks2d= Landmarks2d
                                            )
                                           
        
        return body_landmarks3d