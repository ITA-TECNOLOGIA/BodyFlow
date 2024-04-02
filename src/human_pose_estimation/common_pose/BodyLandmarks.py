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

import numpy as np
import cv2
from common_pose.utils import camera_to_world

class Landmark:
    def __init__(self,
                 name : str,
                 type : str, # ["2d", "3d"]
                 coordinates : np.array,
                 hallucinated : bool
                 ):

        if type not in ["2d", "3d"]:
            raise NotImplementedError(f"Type {type} not valid")

        if (type == "2d" and len(coordinates) != 2) or (type == "3d" and len(coordinates) != 3):
            raise NotImplementedError("Number of coordinates does not match the type")
        
        self.name = name
        self.type = type
        self.coordinates = coordinates
        self.hallucinated = hallucinated

    def get_msg(self):
        msg = {
            'name' : self.name,
            'type' : self.type,
            'hallucinated' : self.hallucinated
        }
        if self.type == "2d":
            msg['coordinate_x'] = self.coordinates[0]
            msg['coordinate_y'] = self.coordinates[1]
        elif self.type == "3d":
            msg['coordinate_x'] = self.coordinates[0]
            msg['coordinate_y'] = self.coordinates[1]
            msg['coordinate_z'] = self.coordinates[2]
        return msg

class BodyLandmarks2d:
    def __init__(self,
                 nose : Landmark,
                 left_eye_inner : Landmark, 
                 left_eye : Landmark,
                 left_eye_outer : Landmark,
                 right_eye_inner : Landmark,
                 right_eye : Landmark,
                 right_eye_outer : Landmark,
                 left_ear : Landmark,
                 right_ear : Landmark,
                 mouth_left : Landmark,
                 mouth_right : Landmark,
                 left_shoulder : Landmark,
                 right_shoulder : Landmark,
                 left_elbow : Landmark,
                 right_elbow : Landmark,
                 left_wrist : Landmark,
                 right_wrist : Landmark,
                 left_pinky : Landmark,
                 right_pinky : Landmark,
                 left_index : Landmark,
                 right_index : Landmark,
                 left_thumb : Landmark,
                 right_thumb : Landmark,
                 left_hip : Landmark,
                 right_hip : Landmark,
                 left_knee : Landmark,
                 right_knee : Landmark,
                 left_ankle : Landmark,
                 right_ankle : Landmark,
                 left_heel : Landmark,
                 right_heel : Landmark,
                 left_foot_index : Landmark,
                 right_foot_index : Landmark,
                 jaw : Landmark,
                 chest : Landmark,
                 spine : Landmark,
                 hips : Landmark):

        self._nose = nose
        self._left_eye_inner = left_eye_inner
        self._left_eye = left_eye
        self._left_eye_outer = left_eye_outer
        self._right_eye_inner = right_eye_inner
        self._right_eye = right_eye
        self._right_eye_outer = right_eye_outer
        self._left_ear = left_ear
        self._right_ear = right_ear
        self._mouth_left = mouth_left
        self._mouth_right = mouth_right
        self._left_shoulder = left_shoulder
        self._right_shoulder = right_shoulder
        self._left_elbow = left_elbow
        self._right_elbow = right_elbow
        self._left_wrist = left_wrist
        self._right_wrist = right_wrist
        self._left_pinky = left_pinky
        self._right_pinky = right_pinky
        self._left_index = left_index
        self._right_index = right_index
        self._left_thumb = left_thumb
        self._right_thumb = right_thumb
        self._left_hip = left_hip
        self._right_hip = right_hip
        self._left_knee = left_knee
        self._right_knee = right_knee
        self._left_ankle = left_ankle
        self._right_ankle = right_ankle
        self._left_heel = left_heel
        self._right_heel = right_heel
        self._left_foot_index = left_foot_index
        self._right_foot_index = right_foot_index
        self._jaw = jaw
        self._chest = chest
        self._spine = spine
        self._hips = hips
    # Test that all parameters are 2d

    def get_msg(self) -> dict:
            """
            Returns a dict which contains all the stored landmarks.
            """
            msg = {
                "nose" : self._nose.get_msg(),
                "left_eye_inner" : self._left_eye_inner.get_msg(),
                "left_eye" : self._left_eye.get_msg(),
                "left_eye_outer" : self._left_eye_outer.get_msg(),
                "right_eye_inner" : self._right_eye_inner.get_msg(),
                "right_eye" : self._right_eye.get_msg(),
                "right_eye_outer" : self._right_eye_outer.get_msg(),
                "left_ear" : self._left_ear.get_msg(),
                "right_ear" : self._right_ear.get_msg(),
                "mouth_left" : self._mouth_left.get_msg(),
                "mouth_right" : self._mouth_right.get_msg(),
                "left_shoulder" : self._left_shoulder.get_msg(),
                "right_shoulder" : self._right_shoulder.get_msg(),
                "left_elbow" : self._left_elbow.get_msg(),
                "right_elbow" : self._right_elbow.get_msg(),
                "left_wrist" : self._left_wrist.get_msg(),
                "right_wrist" : self._right_wrist.get_msg(),
                "left_pinky" : self._left_pinky.get_msg(),
                "right_pinky" : self._right_pinky.get_msg(),
                "left_index" : self._left_index.get_msg(),
                "right_index" : self._right_index.get_msg(),
                "left_thumb" : self._left_thumb.get_msg(),
                "right_thumb" : self._right_thumb.get_msg(),
                "left_hip" : self._left_hip.get_msg(),
                "right_hip" : self._right_hip.get_msg(),
                "left_knee" : self._left_knee.get_msg(),
                "right_knee" : self._right_knee.get_msg(),
                "left_ankle" : self._left_ankle.get_msg(),
                "right_ankle" : self._right_ankle.get_msg(),
                "left_heel" : self._left_heel.get_msg(),
                "right_heel" : self._right_heel.get_msg(),
                "left_foot_index" : self._left_foot_index.get_msg(),
                "right_foot_index" : self._right_foot_index.get_msg(),
                "jaw" : self._jaw.get_msg(),
                "chest" : self._chest.get_msg(),
                "spine" : self._spine.get_msg(),
                "hips" : self._hips.get_msg()
            }
            return msg      

class BodyLandmarks3d:
    """
    This class represents all the possible body landmarks. If a predictor does not predicts
    all of these body landmarks, the developer will manually extrapolate the computed landmarks
    to the unknown ones, e.g., the left_eye position could be the same as the nose.
    """
    def __init__(self,
                raw,
                timestamp,
                nose : Landmark,
                left_eye_inner : Landmark, 
                left_eye : Landmark,
                left_eye_outer : Landmark,
                right_eye_inner : Landmark,
                right_eye : Landmark,
                right_eye_outer : Landmark,
                left_ear : Landmark,
                right_ear : Landmark,
                mouth_left : Landmark,
                mouth_right : Landmark,
                left_shoulder : Landmark,
                right_shoulder : Landmark,
                left_elbow : Landmark,
                right_elbow : Landmark,
                left_wrist : Landmark,
                right_wrist : Landmark,
                left_pinky : Landmark,
                right_pinky : Landmark,
                left_index : Landmark,
                right_index : Landmark,
                left_thumb : Landmark,
                right_thumb : Landmark,
                left_hip : Landmark,
                right_hip : Landmark,
                left_knee : Landmark,
                right_knee : Landmark,
                left_ankle : Landmark,
                right_ankle : Landmark,
                left_heel : Landmark,
                right_heel : Landmark,
                left_foot_index : Landmark,
                right_foot_index : Landmark,
                jaw : Landmark,
                chest : Landmark,
                spine : Landmark,
                hips : Landmark,

                # 2D keypoints, by default None if that particular 2D pose estimator does not compute it
                bodyLandmarks2d : BodyLandmarks2d
                ) -> None:

        self._raw = raw
        self.timestamp = timestamp
        self._nose = nose
        self._left_eye_inner = left_eye_inner
        self._left_eye = left_eye
        self._left_eye_outer = left_eye_outer
        self._right_eye_inner = right_eye_inner
        self._right_eye = right_eye
        self._right_eye_outer = right_eye_outer
        self._left_ear = left_ear
        self._right_ear = right_ear
        self._mouth_left = mouth_left
        self._mouth_right = mouth_right
        self._left_shoulder = left_shoulder
        self._right_shoulder = right_shoulder
        self._left_elbow = left_elbow
        self._right_elbow = right_elbow
        self._left_wrist = left_wrist
        self._right_wrist = right_wrist
        self._left_pinky = left_pinky
        self._right_pinky = right_pinky
        self._left_index = left_index
        self._right_index = right_index
        self._left_thumb = left_thumb
        self._right_thumb = right_thumb
        self._left_hip = left_hip
        self._right_hip = right_hip
        self._left_knee = left_knee
        self._right_knee = right_knee
        self._left_ankle = left_ankle
        self._right_ankle = right_ankle
        self._left_heel = left_heel
        self._right_heel = right_heel
        self._left_foot_index = left_foot_index
        self._right_foot_index = right_foot_index
        self._jaw = jaw
        self._chest = chest
        self._spine = spine
        self._hips = hips

        # 2D keypoints
        self._bodyLandmarks2d = bodyLandmarks2d

    def get_msg(self) -> dict:
        """
        Returns a dict which contains all the stored landmarks.
        """
        msg = {
            "nose" : self._nose.get_msg(),
            "left_eye_inner" : self._left_eye_inner.get_msg(),
            "left_eye" : self._left_eye.get_msg(),
            "left_eye_outer" : self._left_eye_outer.get_msg(),
            "right_eye_inner" : self._right_eye_inner.get_msg(),
            "right_eye" : self._right_eye.get_msg(),
            "right_eye_outer" : self._right_eye_outer.get_msg(),
            "left_ear" : self._left_ear.get_msg(),
            "right_ear" : self._right_ear.get_msg(),
            "mouth_left" : self._mouth_left.get_msg(),
            "mouth_right" : self._mouth_right.get_msg(),
            "left_shoulder" : self._left_shoulder.get_msg(),
            "right_shoulder" : self._right_shoulder.get_msg(),
            "left_elbow" : self._left_elbow.get_msg(),
            "right_elbow" : self._right_elbow.get_msg(),
            "left_wrist" : self._left_wrist.get_msg(),
            "right_wrist" : self._right_wrist.get_msg(),
            "left_pinky" : self._left_pinky.get_msg(),
            "right_pinky" : self._right_pinky.get_msg(),
            "left_index" : self._left_index.get_msg(),
            "right_index" : self._right_index.get_msg(),
            "left_thumb" : self._left_thumb.get_msg(),
            "right_thumb" : self._right_thumb.get_msg(),
            "left_hip" : self._left_hip.get_msg(),
            "right_hip" : self._right_hip.get_msg(),
            "left_knee" : self._left_knee.get_msg(),
            "right_knee" : self._right_knee.get_msg(),
            "left_ankle" : self._left_ankle.get_msg(),
            "right_ankle" : self._right_ankle.get_msg(),
            "left_heel" : self._left_heel.get_msg(),
            "right_heel" : self._right_heel.get_msg(),
            "left_foot_index" : self._left_foot_index.get_msg(),
            "right_foot_index" : self._right_foot_index.get_msg(),
            "jaw" : self._jaw.get_msg(),
            "chest" : self._chest.get_msg(),
            "spine" : self._spine.get_msg(),
            "hips" : self._hips.get_msg(),
            "bodyLandmarks2d" : self._bodyLandmarks2d.get_msg()
        }
        return msg

    # https://github.com/Vegetebird/MHFormer/blob/main/demo/vis.py
    def show2Dpose(self, img):
        # TODO THIS ONLY WORKS WITH CPN
        kps = self._pose_2d
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

        lcolor = (255, 0, 0)
        rcolor = (0, 0, 255)
        thickness = 3

        for j,c in enumerate(connections):
            start = map(int, kps[c[0]])
            end = map(int, kps[c[1]])
            start = list(start)
            end = list(end)
            cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

        return img

    def dict_to_numpy(self):
        poses_in_list = [
            self._hips,
            self._right_hip,
            self._right_knee,
            self._right_ankle,
            self._left_hip,
            self._left_knee,
            self._left_ankle,
            self._spine,
            self._chest,
            self._jaw,
            self._nose,
            self._left_shoulder,
            self._left_elbow,
            self._left_wrist,
            self._right_shoulder,
            self._right_elbow,
            self._right_wrist
        ]
        return np.float32(np.array(poses_in_list))

    # https://github.com/Vegetebird/MHFormer/blob/main/demo/vis.py
    def show3Dpose(self, ax):
        ax.view_init(elev=15., azim=70)
        vals = self.dict_to_numpy()

        # TODO this only works with MHFORMER
        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        vals = camera_to_world(vals, R=rot, t=0)
        vals[:, 2] -= np.min(vals[:, 2])

        lcolor=(0,0,1)
        rcolor=(1,0,0)

        I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
        J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

        LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

        for i in np.arange( len(I) ):
            x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
            ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

        RADIUS = 0.72
        RADIUS_Z = 0.7

        xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
        ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
        ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
        ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
        #ax.set_aspect('equal') # works fine in matplotlib==2.2.2
        ax.set_box_aspect([1,1,1]) # https://github.com/fabro66/GAST-Net-3DPoseEstimation/issues/51

        white = (1.0, 1.0, 1.0, 0.0)
        ax.xaxis.set_pane_color(white) 
        ax.yaxis.set_pane_color(white)
        ax.zaxis.set_pane_color(white)

        ax.tick_params('x', labelbottom = False)
        ax.tick_params('y', labelleft = False)
        ax.tick_params('z', labelleft = False)
        


