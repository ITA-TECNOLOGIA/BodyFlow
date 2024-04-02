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

import unittest
import os
import sys
import numpy as np
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation'))  # To avoid importing problems

from common_pose.BodyLandmarks import BodyLandmarks2d

# Pose Predictors from RGB to 2D
from models.HPE2D import HPE2D
from models.predictors_2d.Dummy2D import Dummy2D
from models.predictors_2d.Mediapipe2D import Mediapipe2D
from models.predictors_2d.CPN import CPN
from models.predictors_2d.Lightweight2D import Lightweight2D

from models.person_detector.YoloV3 import YoloV3
from models.tracking.Single import Single

# Video Capture
from video_capture.VideoFromImages import VideoFromImages


class TestTwoDimensionsPredictor(unittest.TestCase):
    """
    Tests to check that the 2D Human Pose Estimators work properly
    """

    def setUp(self) -> None:
        self._gpu = '0' # Id of the GPU to check
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu

        # Defining input
        images_path = os.path.join("data", "demos", "single_image")
        self._videoCapture = VideoFromImages(images_path, infinite_loop=True)

        # Defining person detector
        self._person_detector = YoloV3()
        # Defining tracker
        self._tracking = Single(max_age=10)
        

    def _get_keypoints_test(self, predictor: HPE2D):
        frame, _ = self._videoCapture.get_frame()
        img_size = frame.shape
        img_w = img_size[1]
        img_h = img_size[0]

        bboxs, scores = self._person_detector.predict(frame)
        bounding_boxes = self._tracking.update(bboxs, scores, frame)

        person_id, person_bbox = next(iter(bounding_boxes.items()))

        x1, y1, x2, y2 = person_bbox
        # May can be out of the image boundaries
        x1 = np.clip(x1, 0, img_w)
        x2 = np.clip(x2, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        y2 = np.clip(y2, 0, img_h)
        person_bbox = [x1, y1, x2, y2]
    
        cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]

        keypoints_2d = predictor.get_frame_keypoints(frame, cropped_frame, person_bbox) # Note that person bbox does not match with frame, it is used for offset keypoints 2d

        # Check all landmarks
        properties = ['nose', 'left_eye_inner', 'left_eye', 
                     'left_eye_outer', 'right_eye_inner', 'right_eye',
                     'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
                     'mouth_right', 'left_shoulder', 'right_shoulder', 
                     'left_elbow', 'right_elbow', 'left_wrist', 
                     'right_wrist', 'left_pinky', 'right_pinky', 
                     'left_index', 'right_index', 'left_thumb', 
                     'right_thumb', 'left_hip', 'right_hip', 'left_knee', 
                     'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 
                     'right_heel', 'left_foot_index', 'right_foot_index', 
                     'hips', 'chest', 'spine', 'jaw', 'bbox'] # Added also bbox
        
        self.assertIsInstance(keypoints_2d, BodyLandmarks2d)
        self.assertEqual(len(properties), len(list(keypoints_2d.get_msg().keys())))
 
        # Check values greater than 0 (goes from 0, to width/height pixels)
        for landmark_name, landmark_info in keypoints_2d.get_msg().items():
            self.assertIn(landmark_name, properties)
            if landmark_name != 'bbox':
                self.assertEqual(landmark_name, landmark_info['name'])
                self.assertEqual(landmark_info['type'], "2d")
                self.assertGreaterEqual(landmark_info['coordinate_x'], 0)
                self.assertGreaterEqual(landmark_info['coordinate_y'], 0)
                self.assertNotIn('coordinate_z', landmark_info.keys())
    
class TestDummy(TestTwoDimensionsPredictor):
    def test_get_keypoints_dummy(self):
        super()._get_keypoints_test(Dummy2D())


class TestMediapipe(TestTwoDimensionsPredictor):
    def test_get_keypoints_mediapipe(self):
        super()._get_keypoints_test(Mediapipe2D())

class TestCPN(TestTwoDimensionsPredictor):
    def test_get_keypoints_cpn(self):
        super()._get_keypoints_test(CPN())

class TestLightweight(TestTwoDimensionsPredictor):
    def test_get_keypoints_lightweight(self):
        super()._get_keypoints_test(Lightweight2D())


if __name__ == '__main__':
    unittest.main()

