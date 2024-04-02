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
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation'))  # To avoid importing problems
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'mixste'))
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'motionbert'))

from common_pose.BodyLandmarks import BodyLandmarks3d

# Video capture
from video_capture.VideoFromImages import VideoFromImages

# Person detector
from models.person_detector.YoloV3 import YoloV3

# Person tracker
from models.tracking.DeepSort import DeepSort
from models.tracking.Single import Single
from models.tracking.Sort import Sort

# Pose Predictors from RGB to 2D
from models.predictors_2d.Dummy2D import Dummy2D
from models.predictors_2d.Mediapipe2D import Mediapipe2D
from models.predictors_2d.CPN import CPN
from models.predictors_2d.Lightweight2D import Lightweight2D

# Pose Predictors from 2D to 3D
from models.predictors_3d.Dummy3D import Dummy3D
from models.predictors_3d.MHFormer import MHFormer
from models.predictors_3d.VideoPose3D import VideoPose3D
from models.predictors_3d.MixSTE import MixSTE
from models.predictors_3d.MotionBert import MotionBert

# Pose Predictor from RGB to 3D
from models.HPE import HPE

class TestPipeline(unittest.TestCase):
    """
    Tests to check that the pipeline is working properly.
    """
    def setUp(self) -> None:
        self._gpu = '0' # Id of the GPU to check
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu

        # Defining input
        images_path = os.path.join("data", "demos", "single_image")
        self._videoCapture = VideoFromImages(images_path, infinite_loop=True)

    def pipeline_test(self, human_pose_estimator: HPE):
        while True:
            frame, timestamp = self._videoCapture.get_frame()
            is_initialized = human_pose_estimator.init_buffers(frame, timestamp)
            if is_initialized:
                break
        
        for _ in range(20):  # Some frames to predict
            frame, timestamp = self._videoCapture.get_frame()
            body_landmarks = human_pose_estimator.predict_pose()
            self.assertEqual(type(body_landmarks[1]), BodyLandmarks3d)


            # Test pose
            landmarks = ['nose', 'left_eye_inner', 'left_eye', 
                        'left_eye_outer', 'right_eye_inner', 'right_eye',
                        'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
                        'mouth_right', 'left_shoulder', 'right_shoulder', 
                        'left_elbow', 'right_elbow', 'left_wrist', 
                        'right_wrist', 'left_pinky', 'right_pinky', 
                        'left_index', 'right_index', 'left_thumb', 
                        'right_thumb', 'left_hip', 'right_hip', 'left_knee', 
                        'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 
                        'right_heel', 'left_foot_index', 'right_foot_index', 
                        'hips', 'chest', 'spine', 'jaw']
            
            self.assertIsInstance(body_landmarks[1], BodyLandmarks3d)
            
            for landmark_name in landmarks:
                self.assertIn(landmark_name, list(body_landmarks[1].get_msg().keys()))
                landmark_info = body_landmarks[1].get_msg()[landmark_name]
                self.assertEqual(landmark_name, landmark_info['name'])
                self.assertEqual(landmark_info['type'], "3d")
                self.assertIn('coordinate_x', landmark_info.keys())
                self.assertIn('coordinate_y', landmark_info.keys())
                self.assertIn('coordinate_z', landmark_info.keys())


            human_pose_estimator.add_frame(frame, timestamp)
        
        while True:
            body_landmarks = human_pose_estimator.destroy_all_buffers()
            if body_landmarks is None:
                break
        
        self.assertEqual(body_landmarks, None)

    def test_pipeline(self):
        person_detectors = [YoloV3()]
        trackings = [Single(max_age=10),
                     Sort(max_age=10),
                     DeepSort(max_age=10)]
        predictors_2d = [Dummy2D(),
                         Mediapipe2D(),
                         CPN(),
                         Lightweight2D()]
        predictors_3d = [Dummy3D(window_length=1), 
                         MHFormer(window_length=243),
                         VideoPose3D(window_length=243),
                         MixSTE(window_length=243),
                         MotionBert(window_length=243)]

        for person_detector in person_detectors:
            for tracking in trackings:
                for predictor_2d in predictors_2d:
                    for predictor_3d in predictors_3d:
                        print(f"Testing {person_detector.__class__.__name__} | {tracking.__class__.__name__} | {predictor_2d.__class__.__name__} | {predictor_3d.__class__.__name__}")
                        hpe = HPE(predictor_2d, predictor_3d, predictor_3d.window_length, person_detector=person_detector, tracking=tracking)
                        self.pipeline_test(hpe)

if __name__ == '__main__':
    unittest.main()
