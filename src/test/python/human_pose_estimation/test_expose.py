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
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks3d
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation'))  # To avoid importing problems
sys.path.append(os.path.join('src', 'main', 'python'))
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d'))
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models'))
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'expose'))
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Pose Predictors from RGB to 2D
from human_pose_estimation.models.predictors_2d.Dummy2D import Dummy2D

# Pose Predictors from 2D to 3D

from human_pose_estimation.models.predictors_3d.ExPose import ExPose


from human_pose_estimation.models.person_detector.YoloV3 import YoloV3
from human_pose_estimation.models.tracking.Single import Single

# Video Capture
from human_pose_estimation.video_capture.VideoFromImages import VideoFromImages

class TestThreeDimensionsPredictor(unittest.TestCase):
    """
    Tests to check that the 3D Human Pose Estimators work properly
    """
    @classmethod
    def setUpClass(cls):
        cls._gpu = '0' # Id of the GPU to check
        os.environ["CUDA_VISIBLE_DEVICES"] = cls._gpu

        # Defining input
        cls._videoCapture = VideoFromImages(os.path.join("data", "demos", "single_image"), infinite_loop=True)

        # Defining person detector
        cls._person_detector = YoloV3()
        # Defining tracker
        cls._tracking = Single(max_age=10)
        # Two dimensions predictor
        cls._predictor_2d = Dummy2D()

        cls.expose = ExPose(window_length=243, video_filename='example')

        

    def test_translate_keypoints_2d(self):
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

        keypoints_2d = self._predictor_2d.get_frame_keypoints(frame, cropped_frame, person_bbox) # Note that person bbox does not match with frame, it is used for offset keypoints 2d
        converted_keypoints = self.expose.translate_keypoints_2d(keypoints_2d)
        #keypoints3d = predictor_3d.get_3d_keypoints(converted_keypoints)
        
        self.assertEqual(converted_keypoints, None)
        #self.assertEqual(converted_keypoints.shape[1], 2)
        #self.assertGreater(converted_keypoints.shape[0], 0)
           # Define test inputs
        
    def test_get_3d_keypoints(self):
        # Define test inputs

        bodyLandMarks2d = None
        timestamp = 1
        input_2D_no = 1
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

        # Call the method under test
        result = self.expose.get_3d_keypoints(img_w, img_h, bodyLandMarks2d, timestamp, input_2D_no, frame, person_bbox)

        # Perform assertions to validate the result
        self.assertIsNotNone(result)
        #self.assertEqual(result.shape[1], 3)
        
class BodyLandmarks3dTestCase(unittest.TestCase):
    def test_body_landmarks3d_initialization(self):
        # Create sample values for the variables
        raw = 1
        timestamp = 1
        nose = 1
        left_eye_inner = 1
        
        # Define other landmark variables
        left_eye = 2
        left_eye_outer = 3
        right_eye_inner = 4
        right_eye = 5
        right_eye_outer = 6
        left_ear = 7
        right_ear = 8
        mouth_left = 9
        mouth_right = 10
        left_shoulder = 11
        right_shoulder = 12
        left_elbow = 13
        right_elbow = 14
        left_wrist = 15
        right_wrist = 16
        left_pinky = 17
        right_pinky = 18
        left_index = 19
        right_index = 20
        left_thumb = 21
        right_thumb = 22
        left_hip = 23
        right_hip = 24
        left_knee = 25
        right_knee = 26
        left_ankle = 27
        right_ankle = 28
        left_heel = 29
        right_heel = 30
        left_foot_index = 31
        right_foot_index = 32
        jaw = 33
        chest = 34
        spine = 35
        hips = 36
        bodyLandmarks2d = 37
        
        # Create the BodyLandmarks3d object
        body_landmarks3d = BodyLandmarks3d(raw=raw, timestamp=timestamp,
                                           nose=nose, left_eye_inner=left_eye_inner,
                                           left_eye=left_eye, left_eye_outer=left_eye_outer,
                                           right_eye_inner=right_eye_inner, right_eye=right_eye,
                                           right_eye_outer=right_eye_outer, left_ear=left_ear,
                                           right_ear=right_ear, mouth_left=mouth_left,
                                           mouth_right=mouth_right, left_shoulder=left_shoulder,
                                           right_shoulder=right_shoulder, left_elbow=left_elbow,
                                           right_elbow=right_elbow, left_wrist=left_wrist,
                                           right_wrist=right_wrist, left_pinky=left_pinky,
                                           right_pinky=right_pinky, left_index=left_index,
                                           right_index=right_index, left_thumb=left_thumb,
                                           right_thumb=right_thumb, left_hip=left_hip,
                                           right_hip=right_hip, left_knee=left_knee,
                                           right_knee=right_knee, left_ankle=left_ankle,
                                           right_ankle=right_ankle, left_heel=left_heel,
                                           right_heel=right_heel, left_foot_index=left_foot_index,
                                           right_foot_index=right_foot_index, jaw=jaw,
                                           chest=chest, spine=spine, hips=hips,
                                           bodyLandmarks2d=bodyLandmarks2d)
        
        # Perform assertions to check if the object is correctly initialized
        self.assertEqual(body_landmarks3d._raw, raw)
        self.assertEqual(body_landmarks3d.timestamp, timestamp)
        self.assertEqual(body_landmarks3d._nose, nose)
        self.assertEqual(body_landmarks3d._left_eye_inner, left_eye_inner)
        self.assertEqual(body_landmarks3d._left_eye, left_eye)
        self.assertEqual(body_landmarks3d._left_eye_outer, left_eye_outer)
        self.assertEqual(body_landmarks3d._right_eye_inner, right_eye_inner)
        self.assertEqual(body_landmarks3d._right_eye, right_eye)
        self.assertEqual(body_landmarks3d._right_eye_outer, right_eye_outer)
        self.assertEqual(body_landmarks3d._left_ear, left_ear)
        self.assertEqual(body_landmarks3d._right_ear, right_ear)
        self.assertEqual(body_landmarks3d._mouth_left, mouth_left)
        self.assertEqual(body_landmarks3d._mouth_right, mouth_right)
        self.assertEqual(body_landmarks3d._left_shoulder, left_shoulder)
        self.assertEqual(body_landmarks3d._right_shoulder, right_shoulder)
        self.assertEqual(body_landmarks3d._left_elbow, left_elbow)
        self.assertEqual(body_landmarks3d._right_elbow, right_elbow)
        self.assertEqual(body_landmarks3d._left_wrist, left_wrist)
        self.assertEqual(body_landmarks3d._right_wrist, right_wrist)
        self.assertEqual(body_landmarks3d._left_pinky, left_pinky)
        self.assertEqual(body_landmarks3d._right_pinky, right_pinky)
        self.assertEqual(body_landmarks3d._left_index, left_index)
        self.assertEqual(body_landmarks3d._right_index, right_index)
        self.assertEqual(body_landmarks3d._left_thumb, left_thumb)
        self.assertEqual(body_landmarks3d._right_thumb, right_thumb)
        self.assertEqual(body_landmarks3d._left_hip, left_hip)
        self.assertEqual(body_landmarks3d._right_hip, right_hip)
        self.assertEqual(body_landmarks3d._left_knee, left_knee)
        self.assertEqual(body_landmarks3d._right_knee, right_knee)
        self.assertEqual(body_landmarks3d._left_ankle, left_ankle)
        self.assertEqual(body_landmarks3d._right_ankle, right_ankle)
        self.assertEqual(body_landmarks3d._left_heel, left_heel)
        self.assertEqual(body_landmarks3d._right_heel, right_heel)
        self.assertEqual(body_landmarks3d._left_foot_index, left_foot_index)
        self.assertEqual(body_landmarks3d._right_foot_index, right_foot_index)
        self.assertEqual(body_landmarks3d._jaw, jaw)
        self.assertEqual(body_landmarks3d._chest, chest)
        self.assertEqual(body_landmarks3d._spine, spine)
        self.assertEqual(body_landmarks3d._hips, hips)
        self.assertEqual(body_landmarks3d._bodyLandmarks2d, bodyLandmarks2d)

        

if __name__ == '__main__':
    unittest.main()
