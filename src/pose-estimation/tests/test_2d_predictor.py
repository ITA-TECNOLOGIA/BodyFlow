import unittest
import os
import sys
sys.path.append(os.path.join('src', 'pose-estimation'))  # To avoid importing problems

from common_pose.BodyLandmarks import BodyLandmarks2d

# Pose Predictors from RGB to 2D
from models.HPE2D import HPE2D
from models.predictors_2d.Dummy2D import Dummy2D
from models.predictors_2d.Mediapipe2D import Mediapipe2D
from models.predictors_2d.CPN import CPN
from models.predictors_2d.Lightweight2D import Lightweight2D

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

    def _get_keypoints_test(self, predictor: HPE2D):
        frame, _ = self._videoCapture.get_frame()
        keypoints = predictor.get_frame_keypoints(frame)

        # Check all landmarks
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
        
        self.assertIsInstance(keypoints, BodyLandmarks2d)
        self.assertEqual(len(landmarks), len(list(keypoints.get_msg().keys())))

        
        # Check values greater than 0 (goes from 0, to width/height pixels)
        for landmark_name, landmark_info in keypoints.get_msg().items():
            self.assertIn(landmark_name, landmarks)

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

