import unittest
import os
import sys
sys.path.append(os.path.join('src', 'pose-estimation'))  # To avoid importing problems
sys.path.append(os.path.join('src', 'pose-estimation', 'models', 'predictors_3d', 'mixste'))
sys.path.append(os.path.join('src', 'pose-estimation', 'models', 'predictors_3d', 'motionbert'))

from common_pose.BodyLandmarks import BodyLandmarks2d, BodyLandmarks3d, Landmark

# Pose Predictors from RGB to 2D
from models.predictors_2d.Dummy2D import Dummy2D

# Pose Predictors from 2D to 3D
from models.predictors_3d.Dummy3D import Dummy3D
from models.predictors_3d.MHFormer import MHFormer
from models.predictors_3d.VideoPose3D import VideoPose3D
from models.predictors_3d.MotionBert import MotionBert
from models.predictors_3d.MixSTE import MixSTE

# Video Capture
from video_capture.VideoFromImages import VideoFromImages

class TestThreeDimensionsPredictor(unittest.TestCase):
    """
    Tests to check that the 3D Human Pose Estimators work properly
    """
    def setUp(self) -> None:
        self._gpu = '0' # Id of the GPU to check
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu

        # Two dimensions predictor:
        self._predictor_2d = Dummy2D()

        # Defining input
        images_path = os.path.join("data", "demos", "single_image")
        self._videoCapture = VideoFromImages(images_path, infinite_loop=True)

    def _init_buffers_test(self, predictor):
        frame, timestamp = self._videoCapture.get_frame()
        keypoints = self._predictor_2d.get_frame_keypoints(frame)
        converted_keypoints = predictor.translate_keypoints_2d(keypoints)

        for _ in range(10000): # If window is larger than 1e4, it should be increased
            is_initialized = predictor.init_buffers(frame, converted_keypoints, timestamp, keypoints)
            if is_initialized:
                break

        self.assertTrue(is_initialized)

    def _destroy_buffer_test(self, predictor):
        frame, timestamp = self._videoCapture.get_frame()
        keypoints = self._predictor_2d.get_frame_keypoints(frame)
        converted_keypoints = predictor.translate_keypoints_2d(keypoints)

        for _ in range(10000): # If window is larger than 1e4, it should be increased
            is_initialized = predictor.init_buffers(frame, converted_keypoints, timestamp, keypoints)
            if is_initialized:
                break
        
        self.assertTrue(is_initialized)

        for _ in range(10000): # If window is larger than 1e4, it should be increased
            body_landmarks = predictor.destroy_buffer()
            if body_landmarks is None:
                break
        
        self.assertEqual(body_landmarks, None)

    def _add_frame_test(self, predictor):
        frame, timestamp = self._videoCapture.get_frame()
        keypoints = self._predictor_2d.get_frame_keypoints(frame)
        converted_keypoints = predictor.translate_keypoints_2d(keypoints)

        for _ in range(10000): # If window is larger than 1e4, it should be increased
            is_initialized = predictor.add_frame(frame, converted_keypoints, timestamp, keypoints)
            if is_initialized:
                break

        self.assertTrue(is_initialized)

    def _translate_keypoints_2d_test(self, predictor):
        frame, timestamp = self._videoCapture.get_frame()
        keypoints = self._predictor_2d.get_frame_keypoints(frame)
        converted_keypoints = predictor.translate_keypoints_2d(keypoints)
        
        self.assertEqual(len(converted_keypoints.shape), 2)
        self.assertEqual(converted_keypoints.shape[1], 2)
        self.assertGreater(converted_keypoints.shape[0], 0)

    def _get_3d_keypoints_test(self, predictor):
        # Init buffers
        frame, timestamp = self._videoCapture.get_frame()
        keypoints = self._predictor_2d.get_frame_keypoints(frame)
        converted_keypoints = predictor.translate_keypoints_2d(keypoints)

        for _ in range(10000): # If window is larger than 1e4, it should be increased
            is_initialized = predictor.init_buffers(frame, converted_keypoints, timestamp, keypoints)
            if is_initialized:
                break

        self.assertTrue(is_initialized)

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
        
        keypoints_3d = predictor.get_3d_keypoints()
        self.assertIsInstance(keypoints_3d, BodyLandmarks3d)
        
        for landmark_name in landmarks:
            self.assertIn(landmark_name, list(keypoints_3d.get_msg().keys()))
            landmark_info = keypoints_3d.get_msg()[landmark_name]

            self.assertEqual(landmark_name, landmark_info['name'])
            self.assertEqual(landmark_info['type'], "3d")
            self.assertIn('coordinate_x', landmark_info.keys())
            self.assertIn('coordinate_y', landmark_info.keys())
            self.assertIn('coordinate_z', landmark_info.keys())



class TestDummy(TestThreeDimensionsPredictor):
    def test_init_buffers_dummy(self):
        super()._init_buffers_test(Dummy3D())

    def test_destroy_buffer_dummy(self):
        super()._destroy_buffer_test(Dummy3D())
    
    def test_add_frame_dummy(self):
        super()._add_frame_test(Dummy3D())
    
    def test_translate_keypoints_2d_dummy(self):
        super()._translate_keypoints_2d_test(Dummy3D())
    
    def test_get_3d_keypoints_dummy(self):
        super()._get_3d_keypoints_test(Dummy3D())

class TestMHFormer(TestThreeDimensionsPredictor):
    def test_init_buffers_mhformer(self):
        super()._init_buffers_test(MHFormer())

    def test_destroy_buffer_mhformer(self):
        super()._destroy_buffer_test(MHFormer())

    def test_add_frame_mhformer(self):
        super()._add_frame_test(MHFormer())
    
    def test_translate_keypoints_2d_mhformer(self):
        super()._translate_keypoints_2d_test(MHFormer())
    
    def test_get_3d_keypoints_mhformer(self):
        super()._get_3d_keypoints_test(MHFormer())

class TestVideoPose(TestThreeDimensionsPredictor):
    def test_init_buffers_videopose(self):
        super()._init_buffers_test(VideoPose3D())

    def test_destroy_buffer_videopose(self):
        super()._destroy_buffer_test(VideoPose3D())

    def test_add_frame_videopose(self):
        super()._add_frame_test(VideoPose3D())
    
    def test_translate_keypoints_2d_videopose(self):
        super()._translate_keypoints_2d_test(VideoPose3D())
    
    def test_get_3d_keypoints_videopose(self):
        super()._get_3d_keypoints_test(VideoPose3D())

class TestMotionBert(TestThreeDimensionsPredictor):
    def test_init_buffers_videopose(self):
        super()._init_buffers_test(MotionBert())

    def test_destroy_buffer_videopose(self):
        super()._destroy_buffer_test(MotionBert())

    def test_add_frame_videopose(self):
        super()._add_frame_test(MotionBert())
    
    def test_translate_keypoints_2d_videopose(self):
        super()._translate_keypoints_2d_test(MotionBert())
    
    def test_get_3d_keypoints_videopose(self):
        super()._get_3d_keypoints_test(MotionBert())

class TestMixSTE(TestThreeDimensionsPredictor):
    def test_init_buffers_videopose(self):
        super()._init_buffers_test(MixSTE())

    def test_destroy_buffer_videopose(self):
        super()._destroy_buffer_test(MixSTE())

    def test_add_frame_videopose(self):
        super()._add_frame_test(MixSTE())
    
    def test_translate_keypoints_2d_videopose(self):
        super()._translate_keypoints_2d_test(MixSTE())
    
    def test_get_3d_keypoints_videopose(self):
        super()._get_3d_keypoints_test(MixSTE())

if __name__ == '__main__':
    unittest.main()
