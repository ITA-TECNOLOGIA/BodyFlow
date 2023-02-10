import unittest
import os
import sys
sys.path.append(os.path.join('src', 'pose-estimation'))  # To avoid importing problems
sys.path.append(os.path.join('src', 'pose-estimation', 'models', 'predictors_3d', 'mixste'))
sys.path.append(os.path.join('src', 'pose-estimation', 'models', 'predictors_3d', 'motionbert'))

from common_pose.BodyLandmarks import BodyLandmarks3d

# Video capture
from video_capture.VideoFromImages import VideoFromImages

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
            self.assertEqual(type(body_landmarks), BodyLandmarks3d)
            human_pose_estimator.add_frame(frame, timestamp)
        
        while True:
            body_landmarks = human_pose_estimator.destroy_buffer()
            if body_landmarks is None:
                break
        
        self.assertEqual(body_landmarks, None)

    def test_pipeline(self):
        predictors_2d = [Dummy2D(), Mediapipe2D(), CPN(), Lightweight2D()]
        predictors_3d = [Dummy3D(), MHFormer(), VideoPose3D(), MixSTE(), MotionBert()]

        for predictor_2d in predictors_2d:
            for predictor_3d in predictors_3d:
                hpe = HPE(predictor_2d, predictor_3d)
                self.pipeline_test(hpe)


if __name__ == '__main__':
    unittest.main()
