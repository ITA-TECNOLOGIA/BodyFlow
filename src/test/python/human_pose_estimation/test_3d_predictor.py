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
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'mixste'))
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'motionbert'))

from common_pose.BodyLandmarks import BodyLandmarks2d, BodyLandmarks3d, Landmark

# Pose Predictors from RGB to 2D
from models.predictors_2d.Dummy2D import Dummy2D

# Pose Predictors from 2D to 3D
from models.predictors_3d.Dummy3D import Dummy3D
from models.predictors_3d.MHFormer import MHFormer
from models.predictors_3d.VideoPose3D import VideoPose3D
from models.predictors_3d.MotionBert import MotionBert
from models.predictors_3d.MixSTE import MixSTE


from models.person_detector.YoloV3 import YoloV3
from models.tracking.Single import Single

# Video Capture
from video_capture.VideoFromImages import VideoFromImages

class TestThreeDimensionsPredictor(unittest.TestCase):
    """
    Tests to check that the 3D Human Pose Estimators work properly
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
        # Two dimensions predictor
        self._predictor_2d = Dummy2D()

    def _translate_keypoints_2d_test(self, predictor_3d):
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
        converted_keypoints = predictor_3d.translate_keypoints_2d(keypoints_2d)
        
        self.assertEqual(len(converted_keypoints.shape), 2)
        self.assertEqual(converted_keypoints.shape[1], 2)
        self.assertGreater(converted_keypoints.shape[0], 0)



class TestDummy(TestThreeDimensionsPredictor):
    def test_translate_keypoints_2d_dummy(self):
        super()._translate_keypoints_2d_test(Dummy3D(window_length=1))


class TestMHFormer(TestThreeDimensionsPredictor):
    def test_translate_keypoints_2d_mhformer(self):
        super()._translate_keypoints_2d_test(MHFormer(window_length=243))

class TestVideoPose(TestThreeDimensionsPredictor):
    def test_translate_keypoints_2d_videopose(self):
        super()._translate_keypoints_2d_test(VideoPose3D(window_length=243))

class TestMotionBert(TestThreeDimensionsPredictor):
    def test_translate_keypoints_2d_videopose(self):
        super()._translate_keypoints_2d_test(MotionBert(window_length=243))

class TestMixSTE(TestThreeDimensionsPredictor):
    def test_translate_keypoints_2d_videopose(self):
        super()._translate_keypoints_2d_test(MixSTE(window_length=243))
        
if __name__ == '__main__':
    unittest.main()
