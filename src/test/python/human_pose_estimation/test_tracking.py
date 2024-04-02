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

from models.Tracking import Tracking
from models.tracking.DeepSort import DeepSort
from models.tracking.Single import Single
from models.tracking.Sort import Sort

from models.person_detector.YoloV3 import YoloV3

# Video Capture
from video_capture.VideoFromImages import VideoFromImages

class TestTracking(unittest.TestCase):

    def setUp(self) -> None:
        self._gpu = '0' # Id of the GPU to check
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu

        # Defining input
        images_path = os.path.join("data", "demos", "single_image")
        self._videoCapture = VideoFromImages(images_path, infinite_loop=True)

        # Defining person detector
        self._person_detector = YoloV3()
    

    def _get_prediction_test(self, tracking: Tracking):
        frame, _ = self._videoCapture.get_frame()

        bboxs, scores = self._person_detector.predict(frame)
        # We assume that this is a very easy test and all person detectors should detect the person in it
        for i in range(0, 50):
            bounding_boxes = tracking.update(bboxs, scores, frame)
            if i > 0:
                assert len(bounding_boxes.keys()) == 1
                assert list(bounding_boxes.keys())[0] == 1
                self.assertListEqual(list(np.array(bounding_boxes[1], dtype=np.float32)), list(bboxs[0].astype(np.float32)))

class TestDeepSort(TestTracking):
    def test_get_prediction_deepsort(self):
        super()._get_prediction_test(DeepSort(max_age=10))

class TestSort(TestTracking):
    def test_get_prediction_deepsort(self):
        super()._get_prediction_test(Sort(max_age=10))

class TestSingle(TestTracking):
    def test_get_prediction_deepsort(self):
        super()._get_prediction_test(Single(max_age=10))

if __name__ == '__main__':
    unittest.main()