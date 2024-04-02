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

from models.PersonDetector import PersonDetector
from models.person_detector.YoloV3 import YoloV3
# Video Capture
from video_capture.VideoFromImages import VideoFromImages

class TestPersonDetector(unittest.TestCase):

    def setUp(self) -> None:
        self._gpu = '0' # Id of the GPU to check
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu

        # Defining input
        images_path = os.path.join("data", "demos", "single_image")
        self._videoCapture = VideoFromImages(images_path, infinite_loop=True)
    

    def _get_prediction_test(self, detector: PersonDetector):
        frame, _ = self._videoCapture.get_frame()

        prediction = detector.predict(frame)
        # We assume that this is a very easy test and all person detectors should detect the person in it
        self.assertIsInstance(prediction, tuple)
        bboxs, scores = prediction
        bboxs_expected_shape = (1, 4)
        assert bboxs.shape == bboxs_expected_shape, f"Unexpected shape. Expected {bboxs_expected_shape}, but got {bboxs.shape}"
        scores_expected_shape = (1, 1)
        assert scores.shape == scores_expected_shape, f"Unexpected shape. Expected {scores_expected_shape}, but got {scores.shape}"

    """
    TODO
    - Check limits of bounding boxes and high scores
    - Test with a picture of more than one single person
    - Test it with a simple demo video?
    """

class TestYoloV3(TestPersonDetector):
    def test_get_prediction_yolov3(self):
        super()._get_prediction_test(YoloV3())

if __name__ == '__main__':
    unittest.main()