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

from human_pose_estimation.models.PersonDetector import PersonDetector
from human_pose_estimation.models.person_detector.yolov3.human_detector import load_model as yolo_model
from human_pose_estimation.models.person_detector.yolov3.human_detector import yolo_human_det as yolo_det
import logging

class YoloV3(PersonDetector):
    def __init__(self, model_path):
        """
        Initializes the YoloV3 object.

        This method sets the configuration parameters for YOLOv3 model, such as the input dimensions,
        score threshold for object detection, and NMS (non-maximum suppression) threshold.
        It also loads the YOLOv3 human detection model.

        Note:
            The configuration parameters can be adjusted based on specific requirements.

        """
        self._input_dim = 416
        self._thread_score = 0.8
        self._nms_thresh = 0.4  # NMS Threshold

        self._human_model = yolo_model(inp_dim=self._input_dim, model_path=model_path)

    def predict(self, frame):
        """
        Performs person detection using YOLOv3 model on a given frame.

        Args:
            frame: The input frame on which person detection is performed.

        Returns:
            A tuple containing two arrays. The first array represents the predicted bounding boxes for persons,
            and the second array represents the corresponding scores or confidences of the predictions.

        Note:
            If no person is detected in the frame, it returns None.

        """
        bboxs, scores = yolo_det(frame, self._human_model, reso=self._input_dim, confidence=self._thread_score,
                                 nms_thresh=self._nms_thresh)

        if bboxs is None or not bboxs.any():
            logging.debug("No person detected!")
            return None, None
        else:
            return bboxs, scores
