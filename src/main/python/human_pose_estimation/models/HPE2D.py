## --------------------------------------------------------------------------------
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

from abc import ABC, abstractmethod
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks2d

class HPE2D(ABC):
    """
    Human pose estimator in two dimensions.
    """
    def __init__(self):
        raise TypeError("TypeError: This is an Abstract Class and it cannot be instantiated")

    @abstractmethod
    def get_frame_keypoints(self, frame_full, cropped_frame, bbox) -> BodyLandmarks2d:
        """
        Returns a dict containing the 2D position of each landmark
        
        Args:
            frame_full: The full frame or image.
            cropped_frame: The cropped portion of the frame or image.
            bbox: The bounding box coordinates associated with the cropped frame.
        
        Returns:
            BodyLandmarks2d: An object representing the 2D position of each landmark.
        """
        pass