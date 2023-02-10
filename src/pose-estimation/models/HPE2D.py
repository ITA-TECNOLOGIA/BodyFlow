# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnológico de Aragón (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernández Ruiz                      ahernandez@itainnova.es
#          Ángel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Marañes Nueno                            cmaranes@itainnova.es
#          Irene López Bosque                                ilopez@itainnova.es
#          María de la Vega Rodrigálvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibáñez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocío Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from common_pose.BodyLandmarks import BodyLandmarks2d

class HPE2D(ABC):
    """
    Human pose estimator in two dimensions.
    """
    def __init__(self):
        raise TypeError("TypeError: This is an Abstract Class and it cannot be instantiated")

    @abstractmethod
    def get_frame_keypoints(self, frame) -> BodyLandmarks2d:
        """
        Returns a dict containing the 2D position of each landmark
        """
        pass