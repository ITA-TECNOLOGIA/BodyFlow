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

from abc import ABC, abstractmethod
from human_pose_estimation.common_pose.BodyLandmarks import BodyLandmarks3d, BodyLandmarks2d
import numpy as np

class HPE3D(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == 'HPE3D':
            raise TypeError("TypeError: This is an Abstract Class and it cannot be instantiated")

    def __init__(self, window_length):
        self._window_length = window_length

    @property
    def window_length(self):
        return self._window_length

    @abstractmethod
    def translate_keypoints_2d(self, keypoints_2d: BodyLandmarks2d) -> np.array:
        """
        It converts a dict of 2D poses into an array ready to be passed to 
        a 3D pose estimator.
        """
        pass

    @abstractmethod
    def get_3d_keypoints(self,
                         img_w:int,
                         img_h:int,
                         bodyLandmarks2d:BodyLandmarks2d,
                         timestamp:int,
                         input_2D_no:np.array,
                         frame:np.array,
                         bbox, # A list of 4 items
                         ) -> BodyLandmarks3d: # It returns the 3d keypoints from the frame in the buffer
        """
        It predicts the corresponding 3D pose.
        """
        pass