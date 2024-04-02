# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnologico de Aragon (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@itainnova.es
#          Angel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Maranes Nueno                            cmaranes@itainnova.es
#          Irene Lopez Bosque                                ilopez@itainnova.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibanez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocio Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from common_pose.BodyLandmarks import BodyLandmarks3d, BodyLandmarks2d
import numpy as np

class HPE3D(ABC):
    @abstractmethod
    def __init__(self):
        raise TypeError("TypeError: This is an Abstract Class and it cannot be instantiated")

    @abstractmethod
    def init_buffers(self, frame, keypoints_2d, timestamp, bodyLandmarks2d) -> bool:
        """
        It returns False while the buffers are not still full. Once they are full,
        it returns True.
        """
        pass

    @abstractmethod
    def destroy_buffer(self) -> BodyLandmarks3d:
        """
        It returns body landmarks until de buffer is destroyed. Once destruyed,
        it returns None
        """
        pass

    @abstractmethod
    def add_frame(self, frame, keypoints_2d, timestamp, bodyLandmarks2d) -> bool:
        """
        It adds a new frame into the buffer, returning true if the buffer is complete,
        meaning that it overrides the oldest frame. If false, it means the that buffer is
        not still full.
        """
        pass

    @abstractmethod
    def translate_keypoints_2d(self, keypoints_2d: BodyLandmarks2d) -> np.array:
        """
        It converts a dict of 2D poses into an array ready to be passed to 
        a 3D pose estimator.
        """
        pass

    @abstractmethod
    def get_3d_keypoints(self) -> BodyLandmarks3d: # It returns the 3d keypoints from the frame in the buffer
        """
        It predicts the corresponding 3D pose.
        """
        pass