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

class VideoCapture(ABC):
    """
    Returns a video capture which provides the frames to estimate the pose.
    """
    def __init__(self):
        raise TypeError("TypeError: This is an Abstract Class and it cannot be instantiated")

    @abstractmethod
    def get_frame(self):
        """
        Returns the frame and its corresponding timestamp if there are still frames remaining. None otherwise.
        """
        pass

    @abstractmethod
    def video_finished(self) -> bool:
        """
        Returns True if the input has finished. False if there are frames still remaining.
        """
        pass