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