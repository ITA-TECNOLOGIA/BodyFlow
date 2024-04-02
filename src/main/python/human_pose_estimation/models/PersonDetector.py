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
from typing import List, Tuple

class PersonDetector(ABC):
    @abstractmethod
    def __init__(self):
        """
        Initializes the PersonDetector object. It detects people in a frame.

        This is an abstract method that should be implemented by subclasses.
        It raises a TypeError since the class is an abstract base class and should not be instantiated directly.
        """
        raise TypeError("TypeError: This is an Abstract Class and it cannot be instantiated")
    
    @abstractmethod
    def predict(self, frame) -> Tuple[List, List]:
        """
        Performs prediction on a frame and returns the results.

        Args:
            frame: The input frame on which the people prediction is performed.

        Returns:
            A tuple containing two lists. The first list represents the predicted bounding boxes for persons,
            and the second list represents the corresponding scores or confidences of the predictions.

        This is an abstract method that should be implemented by subclasses.
        It represents the prediction functionality of a person detector.
        """
        pass
