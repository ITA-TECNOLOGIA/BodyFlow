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

class Tracking(ABC):
    def __init_subclass__(cls, **kwargs):
        """
        Initializes the subclass of Tracking.

        This method is called automatically when a subclass of Tracking is defined.
        It checks if the subclass is Tracking itself and raises a TypeError if so, since it is an abstract class.

        Args:
            **kwargs: Optional keyword arguments to pass to the superclass constructor.
        """
        super().__init_subclass__(**kwargs)
        if cls.__name__ == 'Tracking':
            raise TypeError("TypeError: This is an Abstract Class and it cannot be instantiated")

    
    def __init__(self, max_age):
        """
        Initializes the Tracking object.

        Args:
            max_age: The maximum age in frames for a track to be considered valid.

        This method sets the maximum age and should be called when creating a new Tracking object.
        """
        self._max_age = max_age

    @property
    def max_age(self):
        """
        Returns the maximum age allowed for a track.

        This method returns the maximum age allowed for a track, as set during object initialization.
        """
        return self._max_age

    @abstractmethod
    def update(self, bboxs, scores, frame) -> dict:
        """
        Updates the tracker with new information from a frame.

        Args:
            bboxs: A list of bounding boxes for persons in the frame.
            scores: A list of scores or confidences for each predicted bounding box.
            frame: The input frame to track.

        Returns:
            A dictionary representing the updated tracking information.

        This is an abstract method that should be implemented by subclasses.
        It represents the tracking functionality of a person tracker.
        """
        pass
