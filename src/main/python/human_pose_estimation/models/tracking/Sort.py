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

from human_pose_estimation.models.Tracking import Tracking
from human_pose_estimation.models.tracking.sort.sort_utils import SortTracking

class Sort(Tracking):
    def __init__(self, max_age):
        """
        Initializes the Sort object.

        Args:
            max_age: The maximum age in frames for a track to be considered valid.

        This method initializes the Sort object by setting the maximum age for tracks
        and creating an instance of the SortTracking algorithm.

        """
        super().__init__(max_age)
        self._tracking = SortTracking(max_age=self._max_age, min_hits=0)

    def update(self, bboxs, scores, frame):
        """
        Updates the Sort tracker with new information from a frame.

        Args:
            bboxs: A list of bounding boxes for persons in the frame.
            scores: A list of scores or confidences for each predicted bounding box.
            frame: The input frame to track.

        Returns:
            A dictionary containing the updated bounding boxes for tracked people.
            The dictionary keys represent the track IDs, and the values are the corresponding bounding boxes.

        Note:
            The Sort algorithm is used to update the tracking based on the input bounding boxes.
            The track IDs are extracted from the updated tracking results and converted to integers.
            The bounding boxes are rounded to two decimal places before being added to the dictionary.

        """
        people_track = self._tracking.update(bboxs)
        ids_list_float = people_track[:, -1].tolist()
        ids_list = [int(item) for item in ids_list_float]

        bounding_boxes = {}
        for i in range(0, people_track.shape[0]):
            bbox_float = people_track[i, :-1].tolist()
            bbox = [round(i, 2) for i in bbox_float]
            bounding_boxes[ids_list[i]] = bbox

        return bounding_boxes
