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

class Single(Tracking):
    def __init__(self, max_age=1):
        """
        Initializes the Single object.

        Args:
            max_age: The maximum age in frames for a track to be considered valid.

        This method initializes the Single object by setting the maximum age for tracks.

        """
        super().__init__(max_age)

    def update(self, bboxs, scores, frame):
        """
        Updates the Single tracker with new information from a frame.

        Args:
            bboxs: A list of bounding boxes for persons in the frame.
            scores: A list of scores or confidences for each predicted bounding box.
            frame: The input frame to track.

        Returns:
            A dictionary containing the updated bounding boxes for tracked people.
            The dictionary keys represent the track IDs, and the values are the corresponding bounding boxes.

        Note:
            The input bounding boxes and scores are sorted based on scores in descending order.
            Only the first bounding box from the sorted list is considered, and it is assigned the track ID of 1.

        """
        sorted_data = sorted(zip(bboxs, scores), key=lambda x: x[1], reverse=True)
        sorted_bboxs, sorted_scores = zip(*sorted_data)
        bounding_boxes = {}
        bounding_boxes[1] = [round(i, 2) for i in list(sorted_bboxs[0])]  # Always id 1

        return bounding_boxes
