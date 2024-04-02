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
from deep_sort_realtime.deepsort_tracker import DeepSort as deepSortAlgorithm

class DeepSort(Tracking):
    def __init__(self, max_age):
        """
        Initializes the DeepSort object.

        Args:
            max_age: The maximum age in frames for a track to be considered valid.

        This method initializes the DeepSort object by setting the maximum age for tracks and creating
        an instance of the DeepSort tracking algorithm.

        """
        super().__init__(max_age)
        self._tracking = deepSortAlgorithm(n_init=0, max_age=self._max_age)

    def update(self, bboxs, scores, frame):
        """
        Updates the DeepSort tracker with new information from a frame.

        Args:
            bboxs: A list of bounding boxes for persons in the frame.
            scores: A list of scores or confidences for each predicted bounding box.
            frame: The input frame to track.

        Returns:
            A dictionary containing the updated bounding boxes for tracked people.
            The dictionary keys represent the track IDs, and the values are the corresponding bounding boxes.

        Note:
            The input bounding boxes and scores are converted to the required format for the DeepSort algorithm.

        """
        bboxs_deepsort = []
        for i in range(0, len(scores)):
            bbox_deepsort = (
                [bboxs[i][0], bboxs[i][1], bboxs[i][2] - bboxs[i][0], bboxs[i][3] - bboxs[i][1]],
                scores[i][0],
                1,
            )  # Last is detection class, does not matter, always people
            bboxs_deepsort.append(bbox_deepsort)

        people_track = self._tracking.update_tracks(bboxs_deepsort, frame=frame)
        bounding_boxes = {}
        for track in people_track:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bounding_boxes[int(track_id)] = [round(i, 2) for i in list(ltrb)]
        return bounding_boxes
