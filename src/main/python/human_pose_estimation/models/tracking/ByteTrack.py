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
from human_pose_estimation.models.tracking.bytetrack.byte_tracker import BYTETracker
import numpy as np

class ByteTrack(Tracking):
    def __init__(self, max_age):
        """
        Initializes the DeepSort object.

        Args:
            max_age: The maximum age in frames for a track to be considered valid.

        This method initializes the DeepSort object by setting the maximum age for tracks and creating
        an instance of the DeepSort tracking algorithm.

        """
        super().__init__(max_age)
        self._tracking = BYTETracker(
            track_thresh = 0.8, #0.7,  # tracking confidence threshold
            match_thresh = 0.9,   # the frames for keep lost tracks
            track_buffer = self._max_age,  # matching threshold for tracking
            frame_rate = 30 # FPS
        )
        
        ##track_thresh=0.45, match_thresh=0.8, track_buffer=50, frame_rate=60
    def update(self, bboxs, scores, frame):
        """
        Updates the ByteTrack tracker with new information from a frame.

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
        bboxs_bytetracker = []
        for i in range(0, len(scores)):
            bbox_bytetracker = (
                [bboxs[i][0], bboxs[i][1], bboxs[i][2], bboxs[i][3],
                scores[i][0],
                0]
            )  # Last is detection class, does not matter, always people
            bboxs_bytetracker.append(bbox_bytetracker)
        
        bboxs_bytetracker = np.asarray(bboxs_bytetracker, dtype='float32')
        people_track = self._tracking.update(bboxs_bytetracker, frame)
        bounding_boxes = {}
        for track in people_track:
            track_id = track[4]
            ltrb = track[0:4]
            bounding_boxes[int(track_id)] = [round(i, 2) for i in list(ltrb)]
            
        return bounding_boxes
