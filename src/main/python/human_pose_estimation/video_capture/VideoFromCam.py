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

from human_pose_estimation.video_capture.VideoCapture import VideoCapture
import cv2

class VideoFromCam(VideoCapture):
    """
    Selects a camera (webcam or via rstp) as input source for the pose estimator.
    """
    def __init__(self, video_source: str):
        # Open video file
        source = int(video_source)
        self._cap = cv2.VideoCapture(source)
        self._finished = False
        self._timestamp = -1
        
    def get_frame(self):
        ret, frame = self._cap.read()

        if ret:
            self._timestamp += 1
            return frame, self._timestamp
        else:
            print("Video finished!")
            self._finished = True
            return None, None
    
    def video_finished(self):
        return False # TODO, thing something more intelligent
