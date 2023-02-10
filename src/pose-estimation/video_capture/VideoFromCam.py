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

from video_capture.VideoCapture import VideoCapture
import cv2

class VideoFromCam(VideoCapture):
    """
    Selects a camera (webcam or via rstp) as input source for the pose estimator.
    """
    def __init__(self, video_source: str):
        # Open video file
        self._cap = cv2.VideoCapture(video_source)
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
