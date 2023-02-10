# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnológico de Aragón (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernández Ruiz                      ahernandez@itainnova.es
#          Ángel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Marañes Nueno                            cmaranes@itainnova.es
#          Irene López Bosque                                ilopez@itainnova.es
#          María de la Vega Rodrigálvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibáñez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocío Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------

from video_capture.VideoCapture import VideoCapture
import cv2
import logging

class VideoFromVideo(VideoCapture):
    """
    Selects video frames as input for the pose estimator.
    """
    def __init__(self, video_path: str, infinite_loop: bool):
        # Open video file
        self._cap = cv2.VideoCapture(video_path)
        self._finished = False
        self._infinite_loop = infinite_loop
        self._timestamp = -1
        
    def get_frame(self):
        ret, frame = self._cap.read()

        # Infinite loop
        if self._cap.get(1)>self._cap.get(7)-2:
            if self._infinite_loop:
                self._cap.set(1,0) #if frame count > than total frame number, next frame will be zero
            
        if ret:
            self._timestamp += 1
            return frame, self._timestamp
        else:
            logging.debug("Video finished!")
            self._finished = True
            return None, None
    
    def video_finished(self):
        return self._finished
