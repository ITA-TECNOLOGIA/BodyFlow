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
        self.frames = {}
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        
    
    def get_frame_by_timestamp_and_delete(self, timestamp):
        frame = self.frames[timestamp]
        del self.frames[timestamp]
        return frame
        
    
    def get_frame(self):
        ret, frame = self._cap.read()

        # Infinite loop
        if self._cap.get(1)>self._cap.get(7)-2:
            if self._infinite_loop:
                self._cap.set(1,0) #if frame count > than total frame number, next frame will be zero
            
        if ret:
            self._timestamp += 1
            frame = cv2.putText(frame, f"{self._timestamp}", (5,10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 0, 0), 1, cv2.LINE_AA)
            self.frames[self._timestamp] = frame
            return frame, self._timestamp
        else:
            logging.debug("Video finished!")
            self._finished = True
            return None, None
    
    def video_finished(self):
        return self._finished
