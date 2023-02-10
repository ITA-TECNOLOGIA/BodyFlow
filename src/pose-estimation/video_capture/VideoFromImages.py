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
import glob
import os

class VideoFromImages(VideoCapture):
    """
    Selects images as input for the pose estimator.
    """
    def __init__(self, images_folder: str, infinite_loop: bool):
        self._images_path = glob.glob(os.path.join(images_folder, "*.png"))
        self._images_path.sort(key=os.path.getctime)
        self._finished = False
        self._infinite_loop = infinite_loop
        self._index = -1
        
    def get_frame(self):
        self._index += 1
        if self._index >= len(self._images_path):
            if self._infinite_loop:
                self._index = -1
            else:
                print("Video finished!")
                self._finished = True
                return None, None

        img_path = self._images_path[self._index]
        timestamp = os.path.basename(img_path).replace(".png", "")
        frame = cv2.imread(img_path)
        return frame, timestamp
        
    
    def video_finished(self):
        return self._finished
