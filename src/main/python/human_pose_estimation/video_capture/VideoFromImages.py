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
import glob
from datetime import datetime
import os

class VideoFromImages(VideoCapture):
    """
    Selects images as input for the pose estimator.
    """
    def __init__(self, images_folder: str, infinite_loop: bool):
        jpg_files = glob.glob(os.path.join(images_folder, "*.jpg"))
        png_files = glob.glob(os.path.join(images_folder, "*.png"))
        self._images_path = jpg_files + png_files
        try:
            timestamp_str = os.path.basename(os.path.splitext(self._images_path[0])[0])
            parsed_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H_%M_%S.%f")
            self._images_path = sorted(self._images_path, key=self.extract_timestamp)
        except ValueError:
            self._images_path.sort(key=os.path.getctime)
            
        if len(self._images_path) == 0:
            raise ValueError(f"Images not found in folder {images_folder}")
        self._finished = False
        self._infinite_loop = infinite_loop
        self._index = -1
        self.frames = {}
        self.fps = 30 # By default
        
    def extract_timestamp(self, filename):
        # Assuming the timestamp is always at the beginning of the filename
        timestamp_str = os.path.basename(os.path.splitext(filename)[0])
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H_%M_%S.%f")
        return timestamp

# Path to the directory 
    def get_frame_by_timestamp_and_delete(self, timestamp):
        frame = self.frames[timestamp]
        del self.frames[timestamp]
        return frame

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
        #timestamp = os.path.basename(img_path).replace(".png", "")
        timestamp = self._index
        frame = cv2.imread(img_path)
        #self.frames[timestamp] = frame
        frame = cv2.putText(frame, img_path, (5,10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 0, 0), 1, cv2.LINE_AA)
        self.frames[timestamp] = frame
        return frame, timestamp
        
    
    def video_finished(self):
        return self._finished