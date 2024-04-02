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

import os
import logging
from datetime import datetime
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def get_frames_video(path, vis_aux):
        """
        Generates a list of frames from a video file and saves them as individual images.

        Parameters:
            path (str): The path to the video file.
            vis_aux (str): The directory where the frames will be saved.

        Returns:
            images_path (List[str]): A list of paths to the saved frame images.
            fps (float): The frames per second of the video.
        """
        import cv2
        vcap = cv2.VideoCapture(path)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        images_path = []
        frame_count=0
        ret = True
        while ret:
            ret, img = vcap.read() 
            if ret:
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                output_path = os.path.join(vis_aux, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(output_path, img)
                images_path.append(os.path.join(vis_aux, f"frame_{frame_count:05d}.jpg"))
                frame_count += 1
        
        logging.info('Video frames processed')
        return images_path, fps

def get_frames_pictures(path):
    """
    Get a list of image file paths in a given directory and sort them by creation time.

    Args:
        path (str): The path to the directory containing the image files.

    Returns:
        tuple: A tuple containing the list of image file paths and the default FPS (frames per second) value.

    Raises:
        None

    Example:
        >>> get_frames_pictures('/path/to/images')
        (['/path/to/images/image1.jpg', '/path/to/images/image2.png'], 30)
    """

    import glob
    jpg_files = glob.glob(os.path.join(path, "*.jpg"))
    png_files = glob.glob(os.path.join(path, "*.png"))
    images_path = jpg_files + png_files
    try:
        timestamp_str = os.path.basename(os.path.splitext(images_path[0])[0])
        parsed_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H_%M_%S.%f")
        images_path = sorted(images_path, key=extract_timestamp)
    except ValueError:
        images_path.sort(key=os.path.getctime)
    # images_path.sort(key=os.path.getctime)
    fps = 30 # Default FPS
    logging.info('Pictures frames list processed')
    return images_path, fps

def get_frames(path, input_type, vis_aux):
        """
        Get frames from a video or a set of pictures.

        Parameters:
            path (str): The path to the video or the directory containing the pictures.
            input_type (str): The type of input, either "video" or "pictures".
            vis_aux (bool): Flag indicating whether to visualize auxiliary information.

        Returns:
            list: A list of frames extracted from the video or pictures.

        Raises:
            NotImplementedError: If the input_type is not "video" or "pictures".
        """
        if input_type == "video":
            return get_frames_video(path, vis_aux)
        elif input_type == "pictures":
            return get_frames_pictures(path)
        else:
            raise NotImplementedError(f"Visualization not available for input type {input_type}")
        
def extract_timestamp(filename):
    # Assuming the timestamp is always at the beginning of the filename
    timestamp_str = os.path.basename(os.path.splitext(filename)[0])
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H_%M_%S.%f")
    return timestamp