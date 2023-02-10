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

import argparse
import os
import time
import sys
import signal
import logging
from datetime import datetime

import os
import sys

sys.path.append(os.path.join('src', 'pose-estimation', 'models', 'predictors_3d', 'mixste'))
sys.path.append(os.path.join('src', 'pose-estimation', 'models', 'predictors_3d', 'motionbert'))

# Pose Sender
from common_pose.PoseLogger import PoseLogger

# Video capture
from video_capture.VideoCapture import VideoCapture
from video_capture.VideoFromVideo import VideoFromVideo
from video_capture.VideoFromImages import VideoFromImages
from video_capture.VideoFromCam import VideoFromCam

# Pose Predictors from RGB to 2D
from models.HPE2D import HPE2D
from models.predictors_2d.Dummy2D import Dummy2D
from models.predictors_2d.Mediapipe2D import Mediapipe2D
from models.predictors_2d.CPN import CPN
from models.predictors_2d.Lightweight2D import Lightweight2D

# Pose Predictors from 2D to 3D
from models.HPE3D import HPE3D
from models.predictors_3d.Dummy3D import Dummy3D
from models.predictors_3d.MHFormer import MHFormer

# Pose Predictor from RGB to 3D
from models.HPE import HPE

# Visualization
from common_pose.visualization import Visualization
 

def instance_video_capture(args) -> VideoCapture:
    """
    Returns a video capture which provides the frames to estimate the pose.
    """
    videoCapture = None
    if args.input == 'video':
        videoCapture = VideoFromVideo(args.video_path, args.infinite_loop)
    elif args.input == 'pictures':
        videoCapture = VideoFromImages(args.pictures_path, args.infinite_loop)
    elif args.input == 'cam':
        videoCapture = VideoFromCam(args.video_source)
    else:
        sys.exit(f"Input {args.input} not recognized/implemented.")
    
    return videoCapture


def instance_predictor_2d(args) -> HPE2D:
    """
    Returns a 2D human pose predictor. Given an RGB image, it returns the 3d pose.
        - dummy2d: Place holder. It always return the same value. This is done for dubbing purpuses.
        - mediapipe2d: Included. Fastest one.
        - cpn: Included.
        - lightweight: Included.
    """
    predictor_2d = None
    if args.predictor_2d == "dummy2d":
        predictor_2d = Dummy2D()
    elif args.predictor_2d == "mediapipe2d":
        predictor_2d = Mediapipe2D()  # TODO not running in GPU
    elif args.predictor_2d == "cpn":
        predictor_2d = CPN()
    elif args.predictor_2d == "lightweight":
        predictor_2d = Lightweight2D()
    else:
        sys.exit(f"2D detector {args.predictor_2d} not implemented!")
    return predictor_2d


def instance_predictor_3d(args) -> HPE3D:
    """
    Returns a 3D human pose predictor. Given a 2D estimated human pose, it lifts it to a 3D one.
        - dummy3d: Place holder made for debugging purposes. Included.
        - mhformer: Included.
    IMPORTANT:
        In order to use videopose, motionbert and mixste you need to download additional files 
        and model weights. Please refer to README.md for installation instrucctions.
    """
    predictor_3d = None
    if args.predictor_3d == "dummy3d":
        predictor_3d = Dummy3D()
    elif args.predictor_3d == "mhformer":
        predictor_3d = MHFormer()
    elif args.predictor_3d == "videopose":
        from models.predictors_3d.VideoPose3D import VideoPose3D
        predictor_3d = VideoPose3D()
    elif args.predictor_3d == "motionbert":
        from models.predictors_3d.MotionBert import MotionBert
        predictor_3d = MotionBert()
    elif args.predictor_3d == 'mixste':
        from models.predictors_3d.MixSTE import MixSTE
        predictor_3d = MixSTE()
    
    else:
        sys.exit(f"3D detector {args.predictor_3d} not implemented!")
    return predictor_3d


def inference(videoCapture: VideoCapture, human_pose_estimator: HPE, pose_logger: PoseLogger, export_csv = True):
    """
    Initialization of the buffers of the human pose estimator. Once initialized, it computes the 
    3D human pose in real time, sending the pose through a socket if specified.
    If the script is stopped with CTRL+C, it handles the signal, saves the poses in a .csv and
    finish the script.
    """

    # Initialize HPE, e.g., load buffer
    logging.debug("Initializing HPE...")
    while True:
        # Read frame
        frame, timestamp = videoCapture.get_frame()
        if videoCapture.video_finished():
            logging.error("HPE not initlialized. May not enough frames.")
            return False
        is_initialized = human_pose_estimator.init_buffers(frame, timestamp)
        if is_initialized:
            break
    
    logging.debug("Human pose estimator initilialized!")

    # https://code-maven.com/catch-control-c-in-python
    def handler(signum, frame):
        logging.warning("Received SIGING, exporting to csv and finishing...")
        pose_logger.export_csv()
        exit(1)
    signal.signal(signal.SIGINT, handler)

    counter = 0
    while True:
        start = time.time()

        # Read frame
        frame, timestamp = videoCapture.get_frame()
        if videoCapture.video_finished():
            logging.debug("Input finished, destroying buffer and making last predictions...")
            while True:
                body_landmarks = human_pose_estimator.destroy_buffer()
                if body_landmarks is None:
                    break
                else:
                    pose_logger.log_pose(body_landmarks)
            break

        # Compute pose (inference)
        body_landmarks = human_pose_estimator.predict_pose()

        pose_logger.log_pose(body_landmarks)

        # Add next frame to the buffer
        human_pose_estimator.add_frame(frame, timestamp)

        end = time.time()
        if counter % 30 == 0: logging.debug("Frame %d -> FPS: %2.f", counter, 1 / (end - start))
        counter += 1
    
    logging.info("Pose estimation finished!")
    if export_csv:
        pose_logger.export_csv()
    return True
        

if __name__ == "__main__":
    """
    Given an RGB frame, this module predicts the 3D pose estimation of the most confident person. The pose is log in a csv,
    but it also can be stream with sockets.
    """
    ########################################
    ####          Logging config        ####
    ########################################
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    #logging.basicConfig(level=logging.DEBUG)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logger_{timestamp}.log", level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_2d', type=str, default='cpn', help='2D Predictor: [dummy2d, mediapipe2d, cpn, lightweight]')
    parser.add_argument('--predictor_3d', type=str, default='videopose', help='3D Predictor: [dummy3d, mhformer] if downloaded: [videopose, motionbert, mixste]')
    parser.add_argument('--gpu', type=str, default='0', help='Id of the GPU. Empty string if CPU')
    parser.add_argument('--infinite_loop', action='store_true')
    parser.add_argument('--port', type=int, default=-1, help='Port the machine where the inference server is running. The Unity server will connect to this Port. -1 and pose will not be sent.')

    # ----- Input
    parser.add_argument('--input', default='video', choices=['video', 'pictures', 'cam'], help='Way in which the entry is read')
    
    # Video 
    parser.add_argument('--video_path', type=str, default='data/input/sample_video.mp4', help='Id of the GPU. Empty string if CPU')

    # Pictures
    parser.add_argument('--pictures_path', type=str, default='data/demos/single_image', help='Path of the pictures to estimate the pose')

    # Cam
    parser.add_argument('--video_source', type=str, default='0', help='Cam source. It can also be a rstp url')

    # Visualization
    parser.add_argument('--viz', type=bool, default=True, help='To plot the visualization set to True. Visualization is done after the inference is done.')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    videoCapture = instance_video_capture(args)

    predictor_2d = instance_predictor_2d(args)
    predictor_3d = instance_predictor_3d(args)

    human_pose_estimator = HPE(predictor_2d, predictor_3d)
    
    video_filename=args.video_path.split('/')[-1].split('.')[0]
    log_filename=f'Log_{args.predictor_2d}_{args.predictor_3d}_{video_filename}.csv'
    pose_logger = PoseLogger(args.port, filename=log_filename)

    logging.info("Predicting %s with %s 2d pose detector and %s 3d pose detector", args.input, args.predictor_2d, args.predictor_3d)

    inference(videoCapture, human_pose_estimator, pose_logger)
    if args.viz==True:
        Visualization(args.video_path, log_filename)

    