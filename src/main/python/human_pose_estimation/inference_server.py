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

import argparse
import os
import time
import sys
import signal
import logging
from datetime import datetime
import cv2
import platform
import tempfile
import torch

if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

if __name__ == "__main__":
    sys.path.append(os.path.join('src', 'main', 'python'))

sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'mixste'))
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'motionbert'))
sys.path.append(os.path.join('src', 'main', 'python', 'human_pose_estimation', 'models', 'predictors_3d', 'expose_lib'))

# Pose Sender
from human_pose_estimation.common_pose.PoseLogger import PoseLogger

# Video capture
from human_pose_estimation.video_capture.VideoCapture import VideoCapture
from human_pose_estimation.video_capture.VideoFromVideo import VideoFromVideo
from human_pose_estimation.video_capture.VideoFromImages import VideoFromImages
from human_pose_estimation.video_capture.VideoFromCam import VideoFromCam

# Pose Predictors from RGB to 2D
from human_pose_estimation.models.HPE2D import HPE2D
from human_pose_estimation.models.predictors_2d.Dummy2D import Dummy2D
from human_pose_estimation.models.predictors_2d.Mediapipe2D import Mediapipe2D
from human_pose_estimation.models.predictors_2d.CPN import CPN
from human_pose_estimation.models.predictors_2d.Lightweight2D import Lightweight2D

# Pose Predictors from 2D to 3D
from human_pose_estimation.models.HPE3D import HPE3D
from human_pose_estimation.models.predictors_3d.Dummy3D import Dummy3D
from human_pose_estimation.models.predictors_3d.MHFormer import MHFormer

# Person detector and tracker
from human_pose_estimation.models.PersonDetector import PersonDetector
from human_pose_estimation.models.person_detector.YoloV3 import YoloV3
from human_pose_estimation.models.Tracking import Tracking
from human_pose_estimation.models.tracking.Single import Single
from human_pose_estimation.models.tracking.Sort import Sort
from human_pose_estimation.models.tracking.DeepSort import DeepSort
from human_pose_estimation.models.tracking.ByteTrack import ByteTrack

# Pose Predictor from RGB to 3D
from human_pose_estimation.models.HPE import HPE

def instance_video_capture(input='pictures',path=None,infinite_loop=False) -> VideoCapture:
    """
    Returns a video capture which provides the frames to estimate the pose.
    """
    videoCapture_instance = None
    if input == 'video':
        videoCapture_instance = VideoFromVideo(path, infinite_loop)
    elif input == 'pictures':
        videoCapture_instance = VideoFromImages(path, infinite_loop)
    elif input == 'cam':
        videoCapture_instance = VideoFromCam(path)
    else:
        sys.exit(f"Input {input} not recognized/implemented.")
    
    return videoCapture_instance


def instance_predictor_2d(predictor_2d, developer_parameters) -> HPE2D:
    """
    Returns a 2D human pose predictor. Given an RGB image, it returns the 3d pose.
        - dummy2d: Place holder. It always return the same value. This is done for dubbing purpuses.
        - mediapipe2d: Included. Fastest one.
        - cpn: Included.
        - lightweight: Included.
    """
    predictor_2d_instance = None
    if predictor_2d == "dummy2d":
        predictor_2d_instance = Dummy2D()
    elif predictor_2d == "mediapipe2d":
        predictor_2d_instance = Mediapipe2D()  # TODO not running in GPU
    elif predictor_2d == "cpn":
        predictor_2d_instance = CPN(developer_parameters)
    elif predictor_2d == "lightweight":
        predictor_2d_instance = Lightweight2D(developer_parameters)
    else:
        sys.exit(f"2D detector {predictor_2d} not implemented!")
    return predictor_2d_instance


def instance_predictor_3d(predictor_3d, window_length, path, developer_parameters) -> HPE3D:
    """
    Returns a 3D human pose predictor. Given a 2D estimated human pose, it lifts it to a 3D one.
        - dummy3d: Place holder made for debugging purposes. Included.
        - mhformer: Included.
    IMPORTANT:
        In order to use videopose, motionbert and mixste you need to download additional files 
        and model weights. Please refer to README.md for installation instrucctions.
    """
    predictor_3d_instance = None
    if predictor_3d == "dummy3d":
        predictor_3d_instance = Dummy3D(window_length=1)
    elif predictor_3d == "mhformer":
        predictor_3d_instance = MHFormer(window_length=window_length,
                                         developer_parameters=developer_parameters)
    elif predictor_3d == "videopose":     
        from human_pose_estimation.models.predictors_3d.VideoPose3D import VideoPose3D   
        predictor_3d_instance = VideoPose3D(window_length=243,
                                            developer_parameters=developer_parameters)
    elif predictor_3d == "mediapipe3d":
        from human_pose_estimation.models.predictors_3d.Mediapipe3D import Mediapipe3D
        logging.info("End-to-end model, please use dummy2d as 2d predictor")
        predictor_3d_instance = Mediapipe3D(window_length=1)
    elif predictor_3d == "motionbert": 
        from human_pose_estimation.models.predictors_3d.MotionBert import MotionBert       
        predictor_3d_instance = MotionBert(window_length=window_length,
                                           developer_parameters=developer_parameters)
    elif predictor_3d == 'mixste': 
        from human_pose_estimation.models.predictors_3d.MixSTE import MixSTE    
        predictor_3d_instance = MixSTE(window_length=window_length,
                                       developer_parameters=developer_parameters)
    elif predictor_3d == 'expose':
        from human_pose_estimation.models.predictors_3d.ExPose import ExPose 
        logging.info("End-to-end model, please use dummy2d as 2d predictor") 
        predictor_3d_instance = ExPose(window_length=1,
                                       developer_parameters=developer_parameters,
                                       video_filename=path.split('/')[-1].split('.')[0])
    
    else:
        sys.exit(f"3D detector {predictor_3d} not implemented!")
        
    return predictor_3d_instance

def instance_person_detector(person_detector, developer_parameters) -> PersonDetector:
    person_detector_instance = None
    if person_detector == "yolov3":
        person_detector_instance = YoloV3(developer_parameters['models_path'])
    else:
        sys.exit(f"Person detector {person_detector} not implemented!")
    return person_detector_instance


def instance_tracking(tracking_name, max_age) -> Tracking:
    tracking = None
    if tracking_name == "single":
        tracking = Single(max_age)
    elif tracking_name == "regular":
        tracking = Sort(max_age = max_age)
    elif tracking_name == "deepSort":
        tracking = DeepSort(max_age = max_age)
    elif tracking_name == "bytetrack":
        tracking = ByteTrack(max_age = max_age)
    else:
        sys.exit(f"Tracker {tracking_name} not implemented!")
    return tracking


def draw_bbox(frame, body_landmarks):
    if body_landmarks is None:
        return frame
    line_width = 2
    font_size = 2
    for person_id, data in body_landmarks.items():
        if data._bodyLandmarks2d._bbox is not None and data._bodyLandmarks2d.repeated == False:
            bbox = [int(x) for x in data._bodyLandmarks2d._bbox]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (245, 180, 4), line_width)
            (w, h), _ = cv2.getTextSize(f"id: {person_id}", cv2.FONT_HERSHEY_SIMPLEX, font_size, line_width)
            cv2.rectangle(frame, (bbox[0]-5, bbox[1]+5), (bbox[0] + w+5 ,bbox[1]- h-5), (245, 180, 4), -1)
            cv2.putText(frame, f"id: {person_id}", (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), line_width)
    return frame

def inference(videoCapture: VideoCapture, human_pose_estimator: HPE, pose_logger: PoseLogger, export_csv = True, output_path='out', bboxes_viz = False):
    """
    Initialization of the buffers of the human pose estimator. Once initialized, it computes the 
    3D human pose in real time, sending the pose through a socket if specified.
    If the script is stopped with CTRL+C, it handles the signal, saves the poses in a .csv and
    finish the script.
    """
    # TODO remove this line:
    # os.makedirs("out", exist_ok=True)
    # Initialize HPE, e.g., load buffer
    logging.debug("Initializing HPE...")
    while True:
        # Read frame
        frame, timestamp = videoCapture.get_frame()
        if videoCapture.video_finished():
            logging.error("HPE not initlialized. May not enough frames.")
            break
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
        #if counter > 1000:
        if videoCapture.video_finished():
            logging.debug("Input finished, destroying buffer and making last predictions...")
            while True:
                body_landmarks = human_pose_estimator.destroy_all_buffers()
                if body_landmarks is None:
                    break
                else:
                    # For debugging
                    real_timestamp = list(body_landmarks.values())[0].timestamp
                    debug_img = draw_bbox(videoCapture.get_frame_by_timestamp_and_delete(real_timestamp), body_landmarks)
                    if bboxes_viz == True:
                        cv2.imwrite(os.path.join(output_path, "%06d.png") % counter, debug_img)
                    counter += 1
                    pose_logger.log_pose(body_landmarks)
            break

        # Compute pose (inference)
        body_landmarks, real_timestamp = human_pose_estimator.predict_pose()
        

        # For debugging
        debug_img = draw_bbox(videoCapture.get_frame_by_timestamp_and_delete(real_timestamp), body_landmarks)
        if bboxes_viz == True:
            aspect_ratio = debug_img.shape[1] / debug_img.shape[0]
            target_width = 600
            target_height = int(target_width / aspect_ratio)
            debug_img = cv2.resize(debug_img, (target_width, target_height))
            cv2.imwrite(os.path.join(output_path, "%06d.png") % counter, debug_img)
        pose_logger.log_pose(body_landmarks)

        # Add next frame to the buffer
        human_pose_estimator.add_frame(frame, timestamp)

        end = time.time()
        if counter % 30 == 0: logging.debug("Frame %d -> FPS: %2.f", counter, 1 / (end - start))
        counter += 1
        logging.info(counter)
    
    logging.info("Pose estimation finished!")
    if export_csv:
        pose_logger.export_csv()
    return True


def run_inference(predictor_2d = 'cpn',
                  predictor_3d = 'mhformer',
                  person_detector = 'yolov3',
                  tracking = 'single',
                  max_age = 30,
                  gpu = '0',
                  infinite_loop = False,
                  port = -1,
                  window_length = 243,
                  input = 'pictures',
                  path = None,
                  viz = 1,
                  output_path = '',
                  logo1 = 'figures/ITA_Logo.png',
                  logo2 = 'figures/AI4HealthyAging_logo.png',
                  logo3 = 'figures/ITA_Logo.png',
                  output_resolution = '720p',
                  timestamp = '', 
                  developer_parameters = None,
                  bboxes_viz = False):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"GPU is available. Device: {device}")
    else:
        logging.info("GPU is not available. Running on CPU.")


    resolutions = {
        "480p": "720x480",
        "576p": "720x576",
        "720p": "1280x720",
        "1080p": "1920x1080",
        "1440p": "2560x1440",
        "2K": "2048x1080",
        "4K": "3840x2160",
        "8K": "7680x4320"
    }

    if developer_parameters == None:
        developer_parameters = {   'models_path' : "/models" }   

    
    videoCapture = instance_video_capture(input,path,infinite_loop)

    predictor_2d_instance = instance_predictor_2d(predictor_2d, developer_parameters)
    predictor_3d_instance = instance_predictor_3d(predictor_3d,window_length, path, developer_parameters)
    person_detector = instance_person_detector(person_detector, developer_parameters)
    tracking = instance_tracking(tracking, max_age)
    
    human_pose_estimator = HPE(predictor_2d_instance, predictor_3d_instance, predictor_3d_instance.window_length, person_detector=person_detector, tracking=tracking)
    
    if input == 'video':
        video_filename=path.split('/')[-1].split('.')[0]
        log_filename=f'{output_path}/Log_{predictor_2d}_{predictor_3d}_{video_filename}.csv'
    elif input == 'pictures':
        video_filename=path.split('/')[-1].split('.')[0]
        log_filename=f'{output_path}/Log_{predictor_2d}_{predictor_3d}_{video_filename}.csv'
    
    pose_logger = PoseLogger(port, filename=log_filename)

    logging.info("Predicting %s with %s 2d pose detector and %s 3d pose detector", input, predictor_2d, predictor_3d)

    
    
    temp_dir = f"out_{timestamp}"
    if bboxes_viz == True:
        os.makedirs(temp_dir, exist_ok=True)    
        logging.info(f'Temporal dir {temp_dir}')
        
    inference(videoCapture, human_pose_estimator, pose_logger, output_path = temp_dir, bboxes_viz = bboxes_viz)
    
    if bboxes_viz == True:
        try:
            command = f"ffmpeg -r {videoCapture.fps} -f image2 -s " + resolutions[output_resolution] + " -i " + temp_dir + "/\%06d.png -vcodec libx264 -crf 25  " +  output_path +  "\\" +video_filename +'_processed' + ".mp4 -y"
            #command = "ffmpeg -r 30 -f image2 -s 1280x720 -i " + "out/" + "\%06d.png -vcodec libx264 -crf 25 " + "out" + "\\" + 'video' + ".mp4 -y"
            os.system(command)
        except Exception as e:
            logging.info(f"An error occurred: {e}")
        

    devs = False
    if viz != None:
        assert viz > 0
        if predictor_3d == 'expose':
            from models.predictors_3d.expose_lib.expose_utils import visualize_expose
            visualize_expose(video_filename, input, path, log_filename)

        if input in ['video', 'pictures']:
            from human_pose_estimation.visualization.visualization import Visualization
            Visualization(path, input, pose_logger._filename, person_id=viz, output_path = output_path, logo1 = logo1, logo2 = logo2, logo3 = logo3, timestamp = timestamp, devs = devs)
        elif input == 'cam':
            logging.info("Visualization skipped!: Unavailable for this input format.")
        else:
            raise NotImplementedError(f"Visualization not available for input type {input}")

                
    # Change the log name 
    new_filename = os.path.join(output_path, pose_logger._filename)
    if False:
    #if devs == False:
        old_filename = os.path.join(output_path, pose_logger._filename)
        new_filename = old_filename.replace(predictor_3d, '')
        new_filename = new_filename.replace(predictor_2d, '')
        if predictor_3d == 'mediapipe3d' or predictor_3d == "expose":
            new_filename = new_filename.replace('___', '_algorithm_')
        else:
            new_filename = new_filename.replace('___', '_')
        if os.path.exists(os.path.join(output_path, pose_logger._filename)):
            try:
                os.rename(old_filename, new_filename)
                logging.info(f"File renamed from '{old_filename}' to '{new_filename}'.")
            except OSError as e:
                pass
        else:
            pass
        

    if bboxes_viz == True:
        # Loop through the files and remove them
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)        
        os.rmdir(temp_dir)

    viz_name =f'viz_{video_filename}.mp4'
    try:
        os.symlink(path, os.path.join(output_path, path.split('/')[-1]))
        logging.info(f"Enlace simbólico creado: {os.path.join(output_path, path.split('/')[-1])}")
    except OSError as e:
        logging.info(f"No se pudo crear el enlace simbólico: {e}")
        
    return { 
        "log_filename": os.path.basename(new_filename),
        "viz_filename": viz_name,
        "processed_filename":  path.split('/')[-1]
    }    
    


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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    logging.basicConfig(filename=f"logger_{timestamp}.log", level=logging.DEBUG)
    
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_2d', type = str, default = 'dummy2d', help = '2D Predictor: [dummy2d, mediapipe2d, cpn, lightweight]')
    parser.add_argument('--predictor_3d', type = str, default = 'expose', help = '3D Predictor: [dummy3d, mhformer, mediapipe3d] if downloaded: [videopose, motionbert, mixste, expose]')
    parser.add_argument('--person_detector', type = str, default='yolov3', help = 'Person detector: [yolov3]')
    parser.add_argument('--tracking', type = str, default = 'regular', help = 'Multi-person tracking: [single, regular, deepSort, bytetrack], single is no multi-person tracking')
    parser.add_argument('--max_age', type = int, default = 180, help = 'Tracker buffer max age')
    parser.add_argument('--gpu', type = str, default = '0', help = 'Id of the GPU. Empty string if CPU')
    parser.add_argument('--infinite_loop', action = 'store_true')
    parser.add_argument('--port', type = int, default = -1, help = 'Port the machine where the inference server is running. The Unity server will connect to this Port. -1 and pose will not be sent.')
    parser.add_argument('--window_length', type = int, default = 81, help = 'Available window length for mhformer and mixste: [81, 243]')

    # ----- Input
    parser.add_argument('--input', default='video', choices=['video', 'pictures', 'cam'], help='Way in which the entry is read')
    
    # Video ,Pictures or cam
    parser.add_argument('--path', type=str, default='data/demos/videos/VID20240327100723.mp4', help='Path of the pictures or Video to estimate the pose or Cam source() It can also be a rstp url)')
    parser.add_argument('--output_path', type=str, default='.', help='Path to write the results')

    # Visualization
    #parser.add_argument('--viz', type=int, nargs='?', const=1, default=None, help='To plot the visualization for a specific person ID, pass the ID (e.g., --viz 1 for person 1). If --viz is provided without a value, no visualization is performed. Visualization is done after the inference is done.')
    parser.add_argument('--bboxes_viz', type=bool, default = True, help = 'To plot the visualization while running of the bboxes and IDs.')
    parser.add_argument('--viz', type=int, default = 1, help = 'To plot the visualization while running of the bboxes and IDs.')

    
    args = parser.parse_args()
    logging.info(f'Arguments: {args}')
    developer_parameters = {
        'models_path' : "models"
    }
    
    # EN OUTPUT PATH VOLCAR TODOS LOS RESULTADOS
    out_bf = run_inference(predictor_2d=args.predictor_2d,
                  predictor_3d=args.predictor_3d,
                  person_detector=args.person_detector,
                  tracking=args.tracking,
                  max_age=args.max_age, 
                  gpu=args.gpu,
                  infinite_loop=args.infinite_loop,
                  port=args.port,
                  window_length=args.window_length,
                  input=args.input,
                  path=args.path,
                  viz=args.viz,
                  output_path=args.output_path,
                  timestamp = timestamp,
                  developer_parameters = developer_parameters,
                  bboxes_viz = args.bboxes_viz)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f'Elapsed time: {elapsed_time}')


    