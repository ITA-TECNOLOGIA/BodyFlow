# BodyFlow

<img src="figures/AI4HealthyAging_logo.png" alt="AI4HealthyAging Logo" width="200"/> <img src="figures/itainnova_logo.png" alt="Itainnova Logo" width="200"/>

BodyFlow is a comprehensive library that leverages cutting-edge deep learning and other AI techniques, including new algorithms developed by the AI group at ITAINNOVA, to accurately estimate human pose in 2D and 3D from videos. With its state-of-the-art algorithms, BodyFlow can detect events such as falls and walking, and in the future, the aim is to further expand its capabilities by developing classifiers for certain neurodegenerative diseases. The use of deep learning and advanced AI methods, combined with the innovative algorithms developed by the ITAINNOVA AI group, makes BodyFlow a highly sophisticated and effective tool for analyzing human motion and detecting important events.

The first release of this library contains three 2D detectors (MediaPipe2D, CPN, Lightweight) and two 3D dectectors (Videopose3D, MHFormer, MixSTE, MotionBert) for predicting Human Pose from a set of monocular RGB images in a video sequence. The code from the original works have been refactored in a way that make them easy to manipulate and combine, since the methods used in this project are those which use 2d lifting, this is, first a 2d pose estimator is used and then it is lifted with other algorithm to the final 3d pose. It is possible to add new 2d and 3d pose estimation algorithms if needed.

The code from the original works have been refactored in a way that make them easy to manipulate and combine, since the methods used in this project are those which use 2d lifting, this is, first a 2d pose estimator is used and then it is lifted with other algorithm to the final 3d pose. It is possible to add new 2d and 3d pose estimation algorithms if needed.

The library outputs a csv with the 2d and 3d landmarks. They can be visualized using the [visualization](src/pose-estimation/common_pose/visualization.py) script.

## Installation
The code has been tested with **Python 3.7**. Other Python versions may work, but they have not been tested.

We start by creating a conda environment with the following command:

`$ conda create -n misiones-env python=3.7`

`$ conda activate misiones-env`

Then, you should install the needed libraries (dependencies) which are defined in the requirements file. To do so, run the following command in the prompt:

`$ pip install -r requirements.txt`

Additionally, the installation of **PyTorch** library is needed. Please refer to the official website [Pytorch Webpage](https://pytorch.org/) to obtain the correct PyTorch version according to your CUDA version. You may know the CUDA version by running `nvcc --version` in the prompt.

*Note: The command `nvidia-smi` also provides the CUDA version, and it can be different from the previous command, so this might be the real CUDA version.*

### Downloading the models

All the models weights files have been wrapped and may be downloaded by executing the following command:

`$ python src/pose-estimation/model_downloader.py`

*Note: If you have permission issues, you might need to excecute the above line with `sudo`.*

### Problems & Solutions

1. If there is an error raised by mediapipe, the following command can solve it `$ pip install --upgrade protobuf==3.20.0`

## Running the code

The main script to run the code is located in `src/pose-estimation/inference_server.py`.
To run the code, it is needed to select a 2d and 3d pose estimation predictor and the input data type.
Available 2d predictors are:
1. Mediapipe - (*Included*)
2. Cascade Pyramid Network (CPN) - (*Included*)
3. Lightweight - (*Included*)

Available 3d predictors are:
1. MHFormer - (*Included*)
2. VideoPose3D - (*[Installation instructions](src/pose-estimation/models/predictors_3d/videopose/VideoPose3D_installation.md)*)
3. MixSTE - (*[Installation instructions](src/pose-estimation/models/predictors_3d/mixste/MixSTE_installation.md)*)
4. MotionBert - (*[Installation instructions](src/pose-estimation/models/predictors_3d/motionbert/MotionBert_installation.md)*)

They are indicated in the following form:

`$ python src/pose-estimation/inference_server.py --predictor_2d {mediapipe2d, cpn, lightweight} --predictor_3d {mhformer, motionbert, mixste}`

For example, to run CPN and MHFormer:

`$ python src/pose-estimation/inference_server.py --predictor_2d cpn --predictor_3d mhformer`

The input type can be a .mp4 video, an orderer set of images in format .png or video captured directly with the webcam. In the following examples we show how to run the code which each input type with CPN and MHformer combination.

### Video .mp4
The video must be accessible and the route has to be indicated as follows:

`$ python src/pose-estimation/inference_server.py --predictor_2d cpn --predictor_3d mhformer --input video --video_path route/to/video.mp4`

### Images .png
The folder must contain the images as follows 000001.png, 000002.png, 000003.png, etc. Then, the folder has to be passed as argument as follows:

`$ python src/pose-estimation/inference_server.py --predictor_2d cpn --predictor_3d mhformer --input pictures --pictures_path route/to/pictures`

### Webcam
You have to know the camera number device that will use opencv to access it. If you do not know and do not have any other device connected to the laptop, then run the following code:

`$ python src/pose-estimation/inference_server.py --predictor_2d cpn --predictor_3d mhformer --input cam --video_source 0`

## Output
The output is a .csv which contains the 2d and 3d landmarks per each frame. It is located in the folder `logs`. Additonally, if you input a video, and set visualization to True, a video is created in the folder `data/output` with the 2D and 3D output.

##  

These repositories are used to extend our code. We appreciate the developers sharing the codes.


- [Lightweight](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
- [Deep High-Resolution Representation Learning for Human Pose Estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [VideoPose3d](https://github.com/facebookresearch/VideoPose3D)
- [MixSTE](https://github.com/JinluZhang1126/MixSTE)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)

## Copyright
```
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
```

## License

Please refer to the LICENSE in this folder.

## Acknowledgement

BodyFlow was funded by project MIA.2021.M02.0007 of NextGenerationEU program and Integration and Development of Big Data and Electrical Systems (IODIDE) group of Aragon Goverment program.