# VideoPose3D - Installation instructions

To use VideoPose3D in this library, you have to follow the following steps.

Copy in this folder the model script which is in the [original repository](https://github.com/facebookresearch/VideoPose3D/blob/1afb1ca0f1237776518469876342fc8669d3f6a9/common/model.py#L1). Additionally, you need to download the model weights from [this link](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin) and copy it in `/models/pretrained_h36m_detectron_coco.bin`
The resulting folder structure should be as follows:

```
BodyFlow
│   README.md
|   ...    
│───models
│     | pretrained_h36m_detectron_coco.bin
│
│
└───src
    │───main
        │───python
                │───human_pose_estimation
                        └───models
                                └───predictors_3d
                                        └───videopose
                                                │  model.py
                                                │  VideoPose3D_installation.md

```

Then, to run the code simply use the following parameter when using `src/main/python/human_pose_estimation/inference_server.py`:



`$ python src/main/python/human_pose_estimation/inference_server.py --predictor_3d videopose`