# MotionBert - Installation instructions

To use MotionBert in this library, you have to follow the following steps.

Copy in this folder the model script `DSTformer.py` which is in [this link](https://github.com/Walter0807/MotionBERT/blob/main/lib/model/DSTformer.py). Then, create both folders lib and model and place there `drop.py` [this link](https://github.com/Walter0807/MotionBERT/blob/main/lib/model/drop.py). Additionally, you need to download the model weights 39.1mm (MPJPE) from [this link](https://github.com/Walter0807/MotionBERT) and copy it in `/models/best_epoch.bin`
The resulting folder structure should be as follows:

```
BodyFlow
│   README.md
|   ...    
│───models
│     | best_epoch.bin
│
│
└───src
    │───human_pose_estimation
            └───models
                    └───predictors_3d
                            └───motionbert
                                    └───lib
                                         └───model 
                                               └───drop.py
                                    │  model.py
                                    │  MotionBert_installation.md

```

Then, to run the code simply use the following parameter when using `src/pose-estimation/inference_server.py`:



`$ python src/pose-estimation/inference_server.py --predictor_3d motionbert`