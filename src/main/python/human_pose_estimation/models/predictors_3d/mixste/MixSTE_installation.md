# MixSTE - Installation instructions

To use MixSTE in this library, you have to follow the following steps.

Copy in this folder the model script which is in [this link](https://github.com/JinluZhang1126/MixSTE/blob/main/common/model_cross.py). You also need `linearattention.py`, `rela.py` and `routing_transformer.py` from [common directory](https://github.com/JinluZhang1126/MixSTE/tree/main/common). Additionally, you need to download the model weights from the original repository [this link](https://drive.google.com/drive/folders/1G2mlMHebM6KcbI45FszlosIHgA4jiR3Y) and copy it in `/models/checkpoint_cpn_243f.bin`. You may also download the  `checkpoint_cpn_81f.bin` to change the window length. 
The resulting folder structure should be as follows:

```
BodyFlow
│   README.md
|   ...    
│───models
│     | best_epoch_cpn_243f.bin
│     | best_epoch_cpn_81f.bin
│
│
└───src
    │───main
        │───python
                │───human_pose_estimation
                        └───models
                                └───predictors_3d
                                        └───mixste
                                                └─── common
                                                        └─── linearattention.py
                                                        └─── rela.py
                                                        └─── routing_transformer.py
                                                │  model_cross.py
                                                │  MixSTE_installation.md

```

Then, to run the code simply use the following parameter when using `src/main/python/human_pose_estimation/inference_server.py`:



`$ python src/main/python/human_pose_estimation/inference_server.py --predictor_3d mixste`