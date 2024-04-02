# MotionBert - Installation instructions

To use MotionBert in this library, you need to download the model weights 37.2 mm (MPJPE) from [this link](https://onedrive.live.com/?authkey=%21AM16MxDQ4fEwZkI&id=A5438CD242871DF0%21171&cid=A5438CD242871DF0)(https://github.com/Walter0807/MotionBERT) and copy it in `/models/best_epoch.bin`
The resulting folder structure should be as follows:

```
BodyFlow
│   README.md
|   ...    
│───models
      | best_epoch.bin
```

Then, to run the code simply use the following parameter when using `src/main/python/human_pose_estimation/inference_server.py`:



`$ python src/main/python/human_pose_estimation/inference_server.py --predictor_3d motionbert`