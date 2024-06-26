# Human Activity Recognition module instructions

The HAR module may be used independently with pretrained or custom models. To use this module the UPFALL dataset must be available and synched please refer to the *[UPFALL_synchronization instructions](../synchronization/Synchronization_UPFALL_instruccions.md)*. However, we have already released the preprocessed data ready to be trained. The data can be found in the following [link](https://argon-docker.itainnova.es/repository/war/bodyflow/HAR/dataset/processed_harup.csv). You have to download the file `processed_harup.csv` and copy it in `/data/har/processed_harup.csv`. You may have to create the corresponding route.


To download the models already trained with vision features to predict the activity of the pose computed with the human_pose_estimation module, we already released the models in the following [link](https://argon-docker.itainnova.es/repository/war/bodyflow/HAR/models/HAR_models.zip). The models should be copied as follows:

```
src/main/python/human_activity_recognition
                        |
                        │
                        └───weights
                              | cnn_2d.pth
                              │ cnn_3d.pth
                              | cnn_all.pth
                              | lstm_2d.pth
                              | lstm_3d.pth
                              | lstm_all.pth
                              | transformer_2d.pth
                              | transformer_3d.pth
                              | transformer_all.pth
```

## Running the code

### Training

To train or test (TESTING FEATURE) the model, navigate to the project root directory and execute main.py with the desired arguments:

`$ python src/human_activity_recognition/main.py [ARGUMENTS]`

Below are the arguments you can specify when running main.py. We DO not recommend changing the parameters dataset, training, test, and tuning, since they are testing features still under development:

- --har_model: Specifies the type of model to use for HAR. Options include lstm, cnn, and transformer. The default is transformer.

- --input_data: Determines the type of input data to be used. Valid options are all, 2d, 3d, Imus, and ankle. Default is 3d.

- --batch: Sets the batch size for training or testing. The default is 64.

- --epochs: The number of training epochs. The default is 100.

- --path_dataset: The file path to the dataset. The default is `data/har/processed_harup.csv`.

- --label: Defines the label to be used. Options are activity and tag. Default is activity.

- --window_step: The step size of the window. The default is 1.

- --window_size: The size of the window. Valid options are 21, 41, 81. The default is 41.

- --workers: The number of worker threads to use for data loading. The default is 64.

- --gpu: Specifies the CUDA device to use. Options are 0, 1, 2, 3. The default is 0.

- --dataset: Defines the dataset to use. Options include harup, harup_vision, and itainnova. The default is harup_vision.

- --train: Enables training mode. The default is True.

- --test: Enables testing mode. The default is False.

- --tune: Activates the tuning of hyperparameters mode. The default is False.

### Inference

This guide details the process for running the Human Activity Recognition (HAR) inference pipeline, which leverages pose estimation data to predict activities in videos. The inference process utilizes output from the human_pose_estimation module and supports various models and input types.

Before running the HAR inference, ensure you have completed the human pose estimation process, which generates a .log file containing the poses of detected individuals. This file is typically located in the logs root folder.

#### Step 1: Locate the Pose Estimation Log File
Identify the .log file generated by the human_pose_estimation module. This file contains essential data for activity recognition.

#### Step 2: Execute the Inference Script (Optional)
With the log file path known, run the following command in your terminal:

`$ python src/main/python/human_activity_recognition/human_pose_estimation_inference.py --path_dataset LOG_PATH`

Replace LOG_PATH with the actual path to your .log file.

Optional Arguments: You can customize the inference process with the following optional arguments:

- --har_model: Choose the HAR model. Options: lstm, cnn, transformer (default: transformer).
- --input_data: Set the type of input data. Options: all, 2d, 3d, Imus, ankle (default: 3d).

#### Step 3: Execute the Inference Script andReview the Output
After running the script, the output will be saved as dataset_har.csv, enriching the input log with a new column that includes the predicted activity for each pose.


To overlay the predicted activities on the source video:
- --render_video: Path to the input video file. This should be the output from the human_pose_estimation module.
- --viz: ID of the bounding box (person) for which to display the activity. The default is 1.

Command example: 
`$ python src/main/python/human_activity_recognition/human_pose_estimation_inference.py --path_dataset LOG_PATH --render_video VIDEO_PATH --viz BOUNDING_BOX_ID`

Replace LOG_PATH, VIDEO_PATH, and BOUNDING_BOX_ID with your specific file paths and desired bounding box ID, respectively. The processed video, with activities overlaid, will be saved as `.har_video.mp4`. A frame of the output would look as follows. Note the top-left corner where it is the HAR prediction.

![HAR Prediction](../../../../figures/har_prediction.png)
