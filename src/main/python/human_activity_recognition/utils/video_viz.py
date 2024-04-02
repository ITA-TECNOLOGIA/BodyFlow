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

import cv2
import pandas as pd
from tqdm import tqdm

def create_video(har_predictions_path: str, video_src: str, person_id: int):
    # Read the predictions from the CSV file
    predictions_df = pd.read_csv(har_predictions_path)
    
    # Filter predictions for the specified person_id
    predictions_df = predictions_df[predictions_df['id'] == person_id]
    if len(predictions_df) == 0:
        print(f"No video rendered. Person id: {person_id} is not in the dataframe.")
        return
    
    # Load the video
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    # Get video properties for output video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object to write the video
    out = cv2.VideoWriter('.har_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # Process the video and show progress with tqdm
    for frame_idx in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left to read
        
        # Check if there is a prediction for the current frame
        if frame_idx in predictions_df['timestamp'].values:
            # Get the prediction for the current frame
            prediction_row = predictions_df[predictions_df['timestamp'] == frame_idx]
            if not prediction_row.empty:
                prediction = prediction_row.iloc[0]['predictedHAR']  # Assuming one prediction per frame per person
                
                # Put the prediction on the frame
                cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write the modified frame to the output video
        out.write(frame)
    
    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
