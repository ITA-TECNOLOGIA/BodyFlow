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
import os
from tqdm import tqdm

def save_frames(video_path, start_time, end_time, output_prefix):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the corresponding frame numbers based on time
    fps = video.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Check if the start and end frames are within the valid range
    if start_frame < 0 or end_frame >= total_frames:
        print("Invalid frame range.")
        return
    
    # Set the current frame to the starting frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create the output folder if it does not exist
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
    
    # Initialize tqdm
    pbar = tqdm(total=end_frame - start_frame + 1)
    
    # Iterate through the frames and save them
    frame_count = start_frame
    while frame_count <= end_frame:
        # Read the current frame
        ret, frame = video.read()
        
        if not ret:
            print("Error reading the frame.")
            break
        
        # Save the frame as an image
        output_path = os.path.join(output_prefix, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(output_path, frame)
        
        frame_count += 1
        pbar.update(1)
    
    # Release the video capture object
    video.release()
    
    # Close tqdm
    pbar.close()

# Example usage
video_path = "data/demos/videos/gp4_006.MP4"
start_time = 4   # Start time in seconds
end_time = 10     # End time in seconds
output_folder = "data/demos/gp4_006"
save_frames(video_path, start_time, end_time, output_folder)



