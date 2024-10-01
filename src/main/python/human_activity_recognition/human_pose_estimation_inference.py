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

import os
import io
import argparse
import logging
import torch
from datetime import datetime

# Custom codes
from dataset.human_pose_estimation_dataset import HumanPoseEstimationData
from utils.video_viz import create_video

# Pytorch Lightning for training tracking
import pytorch_lightning as pl

pl.seed_everything(1)

TRAINED_MODELS = {
    'input_features':
    {
        '2d': 34,
        '3d': 51,
        'all': 34+51
    },
    'num_classes': 11,
    'window_size': 41
}

def inference(args):
    data = HumanPoseEstimationData(args)

    # Make sure feature order is the same as in the training data
    X, input_features = data.fetch_data()    
    y = torch.zeros(X.shape[0], dtype=torch.float32) # Dummy

    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch, shuffle = False,
                                                  pin_memory = False, num_workers = args.workers) 

    assert input_features == TRAINED_MODELS['input_features'][args.input_data]

    #model_name = f'har_{args.har_model}_{args.input_data}_{args.window_size}.ckpt' #har_cnn_ankle_41.ckpt
    model_name = f"{args.har_model}_{args.input_data}.pth"

    model_file_path = os.path.join("src/main/python/human_activity_recognition/weights", model_name)

    with open(model_file_path, 'rb') as file:
        buffer = io.BytesIO(file.read())
    model = torch.load(buffer)
    model.eval()
    model.freeze()

    predictor = pl.Trainer(devices = [args.gpu],
                           accelerator = 'gpu',
                           logger = None)
    
    predictions = predictor.predict(model, dataloaders = dataloader)

    y_pred = torch.cat(predictions)
    y_pred = y_pred.argmax(axis=1)
    output_path = data.write_har(y_pred.cpu().numpy())

    if args.render_video is None:
        return
    
    create_video(output_path, args.render_video, args.viz)

if __name__ == '__main__':

    ########################################
    ####          Logging config        ####
    ########################################

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logger_{timestamp}.log", level=logging.INFO)
    logging.getLogger('mlflow').setLevel(logging.INFO)
    logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('fsspec.local').disabled = True

    ########################################
    ####            Parameters          ####
    ########################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--har_model', type = str, default = 'lstm', help = 'HAR model: [lstm, cnn, transformer]')
    parser.add_argument('--input_data', type = str, default = 'all', help = '[all, 2d, 3d]')
    parser.add_argument('--batch', type = int, default = 64, help = 'Batch size')
    parser.add_argument('--path_dataset', type = str, default = 'logs/Log_cpn_mhformer_person_walking.csv') 
    parser.add_argument('--window_step', type = int, default = 1, help = 'Window step: [1]')
    parser.add_argument('--window_size', type = int, default = 21, help = 'window_size: [21, 41, 81]')
    parser.add_argument('--workers', type = int, default = 64, help = 'Number of workers to feed the data')
    parser.add_argument('--gpu', type = int, default = 0, help = 'Cuda device: [0, 1, 2, 3]')
    
    parser.add_argument('--render_video', type = str, default= None, help = 'Video path to overlap the activity in each frame')
    parser.add_argument('--viz', type = int, default = 0, help = 'Id of the user predictions to plot in the video')


    args = parser.parse_args()
    logging.info(args)

    inference(args)

    
