# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnologico de Aragon (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@itainnova.es
#          Angel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Maranes Nueno                            cmaranes@itainnova.es
#          Irene Lopez Bosque                                ilopez@itainnova.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibanez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocio Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------

import os
import sys
import argparse
import logging
import torch
from datetime import datetime

# Custom codes
from dataset import HumanActivityDataset
from models.lstm_model.lstm_model import MulticlassLSTMModel
from models.cnn_model.cnn_model import CNNModelMulticlass
from models.transformer_model.transformer_model import TransformerModel
from utils.util import get_df, get_X_y_datasets, plot_confusion_matrix, auto_logged_info

# MLFlow for output manager
import mlflow
import mlflow.pytorch

# Pytorch Lightning for training tracking
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.memory import get_model_size_mb

def instance_har_model(args, num_classes):
    """
    Returns a Human Activity Recognition Model
    """
    har_model = None
    if args.har_model == "lstm":
        har_model = MulticlassLSTMModel(input_features=args.features[1], window_length = args.window_length, num_classes=num_classes)
    elif args.har_model == "cnn":
        har_model = CNNModelMulticlass(input_features=args.features[1], window_length = args.window_length, num_classes=num_classes)  
    elif args.har_model == "transformer":
        har_model = TransformerModel(input_features=args.features[1], window_length=args.window_length, num_classes=num_classes,
                                     embed_size = 128, num_layers = 4, heads = 4, forward_expansion = 1024, dropout = 0.2)  
    else:
        sys.exit(f"HAR model {args.har_model} not implemented!")
    return har_model


def train(args):
    num_classes, df_train, df_test = get_df(args)
    
    X_train, y_train, X_test, y_test = get_X_y_datasets(df_train, df_test, args)


    train_dataset = HumanActivityDataset(X=X_train, Y=y_train)

    # use 20% of training data for validation
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator = seed)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch, shuffle = True, 
                                                   pin_memory = False, num_workers = args.workers) 
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = args.batch, shuffle = False, 
                                                   pin_memory = False, num_workers = args.workers) 

    test_dataset = HumanActivityDataset(X = X_test, Y = y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch, shuffle = False,
                                                  pin_memory = False, num_workers = args.workers) 

    # Set a custom name for the checkpoints that Lightning saves during training
    checkpoint_folder = f'{args.har_model}_{args.features[0]}_{args.dataset}'
    checkpoint_callback = ModelCheckpoint(filename = os.path.join('src', 'human_activity_recognition', checkpoint_folder)) 

    # Pytorch Lightning Trainer
    trainer = pl.Trainer(devices = [args.gpu], accelerator = 'gpu', max_epochs = args.epochs, 
                         check_val_every_n_epoch = 1, callbacks = checkpoint_callback, 
                         default_root_dir = os.path.join('src', 'human_activity_recognition'))
    
    


    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    with mlflow.start_run( run_name = f'{args.har_model}_{args.features[0]}') as run: #, experiment_id='binaries'
        mlflow.set_tags({"type": "train"})

        model = instance_har_model(args, num_classes = num_classes)

        logging.info(f'Model size: {get_model_size_mb(model)}')

        trainer.fit(model, train_dataloader, valid_dataloader)
        predictions = trainer.predict(model, dataloaders = test_dataloader)

        # As predictions are given by batch, you need to concat them all
        predictions = torch.cat(predictions)
        # Transform One-Hot format into int format
        y_pred = predictions.argmax(axis=1) 

        y_true = []
        for x, y in test_dataloader:
            y_true.append(y)
        y_true = torch.cat(y_true)
        y_true = y_true.argmax(axis=1)

        plot_confusion_matrix(y_true, y_pred, args)
        
    auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    # Generate csv with the activity predictions added to the test dataset
    df_test['activity_prediction'] = y_pred
    df_test['activity_prediction'] += 1
    df_test.to_csv('output_activity_predictions.csv', index=False)




if __name__ == '__main__':

    ########################################
    ####          Logging config        ####
    ########################################

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logger_{timestamp}.log", level=logging.DEBUG)
    # Avoid logging info from matplotlib
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('fsspec.local').disabled = True

    ########################################
    ####            Parameters          ####
    ########################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer', type=str, default='cnn', help='HAR model: [lstm, cnn, transformer]')
    parser.add_argument('-i', '--item', default=['ankle', 12], action='store', dest='features', type=str, nargs='*', help="[features, number of features]:  '['all', 200] or ['2d', 68] or ['3d', 102] or ['imus', 30] or ['ankle', 6]'")
    parser.add_argument('--batch', type=int, default=64, help='Batch: [1, 64]')
    parser.add_argument('--epochs', type=int, default=24, help='Epochs: [8, 12, 24, 50]')
    #parser.add_argument('--path_dataset', type=str, default='/data/proyectos/sarcopenia/har-up-dataset/processed_harup.csv')
    # parser.add_argument('--path_dataset', type=str, default='/data/proyectos/sarcopenia/sarcopenia/data/ITA_dataset/version_auditorio/finals/processed_ita_cpn_mhformer.csv')
    parser.add_argument('--path_dataset', type=str, default='data/har_train/processed_ita_cpn_mhformer.csv')
    parser.add_argument('--label', type=str, default="activity", help='Label: [activity, tag]')
    parser.add_argument('--window_step', type=int, default=1, help='Window step: [1]')
    parser.add_argument('--window_size_past', type=int, default=20, help='Window size past: [10, 20, 40]')
    parser.add_argument('--window_size_future', type=int, default=20, help='Window size future: [10, 20, 40]')
    parser.add_argument('--window_length', type=int, default=41, help='window_length: [21, 41, 81]')
    #parser.add_argument('--num_classes', type=int, default=10, help='number of classes: [10, 11]')
    parser.add_argument('--workers', type = int, default = 4, help = 'Number of workers to feed the data')
    parser.add_argument('--gpu', type=int, default = 0, help='Cuda device: [0, 1, 2, 3]')
    parser.add_argument('--dataset', type=str, default = 'itainnova', help='Dataset: [harup, itainnova]')
 
    args = parser.parse_args()

    
    train(args)
