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
from utils.util import get_df, get_X_y_datasets, plot_confusion_matrix

import mlflow

import pytorch_lightning as pl

def find_ckpt(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        


def instance_har_model(args):
    """
    Returns a Human Activity Recognition Model
    """
    har_model = None
    if args.har_model == "lstm":
        har_model = MulticlassLSTMModel(input_features=args.features[1], window_length = args.window_length, num_classes=args.num_classes)
    elif args.har_model == "cnn":
        har_model = CNNModelMulticlass(input_features=args.features[1], window_length = args.window_length, num_classes=args.num_classes)  
    elif args.har_model == "transformer":
        har_model = TransformerModel(input_features=args.features[1], window_length=args.window_length, num_classes=args.num_classes, embed_size = 128, num_layers = 4, heads = 4, forward_expansion = 1024, dropout = 0.2)  
    else:
        sys.exit(f"HAR model {args.har_model} not implemented!")
    return har_model




def main(model, args):
    num_classes, df_train, df_test = get_df(args)
    _ , _ , X_test, y_test = get_X_y_datasets(df_train, df_test, args)

    test_dataset = HumanActivityDataset(X = X_test, Y = y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch, shuffle = False, pin_memory = False, num_workers = 4) 
    
    # In Pytorch Lightning, a checkpoint is a version of the model. In this case we load the checkpoints representing the pretrained models.
    checkpoint_pretrained_model = find_ckpt(f"{args.har_model}_{args.features[0]}_{args.dataset}-v1.ckpt",  os.path.join('src', 'human_activity_recognition','lightning_logs'))
    pretrained_model = model.load_from_checkpoint(checkpoint_pretrained_model, input_features=args.features[1])

    # Pytorch Lightning Trainer
    trainer = pl.Trainer(devices=[args.cuda_device], accelerator="gpu", max_epochs = args.epochs, enable_checkpointing=False)
     # Auto log all MLflow entities

    mlflow.pytorch.autolog()
    with mlflow.start_run( run_name=f'{args.har_model}_{args.features[0]}') as run: #, experiment_id='binaries'
        mlflow.set_tags({"type": "test"})
        # Predict with the pretrained model
        predictions = trainer.predict(pretrained_model, dataloaders = test_dataloader)

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
    parser.add_argument('--har_model', type=str, default='cnn', help='HAR model: [lstm, cnn, transformer]')
    parser.add_argument('-i', '--item', default=['ankle', 12], action='store', dest='features', type=str, nargs='*', help="[features, number of features]:  '['all', 200] or ['2d', 68] or ['3d', 102] or ['imus', 30] or ['ankle', 6]'")
    parser.add_argument('--batch', type=int, default=64, help='Batch: [1, 64]')
    parser.add_argument('--epochs', type=int, default=24, help='Epochs: [8, 12, 24, 50]')
    #parser.add_argument('--path_dataset', type=str, default='/data/proyectos/sarcopenia/har-up-dataset/processed_harup.csv')
    #parser.add_argument('--path_dataset', type=str, default='/data/proyectos/sarcopenia/sarcopenia/data/ITA_dataset/version_auditorio/finals/processed_ita_cpn_mhformer.csv')
    parser.add_argument('--path_dataset', type=str, default='/data/ahernandez/lib-test/data/har_train/processed_ita_cpn_mhformer.csv')
    parser.add_argument('--label', type=str, default="activity", help='Label: [activity, tag]')
    parser.add_argument('--window_step', type=int, default=1, help='Window step: [1]')
    parser.add_argument('--window_size_past', type=int, default=20, help='Window size past: [10, 20, 40]')
    parser.add_argument('--window_size_future', type=int, default=20, help='Window size future: [10, 20, 40]')
    parser.add_argument('--window_length', type=int, default=41, help='window_length: [21, 41, 81]')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes: [10, 11]')
    parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device: [0, 1, 2, 3]')
    parser.add_argument('--dataset', type=str, default='itainnova', help='Dataset: [harup, itainnova]')
 
    args = parser.parse_args()


    harModel = instance_har_model(args)
    main(harModel, args)




