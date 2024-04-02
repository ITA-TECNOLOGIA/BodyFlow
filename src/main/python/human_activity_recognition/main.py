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
import sys
import argparse
import logging
import torch
from datetime import datetime

# Custom codes
from dataset import HumanActivityDataset
from dataset.upfalldata import UPFalldata
from dataset.upfalldata_vision import UPFalldataVision
from dataset.logs_reader import LogsReader
from dataset.itadata import ITAdata
from models.lstm_model.lstm_model import MulticlassLSTMModel
from models.cnn_model.cnn_model import CNNModelMulticlass
from models.transformer_model.transformer_model import TransformerModel
from utils.util import plot_confusion_matrix, auto_logged_info, predictions_to_csv, save_confusion_matrix

# MLFlow for output manager
import mlflow
import mlflow.pytorch

# Pytorch Lightning for training tracking
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from pytorch_lightning.utilities.memory import get_model_size_mb
import optuna

pl.seed_everything(1)

def data_reader(args):
    """
    Returns a the windowed dataset with labels according to the data type
    """
    data = None
    if args.dataset == "harup":
        data = UPFalldata(args)
    elif args.dataset == "harup_vision":
        data = UPFalldataVision(args)
    elif args.dataset == "itainnova":
        data = ITAdata(args)
    else:
        sys.exit(f"HAR model {args.har_model} not implemented!")
    return data



def instance_har_model(input_features, num_classes, window_size, tune=False, trial=None):
    """
    Returns a Human Activity Recognition Model
    """
    har_model = None
    if args.har_model == "lstm":
        har_model = MulticlassLSTMModel(input_features = input_features, 
                                        num_classes = num_classes, tune=tune, trial=trial)
    elif args.har_model == "cnn":
        har_model = CNNModelMulticlass(input_features = input_features, window_length = window_size, 
                                       num_classes = num_classes, tune=tune, trial=trial)  
    elif args.har_model == "transformer":
        har_model = TransformerModel(input_features = input_features, window_length = window_size, 
                                       num_classes = num_classes, tune=tune, trial=trial)  
    else:
        sys.exit(f"HAR model {args.har_model} not implemented!")
    return har_model 


def train(args):
    data = data_reader(args)
    X_train, y_train, X_test, y_test, df_test, input_features, num_classes, classes = data.fetch_data()    

    train_dataset = HumanActivityDataset(X = X_train, Y = y_train)

    # Split training data for train and validation
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


    print(os.getcwd())
    #logger = MLFlowLogger(tracking_uri = args.tracking_uri)

    checkpoint_callback = ModelCheckpoint(
                        filename = "m-{current_epoch:.0f}-{validation_loss:.2f}",
                        save_top_k = 3,
                        monitor = "validation_loss",
                        mode = "min",
                        every_n_epochs = 1,
                        verbose = False)

    # Pytorch Lightning Trainer
    trainer = pl.Trainer(devices = [args.gpu], 
                         accelerator = 'gpu', 
                         max_epochs = args.epochs,
                         #logger = logger, 
                         check_val_every_n_epoch = 1, 
                         callbacks = [checkpoint_callback])

    with mlflow.start_run() as run:
    #with mlflow.start_run(run_id = mlflow_logger.run_id, experiment_id = mlflow_logger.experiment_id) as run:
        mlflow.set_tags({"har_model": args.har_model, "input_data": args.input_data})
        mlflow.pytorch.autolog()
        model = instance_har_model(input_features, num_classes, args.window_size)

        logging.info(f'Model size: {get_model_size_mb(model)}')

        trainer.fit(model, train_dataloader, valid_dataloader)

        predictions = trainer.predict(model, dataloaders = test_dataloader, ckpt_path = 'best')
        
        # Transform One-Hot format into int format
        y_pred = torch.cat(predictions).argmax(axis = 1) 
        y_true = test_dataset.Y.argmax(axis = 1)
        plot_confusion_matrix(y_true, y_pred, classes, args.har_model, args.input_data, args.window_size)

        auto_logged_info(mlflow.get_run(run_id = run.info.run_id))  

        # Log model to MLFlow
        mlflow.pytorch.log_model(model, "model")
     
        predictions_to_csv(df_test, y_pred)
        mlflow.end_run()



def tune(args):

    data = data_reader(args)
    X_train, y_train, _, _, _, input_features, num_classes, _ = data.fetch_data()    

    train_dataset = HumanActivityDataset(X = X_train, Y = y_train)

    # Split training data for train and validation
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size], generator = seed)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch, shuffle = True, 
                                                pin_memory = False, num_workers = args.workers) 
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = args.batch, shuffle = False, 
                                                pin_memory = False, num_workers = args.workers) 



    def objective(trial):

        with mlflow.start_run(run_name="test_run", nested=True):

            mlflow.pytorch.autolog()
            model = instance_har_model(input_features=input_features, num_classes=num_classes, window_size=args.window_size, tune=args.tune, trial=trial)

            # Define the ModelCheckpoint callback
            checkpoint_callback = ModelCheckpoint(
                filename=f'har_{args.har_model}_{args.input_data}_{args.window_size}-{{epoch}}',
                save_top_k=1,
                monitor="validation_f1",
                mode="max",
                every_n_epochs=1,
                verbose=False
            )

        

            early_stop_callback = EarlyStopping(
            monitor='validation_f1',
            patience=10,
            verbose=False,
            mode='max'
            )

            # logger = MLFlowLogger(
            #     experiment_name=f"{args.har_model}_{args.input_data}_{args.window_size}",
            #     tracking_uri=args.tracking_uri)

            trainer = pl.Trainer(
                
                    devices = [args.gpu], 
                    accelerator = 'gpu', 
                    max_epochs = args.epochs,
                    #logger=logger,
                    check_val_every_n_epoch = 1, 
                    callbacks=[early_stop_callback, checkpoint_callback])
    

            trainer.fit(model, train_dataloader, valid_dataloader)
            
            #trainer.logger.log_hyperparams(trial.params)
            mlflow.pytorch.log_model(model, "model")

            # Store the validation F1 score immediately after training
            validation_f1 = trainer.callback_metrics["validation_f1"].item()

            return validation_f1


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2) # Number of trials in the Optuna study



def test(args):
    saving_path = 'summary'
    data = data_reader(args)
    _, _, X_test, y_test, _, input_features, num_classes, classes = data.fetch_data()    

    test_dataset = HumanActivityDataset(X = X_test, Y = y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch, shuffle = False,
                                                  pin_memory = False, num_workers = args.workers) 

    model = instance_har_model(input_features, num_classes, args.window_size)

    model_name = f'har_{args.har_model}_{args.input_data}_{args.window_size}.ckpt' #har_cnn_ankle_41.ckpt

    model.load_from_checkpoint(os.path.join('models', model_name), input_features=input_features, 
                               num_classes=num_classes, window_length= args.window_size)
    model.freeze()
    predictor = pl.Trainer(devices = [args.gpu],
                           accelerator = 'gpu',
                           logger = None)
    predictions = predictor.predict(model, dataloaders = test_dataloader, ckpt_path= os.path.join('models', model_name))
    y_pred = torch.cat(predictions)
    y_pred = y_pred.argmax( axis=1)
    y_true = test_dataset.Y.argmax(axis = 1)
    save_confusion_matrix(y_true, y_pred, classes, args.har_model, args.input_data , args.window_size, saving_path)


def predict(log_filename, person_id, gpu):
    input_data = '2d' 
    window_size = 41
    window_step = 1
    data = LogsReader(log_filename, person_id, input_data, window_size, window_step)
    X_pred, har_in_features = data.fetch_data()
    # pred_dataset = HumanActivityDataset(X = X_pred, Y = _)
    X_pred= torch.tensor(X_pred, dtype = torch.float32)
    predict_dataloader = torch.utils.data.DataLoader(X_pred, batch_size = args.batch, shuffle = False,
                                                  pin_memory = False, num_workers = args.workers) 

    model = instance_har_model(input_features = 68, num_classes = 11, window_size = 41)
    model_name = 'har_lstm_2d_41.ckpt'
    model.load_from_checkpoint(os.path.join('models', model_name), input_features=68, 
                               num_classes = 11, window_length= 41)
    model.freeze()
    predictor = pl.Trainer(devices = [gpu],
                           accelerator = 'gpu',
                           logger = None)
    out = []
    with torch.no_grad():
        for batch in predict_dataloader:
            # Get the input data from the batch
            inputs = batch

            # Pass the inputs through the model
            outputs = model(inputs)
            out.append(outputs)
    # predictions = predictor.predict(model, dataloaders = test_dataloader, ckpt_path= os.path.join('models', model_name))
    y_pred = torch.cat(out)

    y_pred = y_pred.argmax( axis=1)
    
    
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
    parser.add_argument('--har_model', type = str, default = 'transformer', help = 'HAR model: [lstm, cnn, transformer]')
    parser.add_argument('--input_data', type = str, default = '3d', help = '[all, 2d, 3d, imus, ankle]')
    parser.add_argument('--batch', type = int, default = 64, help = 'Batch size')
    parser.add_argument('--epochs', type = int, default = 100, help = 'Epochs: number of epochs (int)')
    parser.add_argument('--path_dataset', type = str, default = 'data/har/processed_harup.csv') 
    parser.add_argument('--label', type = str, default = "activity", help = 'Label: [activity, tag]')
    parser.add_argument('--window_step', type = int, default = 1, help = 'Window step: [1]')
    parser.add_argument('--window_size', type = int, default = 41, help = 'window_size: [21, 41, 81]')
    parser.add_argument('--workers', type = int, default = 64, help = 'Number of workers to feed the data')
    parser.add_argument('--gpu', type = int, default = 0, help = 'Cuda device: [0, 1, 2, 3]')
    parser.add_argument('--dataset', type = str, default = 'harup_vision', help = 'Dataset: [harup, harup_vision, itainnova]')
    parser.add_argument('--train', type = bool, default = True, help = 'Trainning mode activated')
    parser.add_argument('--test', type = bool, default = False, help = 'Testing mode activated')
    parser.add_argument('--tune', type = bool, default = False, help = 'Tuning hyperparameters mode activated')
    # parser.add_argument('--train', type=str, default='True', help='Training mode activated')
    # parser.add_argument('--test', type=str, default='False', help='Testing mode activated')

    args = parser.parse_args()
    logging.info(args)
    # args.train = args.train.lower() == 'true'
    # args.test = args.test.lower() == 'true'

    if args.train == True and args.tune == False:
        train(args)
    elif args.tune == True:
        tune(args)
    elif args.test == True:
        test(args)
    elif args.predict == True:
        log_filename = 'Log_cpn_mhformer_bailar_1.csv'
        person_id = 1
        gpu = 0
        predict(log_filename, person_id, gpu)
    

    
