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

import torch.nn as nn
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

loss_fn = torch.nn.CrossEntropyLoss()

class MulticlassLSTMModel(pl.LightningModule):
    def __init__(self, input_features=30, num_classes=10,  tune=False, trial=None): #lr=0.001, dropout_rate=0.2, 
        super().__init__()


        # Decide on hyperparameters based on whether Optuna is used
        if tune and trial is not None:
            self.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # Log-uniform distribution
            self.dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)  # Uniform distribution
            self.fc1_size = trial.suggest_categorical('fc1_size', [32, 64, 128, 256, 512])
            self.fc2_size = trial.suggest_categorical('fc2_size', [32, 64, 128, 256, 512])
            self.fc3_size = trial.suggest_categorical('fc3_size', [32, 64, 128, 256, 512])
            self.optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
            self.num_layers = trial.suggest_int('num_layers', 1, 2)

        # Decide on hyperparameters based on whether Optuna is used
        if tune == False and trial is not None:
            self.lr = trial.params['lr']  # Log-uniform distribution
            self.dropout_rate = trial.params['dropout_rate']  # Uniform distribution
            self.fc1_size = trial.params['fc1_size']
            self.fc2_size = trial.params['fc2_size']
            self.fc3_size = trial.params['fc3_size']
            self.optimizer = trial.params['optimizer']
            self.num_layers = trial.params['num_layers']
        else:
            self.lr = 0.001  # Default hardcoded value
            self.dropout_rate = 0.2 # Default hardcoded value
            self.fc1_size = 512
            self.fc2_size = 256
            self.fc3_size = 128
            self.optimizer = 'Adam'
            self.num_layers = 1

        self.save_hyperparameters()  # Save hyperparameters

        self.lstm = nn.LSTM(input_size=input_features, hidden_size=self.fc1_size, num_layers=self.num_layers) # hidden_size=output_features
        self.fc1 = nn.Linear(in_features=self.fc1_size, out_features=self.fc2_size) # Fully connected / dense
        self.batchnorm1 = nn.BatchNorm1d(self.fc2_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.fc2_size, out_features=self.fc3_size)
        self.batchnorm2 = nn.BatchNorm1d(self.fc3_size)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc3 = nn.Linear(in_features=self.fc3_size, out_features= num_classes)

    def forward(self,x):
    
        output,_status = self.lstm(x) # output, (hn, cn) = rnn(input, (h0, c0))
        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.batchnorm2(output)
        output = self.relu2(output)
        output = self.dropout(output)
        output = self.fc3(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self(x)
        pred = y_hat.argmax(dim=1)
        y = y.argmax(dim=1)

        loss = loss_fn(y_hat, y)
        
        accuracy  = accuracy_score(y.cpu(), pred.cpu())
        precision = precision_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        f1        = f1_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        recall    = recall_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        
        # Use the current of PyTorch logger
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_accuracy", accuracy, on_epoch=True, sync_dist=True)
        self.log("train_precision", precision, on_epoch=True, sync_dist=True)
        self.log("train_f1", f1, on_epoch=True, sync_dist=True)
        self.log("train_recall", recall, on_epoch=True, sync_dist=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        pred = y_hat.argmax(dim=1)
        y = y.argmax(dim=1)

        test_loss = loss_fn(y_hat, y)

        accuracy  = accuracy_score(y.cpu(), pred.cpu())
        precision = precision_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        f1        = f1_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        recall    = recall_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        
        self.log("validation_loss", test_loss, on_epoch=True, sync_dist=True)
        self.log("validation_accuracy", accuracy, on_epoch=True, sync_dist=True)
        self.log("validation_precision", precision, on_epoch=True, sync_dist=True)
        self.log("validation_f1", f1, on_epoch=True, sync_dist=True)
        self.log("validation_recall", recall, on_epoch=True, sync_dist=True)
        self.log("current_epoch", self.current_epoch)


  

    
    def predict_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        y_hat_batch = self(X_batch)
        return y_hat_batch
        

    def configure_optimizers(self):

        if self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
