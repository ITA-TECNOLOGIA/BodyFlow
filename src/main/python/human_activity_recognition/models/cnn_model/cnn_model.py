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

import logging
import torch.nn as nn
import torch

import pytorch_lightning as pl

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


loss_fn = torch.nn.CrossEntropyLoss()

# AFAR model
class CNNModelMulticlass(pl.LightningModule):
    def __init__(self, input_features, num_classes, window_length, tune=False, trial=None):
        super().__init__()

        # Decide on hyperparameters based on whether Optuna is used
        if tune and trial is not None:
            self.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # Log-uniform distribution
            self.dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)  # Uniform distribution
            self.conv1_size = trial.suggest_categorical('conv1_size', [32, 64, 128, 256, 512])
            self.out_size = trial.suggest_categorical('out_size', [32, 64, 128, 256, 512])
            self.avg_pool = trial.suggest_int('avg_pool', 9, 75)  # Set your desired range here
            self.kernel_size = trial.suggest_int('kernel_size', 2, 5)  # Set your desired range here
            self.optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

        # Decide on hyperparameters based on whether Optuna is used
        if tune == False and trial is not None:
            self.lr = trial.params['lr']  # Log-uniform distribution
            self.dropout_rate = trial.params['dropout_rate']  # Uniform distribution
            self.conv1_size = trial.params['conv1_size']
            self.out_size = trial.params['out_size']
            self.avg_pool = trial.params['avg_pool']
            self.kernel_size = trial.params['kernel_size']
            self.optimizer = trial.params['optimizer']
        else:
            self.lr = 0.001  # Default hardcoded value
            self.dropout_rate = 0.2 # Default hardcoded value
            self.conv1_size = 512
            self.out_size = 256
            self.avg_pool = 9
            self.kernel_size = 4
            self.optimizer = 'Adam'


        self.save_hyperparameters()  # Save hyperparameters


        self.conv_1d_1 = nn.Conv1d(input_features, self.conv1_size, kernel_size = self.kernel_size)
        self.batchnorm_1 = nn.BatchNorm1d(self.conv1_size)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=self.dropout_rate)
        
        self.conv_1d_2 = nn.Conv1d(self.conv1_size, self.conv1_size, kernel_size = self.kernel_size)
        self.batchnorm_2 = nn.BatchNorm1d(self.conv1_size)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p = self.dropout_rate)

        self.conv_1d_3 = nn.Conv1d(self.conv1_size, self.conv1_size, kernel_size = self.kernel_size)
        self.batchnorm_3 = nn.BatchNorm1d(self.conv1_size)
        self.relu_3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(p=self.dropout_rate)
        



        conv_output_size = lambda in_size, k_size: in_size - k_size + 1
        pool_output_size = lambda in_size, pool_size: (in_size - pool_size) // pool_size + 1

        size_after_conv = conv_output_size(conv_output_size(conv_output_size(window_length, self.kernel_size), self.kernel_size), self.kernel_size)
        size_after_pool = pool_output_size(size_after_conv, self.avg_pool)

        # Check if size is too small
        if size_after_pool <= 0:
            raise ValueError("The output size after pooling is non-positive. Adjust the model configuration.")

        self.avg_pool_1d_3 = nn.AvgPool1d(self.avg_pool)

        # Calculate the number of features going into the linear layer
        linear_input_features = self.conv1_size * size_after_pool
        self.linear_4 = nn.Linear(linear_input_features, self.out_size)
        self.dropout_4 = nn.Dropout(p = self.dropout_rate)
        self.relu_4 = nn.ReLU()
        self.linear_5 = nn.Linear(self.out_size, num_classes)



    def forward(self, x):
        ## First layer
        x = torch.reshape(x, (x.size()[0], x.size()[2], x.size()[1] ))
        x = self.conv_1d_1(x.float())
        x = self.dropout_1(x)
        x = self.relu_1(x)

        ## Second layer
        x = self.conv_1d_2(x)
        x = self.dropout_2(x)
        x = self.relu_2(x)

        ## Average pooling layer
        x = self.avg_pool_1d_3(x)
        x = x.squeeze(2)

        ## Flatten the tensor from the second dimension
        x = torch.flatten(x, start_dim=1)  # Correct flattening to match the linear layer's input size
        
        ## Linear layer
        x = self.linear_4(x)
        x = self.dropout_4(x)
        x = self.relu_4(x)

        ## Second linear layer
        x = self.linear_5(x)
        
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        pred = outputs.argmax(dim=1)
        y = y.argmax(dim=1)
        loss = loss_fn(outputs, y)

        accuracy  = accuracy_score(y.cpu(), pred.cpu())
        precision = precision_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        f1        = f1_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)
        recall    = recall_score(y.cpu(), pred.cpu(), average='macro', zero_division=0)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_accuracy", accuracy, on_epoch=True, sync_dist=True)
        self.log("train_precision", precision, on_epoch=True, sync_dist=True)
        self.log("train_f1", f1, on_epoch=True, sync_dist=True) 
        self.log("train_recall", recall, on_epoch=True, sync_dist=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x) 
        pred = outputs.argmax(dim=1)
        y = y.argmax(dim=1)
        test_loss = loss_fn(outputs, y)    
     
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
        output_batch = self(X_batch)
        return output_batch
        

    def configure_optimizers(self):

        if self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)