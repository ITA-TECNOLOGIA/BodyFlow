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

import torch.nn as nn
import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, f1_score

loss_fn = torch.nn.CrossEntropyLoss()

class MulticlassLSTMModel(pl.LightningModule):
    def __init__(self, input_features=30, window_length=41, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=512, num_layers=1) # hidden_size=output_features
        self.fc1 = nn.Linear(in_features=512, out_features=256) # Fully connected / dense
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=128, out_features= num_classes)

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

        
        # Use the current of PyTorch logger
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_accuracy", accuracy, on_epoch=True, sync_dist=True)
        self.log("train_precision", precision, on_epoch=True, sync_dist=True)
        self.log("train_f1", f1, on_epoch=True, sync_dist=True)
        
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
     
        self.log("validation_loss", test_loss, on_epoch=True, sync_dist=True)
        self.log("validation_accuracy", accuracy, on_epoch=True, sync_dist=True)
        self.log("validation_precision", precision, on_epoch=True, sync_dist=True)
        self.log("validation_f1", f1, on_epoch=True, sync_dist=True)
        self.log("current_epoch", self.current_epoch)
  

    
    def predict_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        y_hat_batch = self(X_batch)
        return y_hat_batch
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)