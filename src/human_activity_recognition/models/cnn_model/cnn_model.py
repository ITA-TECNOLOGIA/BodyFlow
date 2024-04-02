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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, f1_score


loss_fn = torch.nn.CrossEntropyLoss()
# AFAR model
class CNNModelMulticlass(pl.LightningModule):
    """
    CNN model 
    """
    def __init__(self, input_features=30, window_length = 41, num_classes=10):
        super().__init__()
        self.conv_1d_1 = nn.Conv1d(input_features, 512, kernel_size=4)
        self.batchnorm_1 = nn.BatchNorm1d(512)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.relu_1 = nn.ReLU()

        self.conv_1d_2 = nn.Conv1d(512, 512, kernel_size=4)
        self.batchnorm_2 = nn.BatchNorm1d(512)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.relu_2 = nn.ReLU()

        self.conv_1d_3 = nn.Conv1d(512, 512, kernel_size=4)
        self.batchnorm_3 = nn.BatchNorm1d(512)
        self.dropout_3 = nn.Dropout(p=0.2)
        self.relu_3 = nn.ReLU()



        if window_length == 81:
            self.avg_pool_1d_3 = nn.AvgPool1d(75) # 81 window size
        elif window_length == 41:
            self.avg_pool_1d_3 = nn.AvgPool1d(35)
        elif window_length == 21:
            self.avg_pool_1d_3 = nn.AvgPool1d(9) # 15 window size
        #else:
        #    raise NotImplementedError(f"Window length {window_length} not compatible")

        self.linear_4 = nn.Linear(512, 256)     
        self.dropout_4 = nn.Dropout(p=0.2)
        self.relu_4 = nn.ReLU()

        self.linear_5 = nn.Linear(256, num_classes)






    def forward(self, x):
        #############
        ###   1   ###
        #############
        
        # Reshape input for CNN https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        original_input_shape = x.size()
        first_dimension = original_input_shape[0]
        second_dimension = original_input_shape[1]
        third_dimension = original_input_shape[2]
        

        
        x = torch.reshape(x, (first_dimension, third_dimension, second_dimension ))
        x = self.conv_1d_1(x.float())
        x = self.dropout_1(x)
        x = self.relu_1(x)
        #############
        ###   2   ###
        #############
        x = self.conv_1d_2(x)
        x = self.dropout_2(x)
        x = self.relu_2(x)
        #############
        ###   3   ###
        #############
        x = self.avg_pool_1d_3(x)
        x = x.squeeze(2)
        #############
        ###   4   ###
        #############
        x = self.linear_4(x)
        x = self.dropout_4(x)
        x = self.relu_4(x)
        #############
        ###   5   ###
        #############
        x = self.linear_5(x)
        # print('IRENE: atencion con esta ultima capa, es necesaria????? creo que cross entropy ya la impelmenta??')
        # return F.log_softmax(x, dim=1)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        

        # The documentation changes from logits to probabilities, and explicitly states that one needs to apply sigmoid/softmax before
        # passing predictions into the metric (this will be the case when users use CrossEntropyLoss or BCEWithLogitsLoss)

        y_hat = F.log_softmax(y_hat, dim=1)
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
        y_hat = self(x) #   y_hat= ouput, y = target
        y_hat = F.log_softmax(y_hat, dim=1)
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

        y_hat_batch = F.log_softmax(y_hat_batch, dim=1)

        return y_hat_batch
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001) # original 0.00001