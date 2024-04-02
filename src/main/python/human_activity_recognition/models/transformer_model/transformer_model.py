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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
loss_fn = torch.nn.CrossEntropyLoss()

# If window lenght == 41: You can try to decrease the embed_size to 128
# If window lenght == 81: You can try to decrease the embed_size to 256
# Then increase the batch size until you reach the limit, preferably, to 32. 32 doesnt work with embed_size = 256


class SelfAttention(nn.Module):
    """
    This is the Multi-Head Attention module
    """
    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
        self._embed_size = embed_size
        self._heads = heads
        self._head_dim = embed_size // heads

        assert (self._head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get batch
        B = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # TODO value_len is same that embed_size?

        values = self.values(values)  # (B, value_len, embed_size)
        keys = self.keys(keys)  # (B, key_len, embed_size)
        queries = self.queries(query)  # (B, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(B, value_len, self._heads, self._head_dim)
        keys = keys.reshape(B, key_len, self._heads, self._head_dim)
        queries = queries.reshape(B, query_len, self._heads, self._head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        # queries shape: (B, query_len, heads, heads_dim),
        # keys shape: (B, key_len, heads, heads_dim)
        # energy: (B, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self._embed_size ** (1 / 2)), dim=3)
        # attention shape: (B, heads, query_len, key_len)

        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(B, query_len, self._heads * self._head_dim)
        # attention shape: (B, heads, query_len, key_len)
        # values shape: (B, value_len, heads, heads_dim)
        # out after matrix multiply: (B, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (B, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(  # MLP
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(TransformerEncoder, self).__init__()
        self._embed_size = embed_size
        #self.position_embedding = nn.Embedding(max_length, embed_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, embed_size))
        #self.pos_embed = nn.Parameter(torch.zeros(1, embed_size))

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.dropout(
            (x + self.pos_embed)
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out



class TransformerModel(pl.LightningModule):
    def __init__(self, input_features, num_classes, window_length, tune=False, trial=None):        

  
        super().__init__()
        # Define allowed values for heads

        # Predefined list of valid (embed_size, heads) pairs
        valid_pairs = [
            (32, 4), (64, 4), (64, 8), (128, 4), (128, 8), (256, 4), (256, 8)
            # Add more pairs as needed
        ]


        # Decide on hyperparameters based on whether Optuna is used
        if tune and trial is not None:
            self.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  
            self.dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)  
      
            # Suggest a valid pair
            chosen_pair = trial.suggest_categorical('embed_size_heads_pair', valid_pairs)

            # Extract embed_size and heads from the chosen pair
            self.embed_size, self.heads = chosen_pair

            self.num_layers = trial.suggest_int('num_layers', 4, 8)  
            self.forward_expansion = trial.suggest_int('forward_expansion', 2, 6) 
            self.optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])






        # Decide on hyperparameters based on whether Optuna is used
        if tune == False and trial is not None:
            self.lr = trial.params['lr'] 
            self.dropout_rate = trial.params['dropout_rate']  

            chosen_pair = trial.params['embed_size_heads_pair']
            self.embed_size, self.heads = chosen_pair
            self.num_layers = trial.params['num_layers']
            self.forward_expansion = trial.params['forward_expansion']
            self.optimizer = trial.params['optimizer']
        else:

            self.lr = 0.000227003
            self.dropout_rate = 0.298658
            self.heads = 4
            self.embed_size = 128
            self.num_layers = 7
            self.forward_expansion = 4 
            self.optimizer = 'RMSprop'

    
        self.save_hyperparameters()  # Save hyperparameters
 

        max_length = window_length

        self.transformer_encoder = TransformerEncoder(embed_size=self.embed_size,
                                                      num_layers=self.num_layers,
                                                      heads=self.heads,
                                                      forward_expansion=self.forward_expansion,
                                                      dropout=self.dropout_rate,
                                                      max_length=max_length)
        # https://arxiv.org/abs/2107.00606
        # Action Transformer: A Self-Attention Model for Short-Time Pose-Based Human Action Recognition
        self.linear_projection = nn.Linear(input_features, self.embed_size)
        self.linear_last = nn.Linear(self.embed_size, num_classes)



    def forward(self, x):
        #x = x.permute(0, 2, 1).float()
       
        x = self.linear_projection(x)
        x = self.transformer_encoder(x, mask=None)
        # In this page they do avg pooling for classification: http://peterbloem.nl/blog/transformers 
        #x.mean(dim=1)
        x = self.linear_last(x.mean(dim=1))
        #return F.log_softmax(x, dim=1) # Irene: creo que la funcion de perdida ya implementa esto
        return x
    
    


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