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

import torch
from torch.utils.data import Dataset


class HumanActivityDataset(Dataset):

    def __init__(self, X, Y):
        # Converting data to torch tensors
        self.X = torch.tensor(X, dtype = torch.float32)
        self.Y = torch.tensor(Y, dtype = torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        features_X = self.X[idx]
        label_Y = self.Y[idx]
        return features_X, label_Y
