# -------------------------------------------------------------------------------------------------------
# MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation [CVPR 2022]
# Copyright (c) 2016 Julieta Martinez, Rayat Hossain, Javier Romero
#
# @inproceedings{li2022mhformer,
#   title={MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation},
#   author={Li, Wenhao and Liu, Hong and Tang, Hao and Wang, Pichao and Van Gool, Luc},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#   pages={13147-13156},
#   year={2022}
# }
# -------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Modified for BodyFlow Version: 1.0
# Modifications Copyright (c) 2023 Instituto Tecnologico de Aragon
# (www.itainnova.es) (Spain)
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

"""
Code inspired by MHFormer authors:
https://github.com/Vegetebird/MHFormer/blob/main/model/mhformer.py
"""
import torch.nn as nn
from einops import rearrange
from models.predictors_3d.mhformer.module.trans import Transformer as Transformer_encoder
from models.predictors_3d.mhformer.module.trans_hypothesis import Transformer as Transformer_hypothesis

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## Multi-Hypothesis Generation (MHG)
        self.norm_1 = nn.LayerNorm(args['frames'])
        self.norm_2 = nn.LayerNorm(args['frames'])
        self.norm_3 = nn.LayerNorm(args['frames'])

        self.Transformer_encoder_1 = Transformer_encoder(4, args['frames'], args['frames']*2, length=2*args['n_joints'], h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, args['frames'], args['frames']*2, length=2*args['n_joints'], h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, args['frames'], args['frames']*2, length=2*args['n_joints'], h=9)

        ## Embedding
        self.embedding_1 = nn.Conv1d(2*args['n_joints'], args['channel'], kernel_size=1)
        self.embedding_2 = nn.Conv1d(2*args['n_joints'], args['channel'], kernel_size=1)
        self.embedding_3 = nn.Conv1d(2*args['n_joints'], args['channel'], kernel_size=1)

        ## Self-Hypothesis Refinement (SHR) and Cross-Hypothesis Interaction (CHI)
        self.Transformer_hypothesis = Transformer_hypothesis(args['layers'], args['channel'], args['d_hid'], length=args['frames'])
        
        ## Regression for getting the landmarks position
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args['channel']*3, momentum=0.1),
            nn.Conv1d(args['channel']*3, 3*args['out_joints'], kernel_size=1)
        )

    def forward(self, x):
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        ## Multi-Hypothesis Generation (MHG)
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x)) # x_1 = [b, (j c), f]
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) # x_2 = [b, (j c), f]
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2)) # x_3 = [b, (j c), f]
        
        ## Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous() # x_1 = [b, f, 512]
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous() # x_2 = [b, f, 512]
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous() # x_3 = [b, f, 512]

        ## Self-Hypothesis Refinement (SHR) and Cross-Hypothesis Interaction (CHI)
        x = self.Transformer_hypothesis(x_1, x_2, x_3)  # x = [b, f, 512+512+512]

        ## Regression for getting the landmarks position
        x = x.permute(0, 2, 1).contiguous() # x = [b, 512+512+512, f]
        x = self.regression(x) # x = [b, 17*3, f]  
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x





