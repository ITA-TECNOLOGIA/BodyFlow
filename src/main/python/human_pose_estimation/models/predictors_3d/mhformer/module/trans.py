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
# Modified for BodyFlow Version: 2.0
# Modifications Copyright (c) 2024 Instituto Tecnologico de Aragon
# (www.ita.es) (Spain)
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

"""
Code inspired by MHFormer authors:
https://github.com/Vegetebird/MHFormer/blob/main/model/mhformer.py
"""
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath

class Mlp(nn.Module):
    """
    Multi-layer perceptron (MLP). The MLP is a neural network model that consists of multiple layers of linear operations 
    (i.e., matrix multiplications) followed by non-linear activation functions.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        - in_features: the number of input features
        - hidden_features: the number of hidden features, if not provided it will be equal to in_features
        - out_features: the number of output features, if not provided it will be equal to in_features
        - act_layer: the activation function to use (default GELU)
        - drop: the dropout rate to use (default 0)
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    Module for the attention mechanism used in the Transformer architecture. 
    The attention mechanism is a key component in the Transformer, allowing the model to selectively
    focus on certain parts of the input when making predictions.

    - dim: the dimensionality of the input and output tensors
    - num_heads: the number of attention heads to use (default 8)
    - qkv_bias: a boolean indicating whether to use bias in the linear layers that compute the query, 
        key and value representations (default False)
    - qk_scale: a scaling factor for the dot product of the query and key representations 
        (default None, which results in using the square root of the head dimension)
    - attn_drop: the dropout rate to use for the attention scores (default 0)
    - proj_drop: the dropout rate to use for the final linear projection (default 0)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        The forward method of the module takes in a tensor x and applies the attention mechanism to it.   
        """
        B, N, C = x.shape
        # compute the query, key and value representations using a linear layer and reshaping
        # the output to have 4 dimensions: (batch size, sequence length, 3, number of heads, head dimension)
        # Then it permute the dimensions to (3,batch_size, number of heads, sequence length, head dimension)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # The query, key, and value are separated
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        # Computation the dot product of the query and key representations
        # scaled by the scale factor, and applies a softmax function to obtain the attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        # The attention scores are then multiplied with the value representation and a final linear projection is applied to obtain the output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """
    Single block of a transformer architecture
    """
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        - dim: the number of features in the input and output of the block.
        - num_heads: the number of attention heads to use in the Attention module
        - mlp_hidden_dim: the number of hidden features to use in the Multi Layer Perceptron (MLP) module
        - qkv_bias: a boolean indicating whether to use bias in the Attention module
        - qk_scale: a scaling factor to use in the Attention module
        - drop: the dropout rate to use in the MLP module
        - attn_drop: the dropout rate to use in the Attention module
        - drop_path: the dropout rate for drop path in the block
        - act_layer: the activation function to use in the MLP module
        - norm_layer: the normalization layer to use in the block
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# https://github.com/Vegetebird/MHFormer/blob/main/model/module/trans.py
class Transformer(nn.Module):
    """
    Transformer Encoder architecture
    """
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=27):
        """
        - depth: controls the number of blocks in the architecture
        - embed_dim: number of neurons in the embedding layers
        - mlp_hidden_dim: number of neurons in the hidden layers of the MLP
        - h: number of heads in the attention mechanism
        - drop_rate: dropout rate
        - length: length of the input
        """
        super().__init__()
        drop_path_rate = 0.2
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # embedding layer for the position of the input
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        # dropout applied
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        # number of blocks, each of which is an instance of the Block class. 
        # Each block contains an attention mechanism, an MLP, and dropout, drop path and normalization layers
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x += self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x



