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

import torch
import numpy as np
import os
import sys
import logging

from models.predictors_2d.cpn.yolov3.util import *
from models.predictors_2d.cpn.yolov3.darknet import Darknet
from models.predictors_2d.cpn.yolov3 import preprocess

cur_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.join(cur_dir, '../../../')
chk_root = os.path.join(project_root, 'checkpoint/')
data_root = os.path.join(project_root, 'data/')


sys.path.insert(0, project_root)
sys.path.pop(0)


def load_model(inp_dim=416):
    CUDA = torch.cuda.is_available() 
    logging.info(f"People detector (yolo) in GPU: %r", CUDA)

    # Set up the neural network
    model = Darknet(os.path.join(cur_dir, 'cfg', 'yolov3.cfg'))
    model.load_weights(os.path.join('models', 'yolov3.weights'))
    logging.debug("YOLOv3 network successfully loaded")

    model.net_info["height"] = inp_dim
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    return model


def yolo_human_det(img, model, reso=416, confidence=0.70, nms_thresh=0.4):
    CUDA = torch.cuda.is_available()
    inp_dim = reso
    num_classes = 80

    img, ori_img, img_dim = preprocess.prep_image(img, inp_dim)
    img_dim = torch.FloatTensor(img_dim).repeat(1, 2)

    with torch.no_grad():
        if CUDA:
            img_dim = img_dim.cuda()
            img = img.cuda()
        output = model(img, CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thresh, det_hm=True)

        if len(output) == 0:
            return None, None

        img_dim = img_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / img_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * img_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * img_dim[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim[i, 1])

    bboxs = []
    scores = []
    for i in range(len(output)):
        item = output[i]
        bbox = item[1:5].cpu().numpy()
        # conver float32 to .2f data
        bbox = [round(i, 2) for i in list(bbox)]
        score = item[5].cpu().numpy()
        bboxs.append(bbox)
        scores.append(score)
    scores = np.expand_dims(np.array(scores), 1)
    bboxs = np.array(bboxs)

    return bboxs, scores
