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

# This code was extracted from MHFormer: https://github.com/Vegetebird/MHFormer/blob/fcf238631016f906477ec9c1d17582097ecf9803/common/camera.py#L19

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

def normalize_screen_coordinates(X, w, h): 
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio.
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h/w]

def normalize_screen_coordinates_bbox(X, bbox):
    X_centered =  X - np.array([bbox[0], bbox[1]])
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    X_centered = normalize_screen_coordinates(X_centered, w, h)
    return X_centered

import numpy as np

def find_containing_box(boxes):
    # Convert the list of bounding boxes to a NumPy array for easier manipulation
    boxes = np.array(boxes)

    # Find the minimum and maximum values for each coordinate (x and y)
    x1 = max(0, np.min(boxes[:, 0]))
    y1 = max(0, np.min(boxes[:, 1]))
    x2 = max(boxes[:, 2])
    y2 = max(boxes[:, 3])

    width, height = x2 - x1, y2 - y1
    # With this we are ensuring an aspect ratio of 1:1
    quantity_to_add = int(abs(width - height) / 2) # Because we are gonna add it per each side
    if width >= height:
        y1 -= quantity_to_add
        if y1 < 0:
            y2 += quantity_to_add + abs(y1)
            y1 = 0
        else:
            y2 += quantity_to_add
    else:
        x1 -= quantity_to_add
        if x1 < 0:
            x2 += quantity_to_add + abs(x1)
            x1 = 0
        else:
            x2 += quantity_to_add

    containing_box = [x1, y1, x2, y2]

    width, height = x2 - x1, y2 - y1
    aspect_ratio = width / height
    assert 0.9 < aspect_ratio < 1.1, f"Aspect ratio is not close to 1:1 {aspect_ratio} {boxes}"

    return containing_box


def normalize_screen_coordinates_bboxes(X, bboxes):
    assert X.shape[0] == bboxes.shape[0]

    # Get the bounding box that contains all other bounding boxes
    bbox = find_containing_box(bboxes)
    X_centered = normalize_screen_coordinates_bbox(X, bbox)
    return np.array(X_centered)

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def qrot(q, v):
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
	uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)


