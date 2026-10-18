# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt

from .nose_build_model import BiSeNet
import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


def vis_parsing_maps(im, parsing_anno, x=0, y=0, stride=1, class_ids=(10,)):
    """
    Brief:从 19 类解析结果中提取指定类别构成的二值掩码
    Args:
        im:原图，仅用于取得输出掩码的尺寸
        parsing_anno:解析结果，大小为(H,W)，每个像素的值为类别 ID
        x,y,stride:历史遗留参数，默认为 0/0/1
        class_ids:需要提取的类别 ID，可以传入单个 int 或者一组 ID，默认为鼻子(10)
                  常用取值: 2/3 眉毛, 4/5 眼睛, 10 鼻子, 11 口腔, 12/13 上下唇
    """
    im = np.array(im)
    vis_parsing_anno = parsing_anno.copy()

    ids = np.atleast_1d(np.asarray(class_ids)).reshape(-1)
    index = np.where(np.isin(vis_parsing_anno, ids))  # 获得对应分类的像素坐标

    idx_y = (index[0] + y).astype(np.int32)
    idx_x = (index[1] + x).astype(np.int32)

    face_mask = np.zeros((im.shape[0], im.shape[1]), dtype="float32")
    face_mask[idx_y, idx_x] = 1.0
    return face_mask


def parsing_map(img_size, image1, net, input_rgb=False):
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        # Legacy nose/eye inference accepted BGR here. New semantic adapters
        # receive RGB crops from PIL and must not swap their channels again.
        img = Image.fromarray(image1 if input_rgb else cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

        image = img.resize((img_size, img_size))
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing_ = out.squeeze(0).cpu().numpy().argmax(0)
        # print('<{}> image : '.format(idx), np.unique(parsing_))

        parsing_ = cv2.resize(
            parsing_,
            (image1.shape[1], image1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        parsing_ = parsing_.astype(np.uint8)
    return parsing_


def inference(img_size, image1, net, class_ids=(10,)):
    parsing = parsing_map(img_size, image1, net)
    return np.asarray(vis_parsing_maps(image1, parsing, class_ids=class_ids), dtype="float32")
