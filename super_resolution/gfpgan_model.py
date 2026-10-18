import argparse
import time

import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from .gfpgan import GFPGANer
from PIL import Image, ImageFilter
import json
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from utils import timer


class SuperResolutionModel(object):
    def __init__(
        self,
        model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler="realesrgan",
        aligned=False,
        only_center_face=False,
        bg_tile=400,
    ):
        self.model_path = model_path
        self.upscale = upscale
        self.arch = arch
        self.channel_multiplier = channel_multiplier
        self.bg_upsampler = bg_upsampler
        self.aligned = aligned
        self.only_center_face = only_center_face
        self.bg_tile = bg_tile
        """
        Args:
            model_path:模型权重文件
            upscale:背景上采样模型的采样倍率
            arch:和权重版本有关，根据需要设定为clean/original
            channel_multiplier:和权重版本有关，默认为2，不建议修改
            bg_upsampler:上采样模型
            aligned:是否人脸是对齐的，不建议修改
            only_center_face:是否只是输出人脸部分，不建议修改
            bg_tile:背景会被切分为bg_tile*bg_tile进行超分，然后再拼接回去 
        """
        with timer("加载RRDBNet"):
            self.model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
        with timer("加载RealESRGANer"):
            self.bg_upsampler = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=self.model,
                tile=self.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )
        with timer("加载GFPGANer"):
            self.restorer = GFPGANer(
                model_path=self.model_path,
                upscale=self.upscale,
                arch=self.arch,
                channel_multiplier=self.channel_multiplier,
                bg_upsampler=self.bg_upsampler,
            )
    @timer
    def inference(self, input_img):
        input_img = input_img[..., ::-1]  # RGB->BGR
        _, _, out_img = self.restorer.enhance(
            input_img,
            has_aligned=self.aligned,
            only_center_face=self.only_center_face,
            paste_back=True,
        )
        out_img = out_img[..., ::-1]  # BGR->RGB
        return out_img
