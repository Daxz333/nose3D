import numpy as np
import cv2
from PIL import Image


def crop_face_square(image, bbox):
    """

    Brief:
        得到bbox划出来的子图
        但是bbox可能是任意形状的矩形，因此会将其变为正方形，方便后续其他网络的输入处理，尽可能避免图像长宽比例改变带来的失真影响
    Args:
        image:包含人脸的图片
        bbox:人脸检测器得到的人脸框左上角和右下角坐标
    Returns:
        切分出来的人脸子图，具体为：image[beight_begin:height_end,width_begin:width_end]

    """
    height_begin, height_end, width_begin, width_end = (
        bbox[1],
        bbox[3],
        bbox[0],
        bbox[2],
    )
    height = height_end - height_begin
    width = width_end - width_begin
    pad = [[0, 0], [0, 0], [0, 0]]
    if height < width:
        delta = (width - height) // 2
        height_begin = max(0, height_begin - delta)
        height_end = min(image.shape[0], height_end + delta)
    if width < height:
        delta = (height - width) // 2
        width_begin = max(0, width_begin - delta)
        width_end = min(image.shape[1], width_end + delta)
    height_begin = max(0, int(height_begin))
    height_end = min(image.shape[0], int(height_end))
    width_begin = max(0, int(width_begin))
    width_end = min(image.shape[1], int(width_end))
    if height_end <= height_begin or width_end <= width_begin:
        raise ValueError("检测到的人脸框位于图片外，请重新选择图片")
    return image[height_begin:height_end, width_begin:width_end], (
        height_begin,
        height_end,
        width_begin,
        width_end,
    )


def resize_image_with_crop_or_pad(image, target_size=512):
    """
    Author:shijunfeng
    Brief:
        再保证图片长宽比例不会改变失真的情况下，改变图片大小
        具体而言，首先会通过两边黑边补齐的方式把图片变为正方形
        然后再通过插值法将其resize到目标大小
        因为大部分深度学习模型的输入都是正方形的，所以这一步很有必要
    Args:
        image:RGB格式的图片
        target_size:目标正方形图片边长
    Returns:
        一张大小为(target_size,target_size,3)的RGB格式的图片
    """
    size = max(image.shape)
    delta = (np.clip(size - np.array(image.shape[:2]), 0, np.inf) // 2).astype("int32")
    if len(image.shape) == 3:
        image = np.pad(image, [[delta[0], delta[0]], [delta[1], delta[1]], [0, 0]])
    else:
        image = np.pad(image, [[delta[0], delta[0]], [delta[1], delta[1]]])

    image = cv2.resize(image, [target_size, target_size])
    return image
