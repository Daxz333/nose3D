"""
Author:Shijunfeng
Date:2022/8/23
Brief:对鼻子的分割模型进行一个封装，也为其他区域的分割模型提供一个通用接口做出准备，未来将会对接口做出一定的规范
"""
from . import nose_inference
from . import nose_build_model
from abc import ABCMeta, abstractclassmethod
import onnxruntime as ort
import numpy as np


class SegmentationInterface(object):
    @abstractclassmethod
    def __init__(self, **kwargs):
        pass

    @abstractclassmethod
    def inference(self, image):
        pass


#####################################################################################
class FaceParsing(SegmentationInterface):
    """
    BiSeNet 19 类人脸解析的通用封装。
    模型本身一次前向即可输出全部 19 类，取哪几个类别由 class_ids 决定，
    因此「换鼻 / 换眼 / 换唇 / 换眉」共用同一个模型，无需新增权重。
    """

    def __init__(self, model_path, num_classes=19, image_size=512, class_ids=(10,)):
        self.model = nose_build_model.build_model(num_classes, model_path)
        self.image_size = image_size
        self.class_ids = tuple(np.atleast_1d(class_ids).tolist())

    def inference(self, image, class_ids=None):
        """
        Args:
            image:RGB 图片
            class_ids:需要提取的类别 ID，None 表示使用构造时的默认类别
        """
        if class_ids is None:
            class_ids = self.class_ids
        return nose_inference.inference(self.image_size, image, self.model, class_ids)

    def parsing_map(self, image):
        """Return all labels from one forward pass for semantic protection."""
        return nose_inference.parsing_map(self.image_size, image, self.model, input_rgb=True)


class NoseSegmentation(FaceParsing):
    """兼容旧命名：固定取鼻子类别(10)"""

    def __init__(self, model_path, num_classes=19, image_size=512):
        super().__init__(
            model_path, num_classes=num_classes, image_size=image_size, class_ids=(10,)
        )


class FaceSegmentation(object):
    def __init__(self, model_path, downsample_ratio=0.5, threshold=0.5):
        self.session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )
        self.downsample_ratio = np.array([downsample_ratio], dtype=np.float32)
        self.threshold = threshold

    def inference(self, image):
        rec = [np.zeros([1, 1, 1, 1], dtype=np.float16)] * 4  # 必须用模型一样的 dtype
        image = np.transpose(image.astype("float16"), [2, 0, 1])[None, ...] / 255
        fgr, pha, *rec = self.session.run(
            [],
            {
                "src": image,
                "r1i": rec[0],
                "r2i": rec[1],
                "r3i": rec[2],
                "r4i": rec[3],
                "downsample_ratio": self.downsample_ratio,
            },
        )
        mask = np.transpose(pha[0], [1, 2, 0]) > self.threshold
        return mask.astype("float32")[..., 0]
