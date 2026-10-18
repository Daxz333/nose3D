import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from face_detector.utils import crop_face_square
from face_detector.utils import resize_image_with_crop_or_pad
from face_detector import FaceDetector
from segmentation import FaceParsing
from segmentation import NoseSegmentation
from segmentation import FaceSegmentation
from utils import logger
from utils import timer
from aligner import LossFunction
from aligner import AffineSolver
from blender import alpha_blender_v3
from blender import alpha_blender_v4
from blender import alpha_blender_v5
from blender import alpha_blender_v6
from blender import poisson_blender
from region_adapter import ADAPTED_REGIONS, replace_region
from region_adapter import RegionResult
from nose_pose import needs_frontalisation, pose_score, transfer_profile_nose

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@tf.function
def extension_nose_mask(mask, factor=1.2):
    shape = mask.shape
    pts = tf.stack(tf.meshgrid(tf.range(shape[0]), tf.range(shape[1])), axis=-1)[
        mask != 0
    ]
    bottom = tf.reduce_max(pts[::, 1])
    mask2 = mask[:bottom]
    mask2 = tf.image.resize(
        mask2[..., None],
        [tf.cast(tf.cast(bottom, tf.float32) * factor, tf.int32), shape[1]],
    )[..., 0]
    mask2 = tf.pad(mask2, [[0, shape[0] - bottom], [0, 0]])[-shape[0] :]
    mask2 = tf.cast((mask != 0) | (mask2 != 0), mask2.dtype)
    return mask2


def extension_region_mask(mask, factor=1.2, mode="scale"):
    """

    Brief:
        按部位扩展掩码，返回TensorFlow张量（调用方统一 .numpy()）
    Args:
        mask:二值掩码
        factor:扩展幅度。'scale'模式下为纵向拉伸倍数，'dilate'模式下为膨胀半径（像素）
        mode:'scale'为鼻子使用的纵向拉伸扩展，'dilate'为通用膨胀扩展，'none'为不扩展
    Returns:
        扩展之后的掩码张量
    """
    if mode == "none":
        return tf.convert_to_tensor(mask, dtype=tf.float32)
    if mode == "dilate":
        radius = max(int(round(float(factor))), 0)
        if radius == 0:
            return tf.convert_to_tensor(mask, dtype=tf.float32)
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype="float32")
        dilated = cv2.dilate(np.asarray(mask).astype("float32"), kernel)
        return tf.convert_to_tensor(dilated, dtype=tf.float32)
    return extension_nose_mask(mask, factor)


# ---------------------------------------------------------------------------
# 部位 -> BiSeNet 19 类解析类别 ID 的映射
# 已用 images/ 下的样例图实测核实过类别分布（2/3 眉毛、4/5 眼睛、10 鼻子、11 口腔、12/13 上下唇）
# ---------------------------------------------------------------------------
REGION_CLASS_MAP = {
    "nose": [10],
    "lip": [12, 13],
    "u_lip": [12],
    "l_lip": [13],
    "mouth": [11],
    "eye": [4, 5],
    "l_eye": [4],
    "r_eye": [5],
    "brow": [2, 3],
    "l_brow": [2],
    "r_brow": [3],
}

REGION_LABELS = {
    "nose": "鼻子",
    "lip": "嘴唇",
    "u_lip": "上唇",
    "l_lip": "下唇",
    "mouth": "口腔",
    "eye": "眼睛",
    "l_eye": "左眼",
    "r_eye": "右眼",
    "brow": "眉毛",
    "l_brow": "左眉",
    "r_brow": "右眉",
}

# 换鼻的原始参数，作为所有部位的基线（保证 nose 行为与改造前完全一致）
DEFAULT_BLEND_CONFIG = dict(
    blender="alpha_v6",
    meaner="local",
    extension_mode="scale",
    extension_factor=1.2,
    nose_mask_dilate=40,
    blended_mask_erode=20,
    blended_mask_blur=71,
    clean_threshold=40,
    aligned_color_ksize=111,
    mask_pool=20,
    remove_eyes=True,
    fill_mode="region_mean",
    source_fill_blur=51,
)

# 眼睛是高频细节部位：掩码小、容错低，必须关掉「去眼睛」步骤、
# 缩小膨胀/腐蚀/羽化尺度，并把颜色对齐的窗口收到眼睛尺度，否则细节会被肤色平均掉
EYE_BLEND_CONFIG = dict(
    DEFAULT_BLEND_CONFIG,
    extension_mode="dilate",
    extension_factor=4,
    nose_mask_dilate=10,
    blended_mask_erode=3,
    blended_mask_blur=15,
    clean_threshold=45,
    aligned_color_ksize=31,
    mask_pool=10,
    remove_eyes=False,
    fill_mode="skin_mean",
    source_fill_blur=21,
)

# 眉毛/嘴唇/口腔：形状比眼睛规整，但仍需保留边界，羽化尺度取鼻子与眼睛之间
SOFT_REGION_BLEND_CONFIG = dict(
    DEFAULT_BLEND_CONFIG,
    extension_mode="dilate",
    extension_factor=3,
    nose_mask_dilate=15,
    blended_mask_erode=4,
    blended_mask_blur=25,
    clean_threshold=45,
    aligned_color_ksize=51,
    mask_pool=10,
    remove_eyes=False,
    fill_mode="skin_mean",
    source_fill_blur=31,
)

REGION_BLEND_CONFIG = {
    "nose": dict(DEFAULT_BLEND_CONFIG),
    "eye": dict(EYE_BLEND_CONFIG),
    "l_eye": dict(EYE_BLEND_CONFIG),
    "r_eye": dict(EYE_BLEND_CONFIG),
    "lip": dict(SOFT_REGION_BLEND_CONFIG),
    "u_lip": dict(SOFT_REGION_BLEND_CONFIG),
    "l_lip": dict(SOFT_REGION_BLEND_CONFIG),
    "mouth": dict(SOFT_REGION_BLEND_CONFIG),
    "brow": dict(SOFT_REGION_BLEND_CONFIG),
    "l_brow": dict(SOFT_REGION_BLEND_CONFIG),
    "r_brow": dict(SOFT_REGION_BLEND_CONFIG),
}


def get_region_config(region):
    """
    Brief:取得某个部位的融合参数，未登记的部位回退到换鼻参数
    """
    return dict(REGION_BLEND_CONFIG.get(region, DEFAULT_BLEND_CONFIG))


class RegionWarper(object):
    """
    替换人脸图片中某个部位（鼻子/眼睛/嘴唇/眉毛/口腔）的方法类

    换鼻(NoseWarper)只是本类 region='nose' 的一个特例，二者共用同一套
    检测 -> 解析 -> 关键点 -> 仿射对齐 -> 融合 流程，差异只有两点：
        1. 解析掩码取哪几个类别 ID（REGION_CLASS_MAP）
        2. 融合参数（REGION_BLEND_CONFIG）
    """

    def __init__(
        self,
        face_detector_path,
        face_parsing_path=None,
        face_segmentation_path=None,
        region="nose",
        class_ids=None,
        prob=0.5,
        learning_rate=0.2,
        decay=0.99,
        iteration=150,
        threshold=0.4,
        downsample_ratio=0.5,
        nose_segmentation_path=None,
    ):
        """

        Brief:
            构造函数，构造一个RegionWarper对象
        Args:
            face_detector_path:人脸检测(获得bound box)模型的权重地址
            face_parsing_path:人脸解析(获得部位二值掩码)模型权重地址
            face_segmentation_path:人脸分割(获得二值掩码)模型权重地址
            region:待替换的部位，取值见REGION_CLASS_MAP，默认nose
            class_ids:自定义类别ID列表，传入时优先于region对应的默认类别
            prob:对于关键点进行随机采样的概率
            learning_rate:模型优化的学习率
            decay:学习率衰减策略，每次迭代之后的学习率衰减系数
            iteration:模型训练的迭代次数
            threshold:人脸分割模型的阈值，大于该阈值设置为1，否则为0
            downsample_ratio:人脸分割的时候，进行下采样的倍率
            nose_segmentation_path:兼容旧参数名，等价于face_parsing_path
        Returns:
            一个RegionWarper对象
        """
        if face_parsing_path is None:
            face_parsing_path = nose_segmentation_path
        self.prob = prob
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.decay = decay
        self.solve_affine_matrix_v = False
        self.solve_affine_matrix_gif = False
        self.result_contrast_v = False
        self.blend_v = False
        self.dpi=400
        self.set_region(region, class_ids=class_ids)
        with timer("加载人脸检测模型"):
            self.face_detector = FaceDetector(face_detector_path)
        with timer("加载人脸解析模型"):
            self.face_parsing = FaceParsing(
                face_parsing_path, 19, class_ids=self.class_ids
            )
        with timer("加载人脸分割模型"):
            self.face_segmentation = FaceSegmentation(
                face_segmentation_path,
                downsample_ratio=downsample_ratio,
                threshold=threshold,
            )

    def set_region(self, region, class_ids=None):
        """

        Brief:
            切换待替换的部位，只改变解析类别与融合参数，模型权重无需重新加载
        Args:
            region:部位名称，取值见REGION_CLASS_MAP
            class_ids:自定义类别ID列表
        """
        if region not in REGION_CLASS_MAP:
            raise ValueError(
                "不支持的部位:%s，可选:%s"
                % (region, "/".join(sorted(REGION_CLASS_MAP.keys())))
            )
        self.region = region
        self.class_ids = list(class_ids) if class_ids else list(REGION_CLASS_MAP[region])
        self.blend_config = get_region_config(region)
        # 解析模型是共享的，切换部位时同步它的默认类别，避免取到上一次的类别
        face_parsing = getattr(self, "face_parsing", None)
        if face_parsing is not None:
            face_parsing.class_ids = tuple(self.class_ids)
        return self

    @property
    def region_label(self):
        return REGION_LABELS.get(self.region, self.region)

    # 兼容旧属性名 nose_segmentation
    @property
    def nose_segmentation(self):
        return self.face_parsing

    def get_bbox(self, path=None, image=None):
        """

        Brief:
            获得人脸的子图区域，这里格式和人脸检测模型输出bbox不一致，
            这里的bbox格式为(height_begin,height_end,width_begin,width_end)
        Args:
            path:RGB格式的图片文件的路径
            image:RGB格式的图片,如果path为None,就会直接使用image.
        Returns:
            人脸框的切片坐标起始位置
        """
        if path is not None:
            image = np.array(Image.open(path))
        rimage = resize_image_with_crop_or_pad(image, 640)
        bbox, prob = self.face_detector.inference(rimage)
        face, bbox = crop_face_square(rimage, bbox)
        return bbox

    def get_mask(self, path=None, image=None, class_ids=None):
        """

        Brief:
            通过FaceDetector切分得到包含人脸的区域，并在这个区域计算目标部位的掩码
            输入的图片会在保持长宽比例不变的情况下被适当的缩放为640x640，也就是先把长方形的图片用黑色边填充为正方形，然后再用插值法缩放到640x640
            得到的人脸框(bound box)也会被reshape为一个正方形区域，得到一个包含人脸的正方形框
            这个正方形框也会以同样的方式被缩放到512x512
            然后在这个区域计算目标部位(self.class_ids对应的类别)的掩码
        Args:
            path:RGB格式的图片文件的路径
            image:RGB格式的图片,如果path为None,就会直接使用image.
            class_ids:临时指定的类别ID列表，None表示使用当前部位对应的类别
        Returns:
            切分出来的脸部图片,图片中目标部位的掩码
        """
        if path is not None:
            image = np.array(Image.open(path))
        rimage = resize_image_with_crop_or_pad(image, 640)
        bbox, prob = self.face_detector.inference(rimage)
        face, bbox = crop_face_square(rimage, bbox)
        face = resize_image_with_crop_or_pad(face, 512)
        mask = self.face_parsing.inference(face, class_ids or self.class_ids)
        return face, mask, bbox

    def get_parsing(self, image):
        rimage = resize_image_with_crop_or_pad(image, 640)
        bbox, _ = self.face_detector.inference(rimage)
        if bbox is None:
            raise ValueError("未检测到人脸，请使用清晰的正面照片")
        face, bbox = crop_face_square(rimage, bbox)
        if face.size == 0:
            raise ValueError("人脸裁剪区域为空，请重新选择图片")
        face = resize_image_with_crop_or_pad(face, 512)
        return face, self.face_parsing.parsing_map(face), bbox

    def get_pose_parsing(self, image):
        rimage = resize_image_with_crop_or_pad(image, 640)
        bbox, _, points = self.face_detector.inference(rimage, return_landmarks=True)
        face, bbox = crop_face_square(rimage, bbox)
        if face.size == 0:
            raise ValueError("人脸裁剪区域为空，请重新选择图片")
        face = resize_image_with_crop_or_pad(face, 512)
        return face, self.face_parsing.parsing_map(face), points

    def warp_nose(self, face_image, source_image, return_details=False, add_bg=False):
        target, target_parsing, target_points = self.get_pose_parsing(face_image)
        source, source_parsing, source_points = self.get_pose_parsing(source_image)
        if needs_frontalisation(source_points, target_points):
            result = transfer_profile_nose(target, source, target_parsing, source_parsing, pose_score(source_points))
        else:
            image = self.warp(face_image, source_image, region="nose", add_bg=add_bg, pose_correction=False)
            result = RegionResult(image, None, ())
        return result if return_details else result.image

    def warp_adapted(self, face_image, source_image, region, return_details=False):
        with timer("解析并替换%s" % REGION_LABELS[region]):
            source, source_parsing, _ = self.get_parsing(source_image)
            target, target_parsing, _ = self.get_parsing(face_image)
            result = replace_region(target, source, target_parsing, source_parsing, region)
        return result if return_details else result.image

    @tf.function
    def get_keypoints(self, mask, prob=0.9):
        """

        Brief:
            通过掩码生成关键点
            输入一张掩码图片，通过拉普拉斯算子得到二阶微分，将其通过sigmoid函数进行归一化，并设定阈值0.5进行二值化
            得到鼻子掩码的边缘信息。然后据此得到所有边缘像素点的坐标，随机进行采样（按照prob的概率）得到关键点，用于图片对齐
        Args:
            prob:每个点被选中的概率
        Returns:
            关键点构成的NX3的矩阵，每个点的坐标为(x,y,1),最后一个变量没有什么实际含义，永远为1，主要是为了方便矩阵运算，参考仿射变换的计算过程
        """
        grad = tf.nn.conv2d(
            input=mask[None, ..., None],
            filters=np.array([1, 0, 1, 0, -4, 0, 1, 0, 1]).reshape(
                [3, 3, 1, 1]
            ),  # 通过拉普拉斯算子得到二阶微分
            strides=[1, 1, 1, 1],
            padding="SAME",
        )[0, ..., 0]
        edge_mask = tf.nn.sigmoid(grad) < 0.5  # 二值化处理，得到边缘的掩码
        points = tf.stack(
            tf.meshgrid(tf.range(mask.shape[0]), tf.range(mask.shape[1])), axis=-1
        )  # 得到图像每个点的坐标矩阵
        keypoints = tf.boolean_mask(points, grad)  # 选出所有的边缘点
        length = tf.cast(tf.cast(tf.shape(keypoints)[0], tf.float32) * prob, tf.int32)
        keypoints = tf.random.shuffle(keypoints)[:length]
        keypoints3d = tf.concat(
            [tf.cast(keypoints, tf.float32), tf.ones([tf.shape(keypoints)[0], 1])],
            axis=-1,
        )  # (x,y)->(x,y,1) OpenCV说仿射变换需要3x3的矩阵
        return tf.cast(keypoints3d, tf.float32)

    def __call__(
        self,
        face_image,
        nose_image,
        blender=None,
        meaner=None,
        add_bg=False,
        extension_factor=None,
        region=None,
    ):
        """

        Brief:
            替换人脸中某个部位算法的具体逻辑代码
        Args:
            face_image:被替换部位的人脸图片(RGB格式)
            nose_image:提供部位来源的人脸图片(RGB格式)，也可用关键字source_image传入
            blender:融合算法，None表示使用部位配置表中的默认值
            meaner:颜色对齐方式，None表示使用部位配置表中的默认值
            add_bg:是否补充背景
            extension_factor:掩码扩展系数，None表示使用部位配置表中的默认值
            region:本次调用使用的部位，None表示沿用当前部位
        Returns:
            替换部位之后的图片，只包括人脸框的局部部分，如果要得到完整的图片，参考embeded_back方法
        """
        return self.warp(
            face_image,
            nose_image,
            blender,
            meaner,
            add_bg,
            extension_factor,
            region=region,
        )

    @timer
    def warp(
        self,
        face_image,
        nose_image,
        blender=None,
        meaner=None,
        add_bg=False,
        extension_factor=None,
        region=None,
        pose_correction=True,
    ):
        """

        Brief:
            替换人脸中某个部位算法的具体逻辑代码
        Args:
            face_image:被替换部位的人脸图片(RGB格式)
            nose_image:提供部位来源的人脸图片(RGB格式)
            blender:融合算法，None表示使用部位配置表中的默认值
            meaner:颜色对齐方式，None表示使用部位配置表中的默认值
            add_bg:是否补充背景
            extension_factor:掩码扩展系数，None表示使用部位配置表中的默认值
            region:本次调用使用的部位，None表示沿用当前部位
        Returns:
            替换部位之后的图片，只包括人脸框的局部部分，如果要得到完整的图片，参考embeded_back方法
        """
        # 部位解析为局部变量，不修改共享状态，保证并发请求之间互不影响
        if region is None:
            region = self.region
        if region not in REGION_CLASS_MAP:
            raise ValueError(
                "不支持的部位:%s，可选:%s"
                % (region, "/".join(sorted(REGION_CLASS_MAP.keys())))
            )
        config = get_region_config(region)
        if region == "nose" and blender is None and meaner is None and extension_factor is None and pose_correction:
            return self.warp_nose(face_image, nose_image, add_bg=add_bg)
        if region in ADAPTED_REGIONS and blender is None:
            return self.warp_adapted(face_image, nose_image, region)
        class_ids = REGION_CLASS_MAP[region]
        label = REGION_LABELS.get(region, region)
        # 未显式指定时，全部回退到部位配置表（nose 的配置与改造前的硬编码完全一致）
        if blender is None:
            blender = config["blender"]
        if meaner is None:
            meaner = config["meaner"]
        if extension_factor is None:
            extension_factor = config["extension_factor"]
        extension_mode = config["extension_mode"]
        with timer("获得人脸切片和对应%s位置的掩码" % (label)):
            nose_image, nose_mask, _ = self.get_mask(image=nose_image, class_ids=class_ids)  # 获得人脸图片及其对应部位掩码
            face_image, face_mask, _ = self.get_mask(image=face_image, class_ids=class_ids)

        with timer("通过掩码计算关键点"):
            nose_keypoint = self.get_keypoints(nose_mask, self.prob)  # 通过掩码计算关键点
            face_keypoint = self.get_keypoints(face_mask, self.prob)
            if len(nose_keypoint) == 0:
                raise ValueError("Source Image 未检测到%s掩码" % (label))
            if len(face_keypoint) == 0:
                raise ValueError("Face Image 未检测到%s掩码" % (label))
        with timer("求解人脸对齐变换矩阵"):
            affine_model = AffineSolver(nose_image.shape)  # 初始化仿射变换模型
            affine_model.compile(
                loss=LossFunction(face_keypoint, nose_keypoint),
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            )
            affine_model.fit(
                nose_keypoint,
                face_keypoint,
                visualization=self.solve_affine_matrix_gif,
                iteration=self.iteration,
                lr_decay=self.decay,
            )  # 最优化仿射变换参数
        with timer("对齐两张人脸中的%s" % (label)):
            nose_image_affined = affine_model.transform(
                nose_image
            )  # 根据求解出来的变换矩阵对图片进行仿射变换,使得两张图片目标部位对齐
            nose_mask_affined = affine_model.transform(nose_mask)
            nose_keypoint_affined = affine_model.transform(nose_keypoint)
        mixed = None
        with timer("进行图像融合"):
            if blender == "alpha_v3":
                face_seg_mask = self.face_segmentation.inference(face_image)
                nose_seg_mask = self.face_segmentation.inference(nose_image_affined)
                mixed = alpha_blender_v3(
                    face_image=face_image,
                    nose_image=nose_image_affined,
                    face_mask=face_mask,
                    nose_mask=nose_mask_affined,
                    face_seg_mask=face_seg_mask,
                    nose_seg_mask=nose_seg_mask,
                    inpaint=add_bg,
                    blended_mask_erode=10,
                    nose_mask_dilate=30,
                    nose_mask_blur=51,
                    overbright_threshold=230,
                    visualization=self.blend_v,
                )
            elif blender == "alpha_v4":
                face_seg_mask = self.face_segmentation.inference(face_image)
                nose_seg_mask = self.face_segmentation.inference(nose_image_affined)
                mixed = alpha_blender_v4(
                    face_image=face_image,
                    nose_image=nose_image_affined,
                    face_mask=face_mask,
                    nose_mask=nose_mask_affined,
                    face_seg_mask=face_seg_mask,
                    nose_seg_mask=nose_seg_mask,
                    inpaint=add_bg,
                    nose_mask_dilate=20,
                    blended_mask_erode=0,
                    blended_mask_blur=71,
                    overbright_threshold=256,
                    aligned_color_ksize=111,
                    mean_method=meaner,
                    visualization=self.blend_v,
                )
            elif blender == "alpha_v5":
                face_seg_mask = self.face_segmentation.inference(face_image)
                nose_seg_mask = self.face_segmentation.inference(nose_image_affined)
                nose_mask_affined = extension_nose_mask(
                    nose_mask_affined, factor=1.2
                ).numpy()
                face_mask = extension_nose_mask(face_mask, factor=1.2).numpy()
                mixed = alpha_blender_v5(
                    face_image=face_image,
                    nose_image=nose_image_affined,
                    face_mask=face_mask,
                    nose_mask=nose_mask_affined,
                    face_seg_mask=face_seg_mask,
                    nose_seg_mask=nose_seg_mask,
                    inpaint=add_bg,
                    nose_mask_dilate=40,
                    blended_mask_erode=25,
                    blended_mask_blur=71,
                    clean_threshold=40,
                    aligned_color_ksize=111,
                    mean_method=meaner,
                    visualization=self.blend_v,
                )
            elif blender == "alpha_v6":
                face_seg_mask = self.face_segmentation.inference(face_image)
                nose_seg_mask = self.face_segmentation.inference(nose_image_affined)

                # plt.imshow(nose_mask_affined)
                # plt.title("old")
                # plt.show()

                nose_mask_affined = extension_region_mask(
                    nose_mask_affined, factor=extension_factor, mode=extension_mode
                ).numpy()
                face_mask = extension_region_mask(
                    face_mask, factor=extension_factor, mode=extension_mode
                ).numpy()

                # plt.imshow(nose_mask_affined)
                # plt.title("new")
                # plt.show()

                mixed = alpha_blender_v6(
                    face_image=face_image,
                    nose_image=nose_image_affined,
                    face_mask=face_mask,
                    nose_mask=nose_mask_affined,
                    face_seg_mask=face_seg_mask,
                    nose_seg_mask=nose_seg_mask,
                    face_keypoint=face_keypoint,
                    nose_keypoint=nose_keypoint_affined,
                    inpaint=add_bg,
                    nose_mask_dilate=config["nose_mask_dilate"],
                    blended_mask_erode=config["blended_mask_erode"],
                    blended_mask_blur=config["blended_mask_blur"],
                    clean_threshold=config["clean_threshold"],
                    aligned_color_ksize=config["aligned_color_ksize"],
                    mean_method=meaner,
                    remove_eyes=config["remove_eyes"],
                    fill_mode=config["fill_mode"],
                    source_fill_blur=config["source_fill_blur"],
                    mask_pool=config["mask_pool"],
                    visualization=self.blend_v,
                )
            elif blender == "poisson_v1":
                face_seg_mask = self.face_segmentation.inference(face_image)
                nose_seg_mask = self.face_segmentation.inference(nose_image_affined)
                mixed = poisson_blender(
                    face_image=face_image,
                    nose_image=nose_image_affined,
                    face_mask=face_mask,
                    nose_mask=nose_mask_affined,
                    face_seg_mask=face_seg_mask,
                    nose_seg_mask=nose_seg_mask,
                    inpaint=add_bg,
                    nose_mask_dilate=30,
                    blended_mask_erode=10,
                    nose_mask_blur=51,
                    overbright_threshold=256,
                    visualization=self.blend_v,
                )
            else:
                raise ValueError("No Such Blender")
            # 仅仅作为调试代码用，后面会删除他
            self.buffer = type(
                "",
                (object,),
                dict(
                    nose_image=nose_image_affined,
                    face_image=face_image,
                    nose_mask=nose_mask_affined,
                    face_mask=face_mask,
                    blended_image=mixed,
                    nose_keypoint=nose_keypoint_affined,
                    face_keypoint=face_keypoint,
                    affine=affine_model,
                    face_seg_mask=face_seg_mask,
                    nose_seg_mask=nose_seg_mask,
                ),
            )

        if self.solve_affine_matrix_v:
            plt.style.use("default")
            plt.figure(dpi=self.dpi//100*3)
            plt.subplot(2, 3, 1)
            plt.imshow(nose_image)
            plt.imshow(nose_mask, alpha=0.1)
            # plt.scatter(nose_keypoint[::, 0], nose_keypoint[::, 1], s=0.001, color="r")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(face_image)
            plt.imshow(face_mask, alpha=0.1)
            # plt.scatter(face_keypoint[::, 0], face_keypoint[::, 1], s=0.001, color="r")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(nose_image_affined)
            plt.imshow(nose_mask_affined, alpha=0.1)
            # plt.scatter(
            #    nose_keypoint_affined[::, 0],
            #    nose_keypoint_affined[::, 1],
            #    s=0.001,
            #    color="r",
            # )
            plt.axis("off")

            plt.subplot(2, 3, 4)
            nose_mask_affined_rgb = np.stack(
                [
                    nose_mask_affined,
                    np.zeros_like(nose_mask_affined),
                    np.zeros_like(nose_mask_affined),
                ],
                axis=-1,
            )
            plt.imshow(nose_mask_affined_rgb)
            plt.axis("off")

            plt.subplot(2, 3, 5)
            face_mask_rgb = np.stack(
                [np.zeros_like(face_mask), face_mask, np.zeros_like(face_mask)], axis=-1
            )
            plt.imshow(face_mask_rgb)
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(face_mask, cmap="Greens")
            plt.imshow(nose_mask_affined_rgb + face_mask_rgb, cmap="Reds")
            plt.axis("off")
        if self.result_contrast_v:
            plt.figure(dpi=self.dpi)

            plt.subplot(1, 3, 1)
            plt.imshow(nose_image_affined)
            plt.title("Nose Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(face_image)
            plt.title("Face Image")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(mixed)
            plt.title("Blended Image")
            plt.axis("off")
        return mixed

    def embedded_back(self, face_image, replaced_nose_crop):
        """

        Brief:
            将替换比之后的切片图片嵌入到原图中
        Args:
            face_image:最原始的人脸图
            replaced_nose_crop:鼻子替换之后的人脸图切片
        Returns:
            完整的修改了鼻子之后的图片

        """
        threshold = 20
        face_image = resize_image_with_crop_or_pad(face_image, 640)
        bbox = self.get_bbox(image=face_image)
        face_image[bbox[0] : bbox[1], bbox[2] : bbox[3]] = cv2.resize(
            replaced_nose_crop, [bbox[1] - bbox[0], bbox[3] - bbox[2]]
        )
        gray = face_image.mean(axis=-1)
        # 去除黑边
        nrow = gray.shape[0]
        ncol = gray.shape[1]
        rowc = gray[:, int(1 / 2 * nrow)]
        colc = gray[int(1 / 2 * ncol), :]
        rowflag = np.argwhere(rowc > threshold)
        colflag = np.argwhere(colc > threshold)
        # resize到原来的尺寸
        left, bottom, right, top = (
            rowflag[0, 0],
            colflag[-1, 0],
            rowflag[-1, 0],
            colflag[0, 0],
        )
        image = face_image[left:right, top:bottom]
        return image

    def set_visualization(
        self,
        solve_affine_matrix=False,
        solve_affine_matrix_gif=False,
        result_contrast=False,
        blend=False,
        dpi=400
    ):
        self.solve_affine_matrix_v = solve_affine_matrix
        self.solve_affine_matrix_gif = solve_affine_matrix_gif
        self.result_contrast_v = result_contrast
        self.blend_v = blend,
        self.dpi=dpi


class NoseWarper(RegionWarper):
    """
    替换人脸图片中的鼻子的方法类

    保留为 RegionWarper 的薄封装（region 固定为 nose），
    保证所有旧代码（包括 flask_server 的构造方式）零改动即可继续工作
    """

    def __init__(
        self,
        face_detector_path,
        nose_segmentation_path=None,
        face_segmentation_path=None,
        **kwargs,
    ):
        kwargs.pop("region", None)
        kwargs.pop("face_parsing_path", None)
        super().__init__(
            face_detector_path,
            face_parsing_path=nose_segmentation_path,
            face_segmentation_path=face_segmentation_path,
            region="nose",
            **kwargs,
        )
