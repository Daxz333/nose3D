import cv2
import numpy as np
import scipy.sparse
import pyamg
from utils import timer
from utils import deprecated
from utils import logger
import tensorflow as tf
import matplotlib.pyplot as plt


def get_center(mask):
    """

    Brief:获得掩码的中心坐标
    Args:
        mask:二值掩码,大小为(height,width),最好为uint8的数据格式
    """
    box = cv2.boundingRect(mask.astype("uint8"))
    center_x = box[0] + box[2] // 2
    center_y = box[1] + box[3] // 2
    center = (center_x, center_y)
    return center


@tf.function
def align_color(image1, image2, mask1, mask2):
    """

    Brief:对齐颜色，借助TensorFlow-API进行GPU并行计算加速，把image1的颜色修改为image2的颜色，同时轮廓和内容尽可能不变
    Args:
        image1:RGB格式的图片，需要被修改颜色的图片
        image2:RGB格式的图片，颜色参考图片
        mask1:image1对应的鼻子掩码区域
        mask2:image2对应的鼻子掩码区域
    Returns:
        对齐颜色之后的图片，RGB格式
    """
    image1 = tf.cast(image1, dtype=tf.float32)
    image2 = tf.cast(image2, dtype=tf.float32)
    mask1 = tf.cast(mask1, tf.bool)
    mask2 = tf.cast(mask2, tf.bool)
    mean1, std1 = tf.nn.moments(tf.boolean_mask(image1, mask1), axes=[0])
    mean2, std2 = tf.nn.moments(tf.boolean_mask(image2, mask2), axes=[0])
    var1, var2 = std1**0.5, std2**0.5
    image1 = (image1 - mean1) / var1 * var2 + mean2
    image1 = tf.clip_by_value(image1, 0, 255)
    image1 = tf.cast(image1, tf.uint8)
    return image1


def align_color_avg_pool(
    img1,
    img2,
    mask1,
    mask2,
    pool_ksize=31,
    threshold=210,
    inpaint_width=10,
    overbright_dilate=5,
    visualization=False,
):
    """

    Brief:
        修改人脸颜色，让img1图片颜色变为img2图片颜色
        采用平均池化来求解局部均值和方差，而不是直接对整个图求解均值方差
        以适应局部颜色分布变化
    Args:
        img1:需要修改颜色的图片，修改为img2的颜色
        img2:作为参考颜色的图片
        mask1:img1中的鼻子掩码
        mask2:img2中的鼻子掩码
        pool_ksize:求局部均值的平均池化的池化和大小
        threshold:亮度阈值,如果某一个位置的灰度值大于阈值，则认为是过度曝光的区域，会影响图片融合效果，因此这部分像素会被临近至填充
        inpaint_width:上述的填充参考的临近像素范围
        overbright_dilate:对过度曝光的区域的掩码金进行膨胀
    Returns:
        调整颜色后的图片
    """
    img1 = img1.astype("float32")
    img2 = img2.astype("float32")
    mask1 = mask1.astype("bool")
    mask2 = mask2.astype("bool")
    if visualization:
        plt.figure(dpi=400)
        plt.subplot(1, 3, 1)
        plt.imshow(img1 / 255)
        plt.axis("off")
        plt.title("img1 raw")

    def mean(x, mask=None):
        if mask is not None:
            x = x.copy()
            x[mask == 0] = x[mask == 1].reshape([-1, 3]).mean(axis=0)
        if len(x.shape) == 3:
            x = x[None, ...]
        avg_pool = tf.keras.layers.AveragePooling2D(
            [pool_ksize, pool_ksize], [1, 1], padding="same"
        )
        return avg_pool(x.astype("float32")).numpy()[0]

    img1_mean = mean(img1, mask1)
    img1_std = (mean((img1_mean - img1) ** 2, mask1)) ** 0.5
    img2_mean = mean(img2, mask2)
    img2_std = (mean((img2_mean - img2) ** 2, mask2)) ** 0.5
    img1 = (img1 - img1_mean) / img1_std * img2_std + img2_mean
    img1 = img1.clip(0, 255).astype("uint8")
    gray = img1.mean(axis=-1)
    overbright = ((mask1) & (gray > threshold)).astype("uint8")
    kernel = np.ones((overbright_dilate, overbright_dilate), np.uint8)
    overbright = cv2.dilate(overbright, kernel)
    img1 = cv2.inpaint(img1, overbright.astype("uint8"), inpaint_width, cv2.INPAINT_NS)
    if visualization:
        plt.subplot(1, 3, 2)
        plt.imshow(img2_mean / 255)
        plt.title("img2 mean")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(img1 / 255)
        plt.title("img1 aligned")
        plt.axis("off")

    return img1


@tf.function
def momentum(img, mask, ksize=71):
    """

    Brief:
        求出图片的局部均值和方差，直接对整个图片或者整个鼻子求均值存在的问题在于，并不是鼻子的所有位置的分布都是相同的
        所以这样会来带一定的偏差，解决办法就是采用滑动窗口，局部求解均值，来代替全局均值
        采用了平均池化来完成这个操作，考虑到鼻子本身，并不是规则的正方形，除了鼻子以外的其他区域不应该参与计算均值
        因此需要移除他们的影响，对掩码为0的区域，图像用0填充覆盖。
        考虑到对于池化核滑动的ksize x ksize区域，对掩码求和（池化之后的结果total乘以ksize*ksize)
        得到的就是有多少属于鼻子的像素，而对于图像求均值，需要排除掉0的影响，平均池化之后的结果为mean，那么重新计算的均值应该为
        mean=mean*ksize*ksize/(total*ksize*ksize+1e-9)=mean/(total+1e-9)
        对于方差而言，就是对于输入x和均值的方差求和std=avg_pool[(x-mean)**2]
        但是同样没有考虑到0带来的影响，因此进行如下修正
        std=(std*ksize*ksize-(1-total)*ksize*ksize*mean)/(total+e1-9)=(std-(1-total)*mean)/(total+1e-9)
        开方得到方差：var=mean**0.5
        得到最终的均值方差
    Args:
        img:包含鼻子的人脸图片,RG格式
        mask:鼻子位置掩码，0/1二值掩码
        ksize:池化核大小
    Returns:
        img图像的局部均值和方差
    """
    img = tf.cast(img, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    img2 = img * mask[..., None]

    mean = tf.nn.avg_pool2d(
        mask[..., None] * img[None, ...],
        [1, ksize, ksize, 1],
        [1, 1, 1, 1],
        padding="SAME",
    )
    img_global_mean = tf.reduce_mean(tf.boolean_mask(img, mask == 1), axis=0)
    total = tf.nn.avg_pool2d(
        mask[None, ..., None], [1, ksize, ksize, 1], [1, 1, 1, 1], padding="SAME"
    )
    mean = mean / (total + 1e-8)
    mean = tf.clip_by_value(mean, 0, 256)
    mean = mask[..., None] * mean
    mean = mask[..., None] * mean + (1 - mask[..., None]) * img_global_mean
    std = (
        tf.nn.avg_pool2d(
            (img[None, ...] - mean) ** 2,
            [1, ksize, ksize, 1],
            [1, 1, 1, 1],
            padding="SAME",
        )
        ** 0.5
    )
    mean = tf.squeeze(mean)
    std = tf.squeeze(std)
    mean = mean * mask[..., None]
    std = std * mask[..., None] + (1 - mask[..., None])
    return mean, std**0.5


@tf.function
def align_color_avg_pool_v2(img1, img2, mask, ksize=91, pool=20):
    """

    Brief:对齐颜色，借助TensorFlow-API进行GPU并行计算加速，把image1的颜色修改为image2的颜色，同时轮廓和内容尽可能不变
    相比于align_color，此API采用了平均池化来代替对整个鼻子求出一个均值方差，使得鼻子局部颜色对其表现更加好
    Args:
        image1:RGB格式的图片，需要被修改颜色的图片
        image2:RGB格式的图片，颜色参考图片
        mask1:至少包括两个图片目标部位位置并集的掩码，代表需要对其颜色的同区域
        ksize:局部均值方差的滑动窗口大小，部位越小（例如眼睛）取值越小，否则细节会被周围肤色平均掉
        pool:对掩码做最大值池化的核大小，作用是把统计区域向外扩张，避免边界处统计不到目标部位
    Returns:
        对齐颜色之后的图片，RGB格式
    """
    mask = tf.nn.max_pool2d(
        tf.cast(mask, dtype=tf.float32)[None, ..., None],
        [1, pool, pool, 1],
        [1, 1, 1, 1],
        padding="SAME",
    )[0, ..., 0]
    img1 = tf.cast(img1, tf.float32)
    img2 = tf.cast(img2, tf.float32)
    img1_mean, img1_var = momentum(img1, mask, ksize)
    img2_mean, img2_var = momentum(img2, mask, ksize)
    img3 = (img1 - img1_mean) / img1_var * img2_var + img2_mean
    img3 = tf.clip_by_value(img3, 0, 255)
    img3 = tf.cast(img3, dtype=tf.uint8)
    return img3


@timer
def remove_eyes_region(
    nose_image,
    nose_mask,
    blended_mask,
    nose_keypoint,
    mean,
    threshold,
    dilate_and_erode=True,
    visualization=False,
):
    mean = nose_image[blended_mask != 0].mean(axis=0)
    delta = np.mean((nose_image - mean) ** 2, axis=-1) ** 0.5
    half = tf.cast(tf.reduce_mean(nose_keypoint, axis=0)[1], tf.int32).numpy()

    if visualization:
        plt.subplot(1, 2, 1)
        mask = blended_mask.copy()
        mask[half:] = 0
        plt.imshow(delta * mask)
        plt.colorbar()
        plt.title("delta!!")
        plt.axis("off")

    delta = delta < threshold
    blended_mask_down = blended_mask[half:]
    blended_mask_up = blended_mask[:half]
    blended_mask_up = blended_mask_up * delta[:half]
    blended_mask2 = np.concatenate([blended_mask_up, blended_mask_down], axis=0)
    blended_mask2 = np.clip(
        blended_mask2 + cv2.erode(nose_mask, np.ones((5, 5), dtype="float32")), 0, 1
    )
    if dilate_and_erode:
        blended_mask2 = cv2.dilate(blended_mask2, np.ones((5, 5), dtype="float32"))
        blended_mask2 = cv2.erode(blended_mask2, np.ones((5, 5), dtype="float32"))
    if visualization:
        plt.subplot(1, 2, 2)
        plt.imshow(blended_mask2)
        plt.title("blended_mask!!!")
        plt.axis("off")
    return blended_mask2


@timer
def adjust_nose_union_mask(
    face_no_bg,
    nose_no_bg,
    face_mask,
    nose_mask,
    nose_union_mask,
    face_image_mean,
    threshold=40,
    erode_and_dilate=False,
):
    """

    Brief:
    获得一个mask,为融合的图像区域，这个区域为face_image和nose_image的交集
    实际场景中考虑到容错性，会对face_image计算得到的face_mask采用cv2.dilate进行膨胀
    但是这样的膨胀可能会导致比如在60度侧脸中，眼睛的黑色部分被引入的问题，会影响到融合效果
    因此在这里进行一定的修正，排除掉眼睛这个区域的融合，具体做法涉及到一种筛选数据中离群值的做法
    Args:
    Returns:
    """
    # 集合运算A-B=A∪B-B,nose_mask-face_mask的差集
    substract_mask = ((nose_union_mask[..., 0] == 1) & (nose_mask == 0)).astype(
        "float32"
    )
    delta1 = np.mean((nose_no_bg - face_no_bg) ** 2, axis=-1) ** 0.5
    delta2 = np.mean((nose_no_bg - face_image_mean) ** 2, axis=-1) ** 0.5
    delta = np.where(delta1 < delta2, delta1, delta2)

    substract_mask = (substract_mask * (delta < threshold) != 0) | (nose_mask != 0)
    substract_mask = substract_mask.astype("float32")
    # plt.subplot(1,2,1)
    # plt.imshow(substract_mask)
    if erode_and_dilate:
        substract_mask = cv2.dilate(substract_mask, np.ones((5, 5)))
        substract_mask = cv2.erode(substract_mask, np.ones((5, 5)))
    # plt.subplot(1,2,2)
    # plt.imshow(substract_mask)
    # plt.show()
    return substract_mask[..., None]


@timer
def poisson_blender(
    face_image,
    nose_image,
    face_mask,
    nose_mask,
    face_seg_mask,
    nose_seg_mask,
    inpaint=True,
    nose_mask_dilate=30,
    blended_mask_erode=10,
    nose_mask_blur=51,
    overbright_threshold=256,
    visualization=True,
):
    """
    Author:shijunfeng
    Brief:图像融合算法，传入的图片已经计算好鼻子掩码，并且经过了图像对其，这里是最后一步，也就是真正的替换鼻子算法的具体逻辑
    整个过程细节比较繁琐，详细看代码中的注释,该API采用了Poisson融合的算法思路
    Args:
        face_image:人脸图片，被替换鼻子的图片，RGB/uint8格式
        nose_image:鼻子图片，提供鼻子的图片，RGB/uint8格式
        face_mask:人脸图片的鼻子位置掩码（0代表非鼻子区域，1代表鼻子区域，其他掩码也是如此）
        nose_mask:鼻子图片的鼻子位置掩码
        face_seg_mask:人脸图片的人脸位置掩码
        nose_seg_mask:鼻子图片的鼻子位置掩码
        inpaint:bool，是否补充背景，这一步需要较长时间计算
        nose_mask_dilate:对鼻子掩码进行膨胀操作的宽度，可能掩码覆盖的区域不够，所以需要进行一次膨胀
        blended_mask_erode:对最终计算出来的替换鼻子的位置掩码，进行腐蚀的宽度
        blended_mask_blur:对最终计算出来的替换鼻子的位置掩码，进行高斯模糊的核大小
        overbright_threshold:对于过度曝光区域进行修正的阈值（暂时此参数无效）
        visualization:是否可视化中间结果，主要是方便代码调试用
    Returns:
        被替换鼻子之后的人脸图，RGB/uint8格式
    """
    logger.debug(
        "call poisson_blender_v1(...,inpaint=%s,visualization=%s)"
        % (inpaint, visualization)
    )
    # 可能本来的脸图中就有纯黑色，将其排除在外
    face_image[face_image.mean(axis=-1) == 0] = 1
    face_no_bg = face_image * face_seg_mask[..., None]
    nose_no_bg = nose_image * nose_seg_mask[..., None]
    # 求出替换鼻子的区域
    nose_mask_dilate = cv2.dilate(
        nose_mask.astype("uint8"),
        np.ones(shape=(nose_mask_dilate, nose_mask_dilate), dtype="float32"),
    )

    face_mask = face_mask.astype("bool")
    nose_mask = nose_mask.astype("bool")
    nose_mask_dilate = nose_mask_dilate.astype("bool")
    nose_union_mask = (nose_mask | face_mask).astype("float32")[..., None]

    # 第一次替换，目的是为了确定替换鼻子后的人脸精细掩码
    mixed = nose_no_bg * nose_union_mask + face_no_bg * (1 - nose_union_mask)

    non_face_mask = mixed.mean(axis=-1) == 0
    # 修改非人脸区域的颜色为目标图片鼻子平均颜色
    face_image_mean = face_image[face_mask].mean(axis=0)
    face_no_bg[face_seg_mask == 0] = face_image_mean
    nose_no_bg[nose_seg_mask == 0] = face_image_mean
    # 进行图像融合

    nose_union_mask = (nose_mask_dilate | face_mask).astype("float32")[..., None]

    nose_no_bg = align_color(
        nose_no_bg, face_no_bg, face_mask, nose_union_mask[..., 0]
    ).numpy()

    nose_union_mask = adjust_nose_union_mask(
        face_no_bg,
        nose_no_bg,
        face_mask,
        nose_mask,
        nose_union_mask,
        face_image_mean,
        threshold=40,
    )

    nose_mean_bg = np.ones_like(nose_image, dtype=np.float32) * face_image_mean
    nose_no_bg = nose_no_bg * nose_union_mask + nose_mean_bg * (1 - nose_union_mask)

    center = get_center(face_mask)

    # face_no_bg=blur_nose(face_no_bg.astype("uint8"),face_mask,nose_mask)
    face_mask_erode = cv2.erode(
        face_mask.astype("uint8"), np.ones((blended_mask_erode, blended_mask_erode))
    )

    # face_no_bg[(nose_mask | face_mask)]=face_image_mean
    if visualization:
        plt.figure(dpi=400)
        plt.subplot(2, 3, 1)
        plt.imshow(nose_union_mask)
        plt.title("nose_union_mask")
        plt.axis("off")

    mixed2 = cv2.seamlessClone(
        nose_no_bg.astype("uint8"),
        face_no_bg.astype("uint8"),
        nose_union_mask.astype("uint8") * 255,
        center,
        cv2.NORMAL_CLONE,
    )
    if inpaint is True:
        background = face_image.copy()
        mask = non_face_mask == 0
        points = np.stack(
            np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1])), axis=-1
        )[(nose_union_mask[..., 0] != 0)]
        bbox = np.concatenate([points.min(axis=0), points.max(axis=0)], axis=-1)
        mask_crop = face_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype("uint8")
        mask_crop = cv2.dilate(mask_crop, np.ones((10, 10), dtype="float32"))
        nose_crop = face_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        inpainted = cv2.inpaint(nose_crop, mask_crop, 5, cv2.INPAINT_NS)
        background = face_image.copy()
        background[bbox[1] : bbox[3], bbox[0] : bbox[2]] = inpainted
        mixed2 = mixed2 * mask[..., None] + background * (1 - mask[..., None])
    if visualization:
        plt.subplot(2, 3, 2)
        plt.imshow(nose_no_bg / 255)
        plt.title("nose_no_bg")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(face_no_bg / 255)
        plt.title("face_no_bg")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(mixed / 255)
        plt.title("mixed")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(mixed2 / 255)
        plt.title("mixed2")
        plt.axis("off")
    return mixed2


@deprecated
def alpha_blender_v3(
    face_image,
    nose_image,
    face_mask,
    nose_mask,
    face_seg_mask,
    nose_seg_mask,
    inpaint=True,
    blended_mask_erode=10,
    nose_mask_dilate=20,
    nose_mask_blur=51,
    overbright_threshold=230,
    visualization=False,
):
    """

    Brief:图像融合算法，传入的图片已经计算好鼻子掩码，并且经过了图像对其，这里是最后一步，也就是真正的替换鼻子算法的具体逻辑
    整个过程细节比较繁琐，详细看代码中的注释,该API采用了Alpha融合的算法思路
    Args:
        face_image:人脸图片，被替换鼻子的图片，RGB/uint8格式
        nose_image:鼻子图片，提供鼻子的图片，RGB/uint8格式
        face_mask:人脸图片的鼻子位置掩码（0代表非鼻子区域，1代表鼻子区域，其他掩码也是如此）
        nose_mask:鼻子图片的鼻子位置掩码
        face_seg_mask:人脸图片的人脸位置掩码
        nose_seg_mask:鼻子图片的鼻子位置掩码
        inpaint:bool，是否补充背景，这一步需要较长时间计算
        nose_mask_dilate:对鼻子掩码进行膨胀操作的宽度，可能掩码覆盖的区域不够，所以需要进行一次膨胀
        blended_mask_blur:对最终计算出来的替换鼻子的位置掩码，进行高斯模糊的核大小
        overbright_threshold:对于过度曝光区域进行修正的阈值（暂时此参数无效）
        visualization:是否可视化中间结果，主要是方便代码调试用
    Returns:
        被替换鼻子之后的人脸图，RGB/uint8格式
    """
    logger.debug(
        "call alpha_blender_v3(...,inpaint=%s,visualization=%s)"
        % (inpaint, visualization)
    )
    face_image_mean = face_image[face_mask.astype("bool")].mean(axis=0)
    # 可能本来的脸图中就有纯黑色，将其排除在外
    face_image[face_image.mean(axis=-1) == 0] = 1
    face_no_bg = face_image * face_seg_mask[..., None]
    nose_no_bg = nose_image * nose_seg_mask[..., None]
    # 求出替换鼻子的区域

    nose_mask_dilate = cv2.dilate(
        nose_mask, np.ones((nose_mask_dilate, nose_mask_dilate))
    )

    face_mask = face_mask.astype("bool")
    nose_mask = nose_mask.astype("bool")
    nose_mask_dilate = nose_mask_dilate.astype("bool")

    nose_union_mask = (nose_mask | face_mask).astype("float32")[..., None]
    nose_no_bg = align_color_avg_pool(
        nose_no_bg,
        face_no_bg,
        face_mask,
        nose_mask_dilate,
        threshold=overbright_threshold,
    )
    # 第一次替换，目的是为了确定替换鼻子后的人脸精细掩码
    mixed = nose_no_bg * nose_union_mask + face_no_bg * (1 - nose_union_mask)

    non_face_mask = mixed.mean(axis=-1) == 0
    # 修改非人脸区域的颜色为目标图片鼻子平均颜色
    face_no_bg[face_seg_mask == 0] = face_image_mean
    nose_no_bg[nose_seg_mask == 0] = face_image_mean
    # 进行图像融合

    nose_union_mask = (nose_mask_dilate | face_mask).astype("float32")[..., None]
    nose_union_mask = adjust_nose_union_mask(
        face_no_bg, nose_no_bg, face_mask, nose_mask, nose_union_mask, face_image_mean
    )

    if visualization is True:
        plt.figure(dpi=200)
        plt.subplot(2, 3, 6)
        plt.axis("off")
        plt.title("adjusted_blend_mask")
        plt.imshow(nose_union_mask)

    nose_union_mask = cv2.erode(
        nose_union_mask,
        np.ones((blended_mask_erode, blended_mask_erode), dtype="float32"),
    )[..., None]

    nose_union_mask = cv2.GaussianBlur(
        nose_union_mask, (nose_mask_blur, nose_mask_blur), 0
    )[..., None]

    nose_no_bg = nose_no_bg * nose_union_mask + face_image_mean * (1 - nose_union_mask)

    mixed2 = nose_no_bg * nose_union_mask + face_no_bg * (1 - nose_union_mask)
    mixed2[non_face_mask] = 0

    if visualization is True:
        plt.subplot(2, 3, 1)
        plt.axis("off")
        plt.title("face")
        plt.imshow(face_no_bg / 255)

        plt.subplot(2, 3, 2)
        plt.axis("off")
        plt.title("nose")
        plt.imshow(nose_no_bg / 255)

        plt.subplot(2, 3, 3)
        plt.axis("off")
        plt.title("blended_mask")
        plt.imshow(nose_union_mask / 255)

        plt.subplot(2, 3, 4)
        plt.axis("off")
        plt.title("blend_1")
        plt.imshow(mixed / 255)
    if inpaint is True:
        background = face_image.copy()
        mask = mixed2.mean(axis=-1) != 0
        points = np.stack(
            np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1])), axis=-1
        )[(nose_union_mask[..., 0] != 0)]
        bbox = np.concatenate([points.min(axis=0), points.max(axis=0)], axis=-1)
        mask_crop = face_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype("uint8")
        mask_crop = cv2.dilate(mask_crop, np.ones((10, 10), dtype="float32"))
        nose_crop = face_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        inpainted = cv2.inpaint(nose_crop, mask_crop, 5, cv2.INPAINT_NS)
        background = face_image.copy()
        background[bbox[1] : bbox[3], bbox[0] : bbox[2]] = inpainted
        mixed2 = mixed2 * mask[..., None] + background * (1 - mask[..., None])
    if visualization is True:
        plt.subplot(2, 3, 5)
        plt.axis("off")
        plt.title("blend_2")
        plt.imshow(mixed2 / 255)
        plt.show()
    return mixed2.astype("uint8")


def alpha_blender_v4(
    face_image,
    nose_image,
    face_mask,
    nose_mask,
    face_seg_mask,
    nose_seg_mask,
    inpaint,
    nose_mask_dilate,
    blended_mask_erode,
    blended_mask_blur,
    overbright_threshold,
    aligned_color_ksize,
    mean_method,
    visualization,
):
    """

    Brief:图像融合算法，传入的图片已经计算好鼻子掩码，并且经过了图像对其，这里是最后一步，也就是真正的替换鼻子算法的具体逻辑
    整个过程细节比较繁琐，详细看代码中的注释,该API采用了Alpha融合的算法思路
    Args:
        face_image:人脸图片，被替换鼻子的图片，RGB/uint8格式
        nose_image:鼻子图片，提供鼻子的图片，RGB/uint8格式
        face_mask:人脸图片的鼻子位置掩码（0代表非鼻子区域，1代表鼻子区域，其他掩码也是如此）
        nose_mask:鼻子图片的鼻子位置掩码
        face_seg_mask:人脸图片的人脸位置掩码
        nose_seg_mask:鼻子图片的鼻子位置掩码
        inpaint:bool，是否补充背景，这一步需要较长时间计算
        nose_mask_dilate:对鼻子掩码进行膨胀操作的宽度，可能掩码覆盖的区域不够，所以需要进行一次膨胀
        blended_mask_erode:对最终计算出来的替换鼻子的位置掩码，进行腐蚀的宽度
        blended_mask_blur:对最终计算出来的替换鼻子的位置掩码，进行高斯模糊的核大小
        overbright_threshold:对于过度曝光区域进行修正的阈值（暂时此参数无效）
        aligned_color_ksize:进行颜色对其的时候，如果选用了'local',需要设置平均池化的池化和大小
        mean_method:计算均值的方式，'global'为计算整个鼻子区域的均值，'local'采用平均值话，以滑动窗口的形式进行计算
        visualization:是否可视化中间结果，主要是方便代码调试用
    Returns:
        被替换鼻子之后的人脸图，RGB/uint8格式
    """
    logger.debug("call alpha_blender_v4")
    # 可能本来的脸图中就有纯黑色，将其排除在外
    face_image[face_image.mean(axis=-1) == 0] = 1
    face_no_bg = face_image * face_seg_mask[..., None]
    nose_no_bg = nose_image * nose_seg_mask[..., None]
    # 求出替换鼻子的区域
    nose_mask_dilate = cv2.dilate(
        nose_mask.astype("uint8"),
        np.ones(shape=(nose_mask_dilate, nose_mask_dilate), dtype="float32"),
    )
    face_mask = face_mask.astype("bool")
    nose_mask = nose_mask.astype("bool")
    nose_mask_dilate = nose_mask_dilate.astype("bool")
    nose_union_mask = (nose_mask | face_mask).astype("float32")[..., None]
    # 第一次替换，目的是为了确定替换鼻子后的人脸精细掩码
    mixed = nose_no_bg * nose_union_mask + face_no_bg * (1 - nose_union_mask)
    non_face_mask = mixed.mean(axis=-1) == 0
    # 修改非人脸区域的颜色为目标图片鼻子平均颜色
    face_image_mean = face_image[face_mask].mean(axis=0)
    face_no_bg[face_seg_mask == 0] = face_image_mean
    nose_no_bg[nose_seg_mask == 0] = face_image_mean
    # 进行图像融合
    nose_union_mask = (nose_mask_dilate | face_mask).astype("float32")[..., None]
    # 修改鼻子颜色
    if mean_method == "local":
        logger.debug("call aligned_color_avg_pool_v2")
        nose_no_bg = align_color_avg_pool_v2(
            nose_no_bg, face_no_bg, nose_union_mask[..., 0], ksize=aligned_color_ksize
        ).numpy()
    elif mean_method == "global":
        logger.debug("call aligned_color")
        nose_no_bg = align_color(
            nose_no_bg, face_no_bg, nose_union_mask[..., 0], face_mask
        ).numpy()
    else:
        raise ValueError("mo such mean method:%s" % (mean_method))
    # 对掩码进行调整，主要是怕把眼睛等其他非必要部分包括进来
    nose_union_mask = adjust_nose_union_mask(
        face_no_bg,
        nose_no_bg,
        face_mask,
        nose_mask,
        nose_union_mask,
        face_image_mean,
        threshold=50,
    )
    # 防止把除了脸以外的背景包括进来
    nose_union_mask = nose_union_mask * nose_seg_mask[..., None]
    nose_union_mask_blur = cv2.GaussianBlur(nose_union_mask, [51, 51], 0)[..., None]
    # 删掉除了鼻子以外的其他区域，用均值进行填充
    nose_no_bg = nose_no_bg * nose_union_mask_blur + face_image_mean * (
        1 - nose_union_mask_blur
    )
    nose_union_mask_erode = cv2.erode(
        nose_union_mask.astype("uint8"),
        np.ones((blended_mask_erode, blended_mask_erode), dtype="float32"),
    ).astype("float32")
    nose_union_mask_blur = cv2.GaussianBlur(
        nose_union_mask_erode, [blended_mask_blur, blended_mask_blur], 0
    )[..., None]
    # 删除被替换鼻子的脸的原本的鼻子
    face_no_bg = (
        face_no_bg * (1 - nose_union_mask_blur) + face_image_mean * nose_union_mask_blur
    )
    # 对图像进行融合
    mixed2 = nose_no_bg * nose_union_mask_blur + face_no_bg * (1 - nose_union_mask_blur)
    # 可视化一些中间结果，主要是方便调试用
    if visualization:
        plt.figure(dpi=400)
        plt.subplot(2, 3, 1)
        plt.imshow(nose_union_mask_blur)
        plt.title("nose_union_mask_blur")
        plt.axis("off")
    # 补上原本的背景
    if inpaint is True:
        background = face_image.copy()
        mask = non_face_mask == 0
        points = np.stack(
            np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1])), axis=-1
        )[(nose_union_mask[..., 0] != 0)]
        bbox = np.concatenate([points.min(axis=0), points.max(axis=0)], axis=-1)
        mask_crop = face_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype("uint8")
        mask_crop = cv2.dilate(mask_crop, np.ones((10, 10), dtype="float32"))
        nose_crop = face_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        inpainted = cv2.inpaint(nose_crop, mask_crop, 5, cv2.INPAINT_NS)
        background = face_image.copy()
        background[bbox[1] : bbox[3], bbox[0] : bbox[2]] = inpainted
        mixed2 = mixed2 * mask[..., None] + background * (1 - mask[..., None])
    if visualization:
        plt.subplot(2, 3, 2)
        plt.imshow(nose_no_bg / 255)
        plt.title("nose_no_bg")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(face_no_bg / 255)
        plt.title("face_no_bg")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(mixed / 255)
        plt.title("mixed")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(mixed2 / 255)
        plt.title("mixed2")
        plt.axis("off")
    return mixed2.astype("uint8")


@timer
def alpha_blender_v5(
    face_image,
    nose_image,
    face_mask,
    nose_mask,
    face_seg_mask,
    nose_seg_mask,
    inpaint,
    nose_mask_dilate,
    blended_mask_erode,
    blended_mask_blur,
    clean_threshold,
    aligned_color_ksize,
    mean_method,
    visualization,
):
    """

    Brief:图像融合算法，传入的图片已经计算好鼻子掩码，并且经过了图像对其，这里是最后一步，也就是真正的替换鼻子算法的具体逻辑
    整个过程细节比较繁琐，详细看代码中的注释,该API采用了Alpha融合的算法思路
    Args:
        face_image:人脸图片，被替换鼻子的图片，RGB/uint8格式
        nose_image:鼻子图片，提供鼻子的图片，RGB/uint8格式
        face_mask:人脸图片的鼻子位置掩码（0代表非鼻子区域，1代表鼻子区域，其他掩码也是如此）
        nose_mask:鼻子图片的鼻子位置掩码
        face_seg_mask:人脸图片的人脸位置掩码
        nose_seg_mask:鼻子图片的鼻子位置掩码
        inpaint:bool，是否补充背景，这一步需要较长时间计算
        nose_mask_dilate:对鼻子掩码进行膨胀操作的宽度，可能掩码覆盖的区域不够，所以需要进行一次膨胀
        blended_mask_erode:对最终计算出来的替换鼻子的位置掩码，进行腐蚀的宽度
        blended_mask_blur:对最终计算出来的替换鼻子的位置掩码，进行高斯模糊的核大小
        clean_threshold:防止其他非鼻子区域，特别是眼睛这种颜色差异很大的被包括进来，调整该阈值可以帮助过滤掉这部分信息
        aligned_color_ksize:进行颜色对其的时候，如果选用了'local',需要设置平均池化的池化和大小
        mean_method:计算均值的方式，'global'为计算整个鼻子区域的均值，'local'采用平均值话，以滑动窗口的形式进行计算
        visualization:是否可视化中间结果，主要是方便代码调试用
    Returns:
        被替换鼻子之后的人脸图，RGB/uint8格式
    """
    logger.debug("call alpha_blender_v5")
    # 可能本来的脸图中就有纯黑色，将其排除在外
    face_image[face_image.mean(axis=-1) == 0] = 1
    face_no_bg = face_image * face_seg_mask[..., None]
    nose_no_bg = nose_image * nose_seg_mask[..., None]
    nose_mask_dilate = cv2.dilate(
        nose_mask,
        np.ones(shape=(nose_mask_dilate, nose_mask_dilate), dtype="float32"),
    )
    face_mask = face_mask.astype("bool")
    nose_mask = nose_mask.astype("bool")
    nose_mask_dilate = nose_mask_dilate.astype("bool")
    face_seg_mask = face_seg_mask.astype("bool")
    nose_seg_mask = nose_seg_mask.astype("bool")
    # 求出替换鼻子的区域
    nose_union_mask = (nose_mask | face_mask).astype("float32")[..., None]

    # 第一次替换，目的是为了确定替换鼻子后的人脸精细掩码
    mixed = nose_no_bg * nose_union_mask + face_no_bg * (1 - nose_union_mask)
    non_face_mask = mixed.mean(axis=-1) == 0

    # 修改鼻子颜色，使其和被替换的鼻子颜色靠近，这一步需要的掩码包含两个鼻子交集，并且排除非人脸区域
    nose_union_mask = ((nose_mask_dilate | face_mask) & nose_seg_mask).astype("float32")
    if mean_method == "local":
        logger.debug("call aligned_color_avg_pool_v2")
        nose_no_bg = align_color_avg_pool_v2(
            nose_no_bg, face_no_bg, nose_union_mask, ksize=aligned_color_ksize
        ).numpy()
    elif mean_method == "global":
        logger.debug("call aligned_color")
        nose_no_bg = align_color(
            nose_no_bg, face_no_bg, nose_union_mask, face_mask
        ).numpy()
    else:
        raise ValueError("mo such mean method:%s" % (mean_method))
    # 修改非人脸区域的颜色为目标图片鼻子平均颜色
    face_image_mean = face_image[face_mask].mean(axis=0)
    face_no_bg[face_seg_mask == 0] = face_image_mean
    nose_no_bg[nose_seg_mask == 0] = face_image_mean
    # 计算最终融合需要的掩码，对掩码进行调整，排除眼睛等其他非必要部分
    nose_union_mask = (nose_mask_dilate | face_mask).astype("float32")[..., None]
    nose_union_mask = adjust_nose_union_mask(
        face_no_bg,
        nose_no_bg,
        face_mask,
        nose_mask,
        nose_union_mask,
        face_image_mean,
        threshold=clean_threshold,
        erode_and_dilate=True,
    )
    # 防止把除了脸以外的背景包括进来
    # nose_union_mask = nose_union_mask * nose_seg_mask[..., None]
    # 然后进行一次羽化，先把鼻子图删除非鼻子区域，并且模糊化边缘
    nose_union_mask_blur = cv2.GaussianBlur(nose_union_mask, [51, 51], 0)[..., None]
    # 删掉除了鼻子以外的其他区域，用均值进行填充
    nose_no_bg = nose_no_bg * nose_union_mask_blur + face_image_mean * (
        1 - nose_union_mask_blur
    )
    # 然后准备图像融合的区域掩码，同样需要模糊化边缘
    nose_union_mask_erode = cv2.erode(
        nose_union_mask.astype("uint8"),
        np.ones((blended_mask_erode, blended_mask_erode), dtype="float32"),
    ).astype("float32")
    # 防止把除了脸以外的背景包括进来
    nose_union_mask_erode = nose_union_mask_erode * nose_seg_mask
    nose_union_mask_blur = cv2.GaussianBlur(
        nose_union_mask_erode, [blended_mask_blur, blended_mask_blur], 0
    )[..., None]
    # 对图像进行融合
    mixed2 = nose_no_bg * nose_union_mask_blur + face_no_bg * (1 - nose_union_mask_blur)
    # 可视化一些中间结果，主要是方便调试用
    if visualization:
        plt.figure(dpi=400)
        plt.subplot(2, 3, 1)
        plt.imshow(nose_union_mask_blur)
        plt.title("nose_union_mask_blur")
        plt.axis("off")
    # 补上原本的背景，因为不同鼻子大小不同，大鼻子换成小鼻子后，就有部分空缺需要想办法填充
    if inpaint is True:
        background = face_image.copy()
        mask = non_face_mask == 0
        points = np.stack(
            np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1])), axis=-1
        )[(nose_union_mask[..., 0] != 0)]
        bbox = np.concatenate([points.min(axis=0), points.max(axis=0)], axis=-1)
        mask_crop = face_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype("uint8")
        mask_crop = cv2.dilate(mask_crop, np.ones((10, 10), dtype="float32"))
        nose_crop = face_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        inpainted = cv2.inpaint(nose_crop, mask_crop, 5, cv2.INPAINT_NS)
        background = face_image.copy()
        background[bbox[1] : bbox[3], bbox[0] : bbox[2]] = inpainted
        mixed2 = mixed2 * mask[..., None] + background * (1 - mask[..., None])
    if visualization:
        plt.subplot(2, 3, 2)
        plt.imshow(nose_no_bg * nose_union_mask_blur / 255)
        plt.title("nose_no_bg")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(face_no_bg / 255)
        plt.title("face_no_bg")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(mixed / 255)
        plt.title("mixed")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(mixed2 / 255)
        plt.title("mixed2")
        plt.axis("off")
        plt.subplot(2, 3, 6)
        plt.imshow(non_face_mask / 255)
        plt.title("non_face_mask")
        plt.axis("off")
    return mixed2.astype("uint8")


@timer
def alpha_blender_v6(
    face_image,
    nose_image,
    face_mask,
    nose_mask,
    face_seg_mask,
    nose_seg_mask,
    face_keypoint,
    nose_keypoint,
    inpaint,
    nose_mask_dilate,
    blended_mask_erode,
    blended_mask_blur,
    clean_threshold,
    aligned_color_ksize,
    mean_method,
    visualization,
    remove_eyes=True,
    fill_mode="region_mean",
    source_fill_blur=51,
    mask_pool=20,
):
    """
    Author:shijunfeng
    Brief:图像融合算法，传入的图片已经计算好鼻子掩码，并且经过了图像对其，这里是最后一步，也就是真正的替换鼻子算法的具体逻辑
    整个过程细节比较繁琐，详细看代码中的注释,该API采用了Alpha融合的算法思路
    本函数已泛化为「任意部位」融合，部位相关的差异通过下面的新增参数控制，
    所有新增参数取默认值时，行为与原本的换鼻逻辑完全一致。
    Args:
        face_image:人脸图片，被替换鼻子的图片，RGB/uint8格式
        nose_image:鼻子图片，提供鼻子的图片，RGB/uint8格式
        face_mask:人脸图片的鼻子位置掩码（0代表非鼻子区域，1代表鼻子区域，其他掩码也是如此）
        nose_mask:鼻子图片的鼻子位置掩码
        face_seg_mask:人脸图片的人脸位置掩码
        nose_seg_mask:鼻子图片的鼻子位置掩码
        face_keypoint:人脸图片边缘采样点
        nose_keypoint:鼻子图片边缘采样点
        inpaint:bool，是否补充背景，这一步需要较长时间计算
        nose_mask_dilate:对鼻子掩码进行膨胀操作的宽度，可能掩码覆盖的区域不够，所以需要进行一次膨胀
        blended_mask_erode:对最终计算出来的替换鼻子的位置掩码，进行腐蚀的宽度
        blended_mask_blur:对最终计算出来的替换鼻子的位置掩码，进行高斯模糊的核大小
        clean_threshold:防止其他非鼻子区域，特别是眼睛这种颜色差异很大的被包括进来，调整该阈值可以帮助过滤掉这部分信息
        aligned_color_ksize:进行颜色对其的时候，如果选用了'local',需要设置平均池化的池化和大小
        mean_method:计算均值的方式，'global'为计算整个鼻子区域的均值，'local'采用平均值话，以滑动窗口的形式进行计算
        visualization:是否可视化中间结果，主要是方便代码调试用
        remove_eyes:是否执行「去眼睛区域」。该步骤原本用于避免换鼻时把供体的眼睛带进来，
                    换眼时必须关闭，否则会把要替换的眼睛本身当成离群值剔除掉
        fill_mode:掩码以外区域的填充色取值方式，
                  'region_mean'使用目标图部位区域的均值（换鼻的原始行为），
                  'skin_mean'使用目标图人脸内非部位区域的肤色均值，
                  眼睛等颜色与肤色差异大的部位必须用后者，否则会出现暗环
        source_fill_blur:填充供体图掩码以外区域时使用的羽化核大小，部位越小取值越小
        mask_pool:颜色对齐时对掩码做最大值池化的核大小
    Returns:
        被替换鼻子之后的人脸图，RGB/uint8格式
    """
    logger.debug("call alpha_blender_v6")
    # 可能本来的脸图中就有纯黑色，将其排除在外
    face_image[face_image.mean(axis=-1) == 0] = 1
    face_no_bg = face_image * face_seg_mask[..., None]
    nose_no_bg = nose_image * nose_seg_mask[..., None]
    nose_mask_dilate = cv2.dilate(
        nose_mask,
        np.ones(shape=(nose_mask_dilate, nose_mask_dilate), dtype="float32"),
    )
    face_mask = face_mask.astype("bool")
    nose_mask = nose_mask.astype("bool")
    nose_mask_dilate = nose_mask_dilate.astype("bool")
    face_seg_mask = face_seg_mask.astype("bool")
    nose_seg_mask = nose_seg_mask.astype("bool")
    # 求出替换鼻子的区域
    nose_union_mask = (nose_mask | face_mask).astype("float32")[..., None]

    # 第一次替换，目的是为了确定替换鼻子后的人脸精细掩码
    mixed = nose_no_bg * nose_union_mask + face_no_bg * (1 - nose_union_mask)
    non_face_mask = mixed.mean(axis=-1) == 0

    # 修改鼻子颜色，使其和被替换的鼻子颜色靠近，这一步需要的掩码包含两个鼻子交集，并且排除非人脸区域
    nose_union_mask = ((nose_mask_dilate | face_mask) & nose_seg_mask).astype("float32")
    if mean_method == "local":
        logger.debug("call aligned_color_avg_pool_v2")
        nose_no_bg = align_color_avg_pool_v2(
            nose_no_bg,
            face_no_bg,
            nose_union_mask,
            ksize=aligned_color_ksize,
            pool=mask_pool,
        ).numpy()
    elif mean_method == "global":
        logger.debug("call aligned_color")
        nose_no_bg = align_color(
            nose_no_bg, face_no_bg, nose_union_mask, face_mask
        ).numpy()
    else:
        raise ValueError("mo such mean method:%s" % (mean_method))
    # 修改非人脸区域的颜色为目标图片部位平均颜色
    face_image_mean = face_image[face_mask].mean(axis=0)
    # 掩码以外区域的填充色：换鼻沿用部位均值，眼睛等部位改用肤色均值避免出现暗环
    fill_mean = face_image_mean
    if fill_mode == "skin_mean":
        skin = (face_seg_mask & (~face_mask))
        if skin.any():
            fill_mean = face_image[skin].mean(axis=0)
    face_no_bg[face_seg_mask == 0] = fill_mean
    nose_no_bg[nose_seg_mask == 0] = fill_mean
    # 计算最终融合需要的掩码，对掩码进行调整，排除眼睛等其他非必要部分
    nose_union_mask = (nose_mask_dilate | face_mask).astype("float32")
    # 消除眼睛区域（换鼻时需要；换眼时必须关闭，否则会把眼睛本身剔除）
    if remove_eyes:
        nose_union_mask = remove_eyes_region(
            nose_image=nose_no_bg,
            nose_mask=nose_mask.astype("float32"),
            blended_mask=nose_union_mask,
            nose_keypoint=nose_keypoint,
            mean=face_image_mean,
            threshold=clean_threshold,
        )[..., None]

    # 防止把除了脸以外的背景包括进来
    # nose_union_mask = nose_union_mask * nose_seg_mask[..., None]

    # 然后进行一次羽化，先把鼻子图删除非鼻子区域，并且模糊化边缘
    nose_union_mask_blur = cv2.GaussianBlur(
        nose_union_mask, [source_fill_blur, source_fill_blur], 0
    )[..., None]
    # 删掉除了鼻子以外的其他区域，用均值进行填充
    nose_no_bg = nose_no_bg * nose_union_mask_blur + fill_mean * (
        1 - nose_union_mask_blur
    )

    # 然后准备图像融合的区域掩码，同样需要模糊化边缘
    nose_union_mask_erode = cv2.erode(
        nose_union_mask.astype("uint8"),
        np.ones((blended_mask_erode, blended_mask_erode), dtype="float32"),
    ).astype("float32")

    # 防止把除了脸以外的背景包括进来
    sub = np.clip(
        nose_union_mask_erode - nose_union_mask_erode * nose_seg_mask, 0, 1
    ).astype("uint8")
    nose_no_bg = cv2.inpaint(nose_no_bg.astype("uint8"), sub, 40, cv2.INPAINT_NS)
    nose_union_mask_erode = nose_union_mask_erode * nose_seg_mask
    nose_union_mask_blur = cv2.GaussianBlur(
        nose_union_mask_erode[..., None], [blended_mask_blur, blended_mask_blur], 0
    )[..., None]

    mixed2 = nose_no_bg * nose_union_mask_blur + face_no_bg * (1 - nose_union_mask_blur)

    # 可视化一些中间结果，主要是方便调试用
    if visualization:
        plt.figure(dpi=400)
        plt.subplot(2, 3, 1)
        plt.imshow(nose_union_mask_blur, cmap="binary")
        plt.title("nose_union_mask_blur")
        plt.axis("off")
    # 补上原本的背景，因为不同鼻子大小不同，大鼻子换成小鼻子后，就有部分空缺需要想办法填充
    if inpaint is True:
        background = face_image.copy()
        points = np.stack(
            np.meshgrid(
                np.arange(nose_union_mask.shape[0]), np.arange(nose_union_mask.shape[1])
            ),
            axis=-1,
        )[((nose_union_mask_blur > 0.1)[..., 0] != 0)]
        bbox = np.concatenate([points.min(axis=0), points.max(axis=0)], axis=-1)
        mask = non_face_mask == 0
        nose_crop = face_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        mask_crop = face_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype("uint8")
        mask_crop = cv2.dilate(mask_crop, np.ones((10, 10), dtype="float32"))

        # mask_crop=mask_crop*(nose_crop.mean(axis=-1)==0)

        inpainted = cv2.inpaint(nose_crop, mask_crop, 20, cv2.INPAINT_NS)
        background[bbox[1] : bbox[3], bbox[0] : bbox[2]] = inpainted
        mixed2 = mixed2 * mask[..., None] + background * (1 - mask[..., None])

    if visualization:
        plt.subplot(2, 3, 2)
        plt.imshow(nose_no_bg * nose_union_mask_blur / 255)
        plt.title("nose_no_bg")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(face_no_bg / 255)
        plt.title("face_no_bg")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(mixed / 255, cmap="binary")
        # plt.scatter(nose_keypoint[::,0],nose_keypoint[::,1],alpha=0.5,s=0.1)
        plt.title("mixed")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(mixed2 / 255, cmap="binary")
        plt.title("mixed2")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(~non_face_mask, cmap="binary")
        plt.title("non_face_mask")
        plt.axis("off")

    return mixed2.astype("uint8")
