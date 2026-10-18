import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from utils import logger
from utils import timer
import numpy as np
import cv2

try:
    from IPython import display
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class LossFunction(object):
    """
    用于求解仿射变换矩阵的的损失函数
    首先每个鼻子都通过掩码求解得到若干的点，根据极坐标的角度拍进行排序，将其抽象为多边形以及对应的顶点，那么问题就是如何让多边形1通过仿射变换和多边形2对齐（尽可能重合）
    1.多边形每个点去匹配目标多边形中离他最近的点，多对一映射，计算出所有点对距离平方和
    2.计算两个多边形直径，计算他们的差值的归一化均方误差
    3.加起来作为总的损失函数
    """

    def __init__(self, target_keypoints, input_keypoints):
        """

        Brief:
            构造函数，构造一个LossFunction对象
        Args:
            target_keypoints:目标关键点,模型的标签
            input_keypoints:输入的关键点,模型的输入
        Returns:
        """
        pt1, pt2 = self.get_farthest_point_pair(
            target_keypoints
        )  # 计算散点图中最远的两个点，这两个点的坐标不随仿射变换而变化，是常量，因此可以事先计算出来

        pt3, pt4 = self.get_farthest_point_pair(input_keypoints)
        self.target_id1 = tf.stop_gradient(pt1)
        self.target_id2 = tf.stop_gradient(pt2)  # 作为一个常量，他的梯度也是不需要计算的
        self.output_id1 = tf.stop_gradient(pt3)
        self.output_id2 = tf.stop_gradient(pt4)

    def get_farthest_point_pair(self, points):
        """
        Brief:
            计算得到最远点对（图的直径两端）的下标，双重for循环计算平面最远点对的GPU并行计算版
        Args:
            points:关键点
        Returns:
            最远点对分别的下标
        """
        center = tf.reduce_mean(points, axis=0)
        delta = points - center
        theta = tf.math.atan2(delta[::, 1], delta[::, 0])
        idx = tf.argsort(theta)
        points = tf.gather(points, idx)
        theta = tf.gather(theta, idx)
        theta += np.pi
        theta2 = (theta.numpy() + np.pi) % (2 * np.pi)
        delta = tf.abs(theta[..., None] - theta2[None, ...])
        matched = tf.argmin(delta)
        diameters = tf.reduce_sum(
            (points - tf.gather(points, matched)) ** 2, axis=-1
        )  # 所有直径长度
        end1 = tf.argmax(diameters)  # 找出最长的直径的端点
        end2 = matched[end1]  # 找出另外一个端点
        return idx[end1], idx[end2]

    def __call__(self, target_keypoints, output_keypoints):
        """
        Brief:损失函数的具体逻辑计算过程
        Args:
            target_keypoints:目标多边形的顶点（标签值）
            output_keypoints:模型输出的多边形顶点（预测值）
        Returns:
        """
        kp1 = output_keypoints[::, None, :2]
        kp2 = target_keypoints[None, ::, :2]
        distance = tf.reduce_mean((kp1 - kp2) ** 2, axis=-1)
        # kp1到kp2中的点最近的，作为匹配结果，然后让他们之间的距离尽可能小
        matched = tf.one_hot(
            tf.argmin(distance, axis=-1), tf.shape(target_keypoints)[0]
        )
        distance = tf.reduce_sum(matched * distance, axis=-1)
        distance = tf.reduce_mean(distance)

        pt1 = target_keypoints[self.target_id1]
        pt2 = target_keypoints[self.target_id2]
        pt3 = output_keypoints[self.output_id1]
        pt4 = output_keypoints[self.output_id2]

        d1 = tf.reduce_sum((pt1 - pt2) ** 2)
        d2 = tf.reduce_sum((pt3 - pt4) ** 2)
        dist = tf.maximum(5.0, tf.abs(d1 - d2))  # 小于2就不管了,这里只是为了控制scale的训练,不需要真的完全拟合
        distance2 = tf.square(dist) / d1
        return distance + distance2


class AffineLayer(tf.keras.layers.Layer):
    """
    图像和点集的仿射变换层，对图或者点进行平移，缩放，旋转变换
    参考tensorflow.keras.layers.Layer的文档
    """

    def __init__(self, image_shape, **kwargs):
        super(AffineLayer, self).__init__(**kwargs)
        self.image_shape = image_shape

    def build(self, input_shape):
        self.dx = self.add_weight(
            initializer=tf.keras.initializers.Constant(0.0),
            name="dx",
            shape=(),
            dtype=tf.float32,
            trainable=True,
        )
        self.dy = self.add_weight(
            initializer=tf.keras.initializers.Constant(0.0),
            name="dy",
            shape=(),
            dtype=tf.float32,
            trainable=True,
        )
        self.theta = self.add_weight(
            initializer=tf.keras.initializers.Constant(0 * np.pi / 180),
            name="theta",
            shape=(),
            dtype=tf.float32,
            trainable=True,
        )

        self.scale = self.add_weight(
            initializer=tf.keras.initializers.Constant(1.0),
            name="scale",
            shape=(),
            dtype=tf.float32,
            trainable=True,
        )
        self.cx = self.add_weight(
            initializer=tf.keras.initializers.Constant(0),
            name="cx",
            shape=(),
            dtype=tf.float32,
            trainable=False,
        )
        self.cy = self.add_weight(
            initializer=tf.keras.initializers.Constant(0),
            name="cy",
            shape=(),
            dtype=tf.float32,
            trainable=False,
        )

    def apply_shift(self, keypoints):
        x = keypoints[::, 0] + keypoints[::, 2] * self.dx * self.image_shape[0]
        y = keypoints[::, 1] + keypoints[::, 2] * self.dy * self.image_shape[1]
        z = keypoints[::, 2]
        return tf.stack([x, y, z], axis=-1)

    def apply_rotate(self, keypoints):
        x = keypoints[::, 0]
        y = keypoints[::, 1]
        z = keypoints[::, 2]
        sin_v = tf.math.sin(self.theta)
        cos_v = tf.math.cos(self.theta)
        cx = self.cx
        cy = self.cy

        new_x = x * cos_v - y * sin_v + z * (-cx * cos_v + cy * sin_v + cx)
        new_y = x * sin_v + y * cos_v + z * (-cx * sin_v - cy * cos_v + cy)
        new_z = z
        return tf.stack([new_x, new_y, new_z], axis=-1)

    def apply_scale(self, keypoints):
        x = keypoints[::, 0]
        y = keypoints[::, 1]
        z = keypoints[::, 2]
        scale_x = self.scale
        scale_y = self.scale
        cx = self.cx
        cy = self.cy
        new_x = x * scale_x + z * cx * (1 - scale_x)
        new_y = y * scale_y + cy * (1 - scale_y)
        new_z = z
        return tf.stack([new_x, new_y, new_z], axis=-1)

    def call(self, keypoints):
        means = tf.reduce_mean(keypoints, axis=0)
        tf.compat.v1.assign(self.cx, means[0])
        tf.compat.v1.assign(self.cy, means[1])

        keypoints = self.apply_rotate(keypoints)
        keypoints = self.apply_scale(keypoints)
        keypoints = self.apply_shift(keypoints)
        return keypoints


class AffineSolver(tf.keras.models.Model):
    """
    图像和点集的仿射变换
    对图或者点进行平移，缩放，旋转变换
    将其封装为tensorflow.keras模型
    """

    def __init__(self, image_shape):
        """
        Args:
            image_shape:待匹配的图像的大小
        """
        inputs = tf.keras.layers.Input(shape=(None,))
        layer = AffineLayer(image_shape)
        output = layer(inputs)
        super(AffineSolver, self).__init__(inputs, output)

    def fit(
            self,
            inputs,
            labels,
            iteration,
            lr_decay=0.99,
            visualization=False,
            save_file="test.gif",
    ):
        """
        提供模型训练的功能，通过梯度下降求解最优参数
        Args:
            inputs:输入数据，为多边形顶点坐标，矩阵形状为Nx3 (x,y,1)
            labels:标签,为多边形顶点坐标，矩阵形状为Nx3 (x,y,1)
            iteration:梯度下降算法的迭代次数
            lr_decay:每次迭代后的学习率衰减系数
            visualization:是否进行过程可视化，查看每次迭代之后的多边形绘制
            save_file:仅仅当visualization为True的时候有效，将过程保存为GIF文件储存下来
        Returns:
            最后一次迭代中损失函数的返回损失值
        """

        @tf.function
        def _fit(inputs, labels, iteration=1):
            for i in range(iteration):
                with tf.GradientTape() as tape:
                    tape.watch(inputs)
                    outputs = self(inputs)
                    loss = self.loss(labels, outputs)
                    grads = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                return loss, outputs

        if visualization is False:
            for i in range(iteration):
                loss, outputs = _fit(inputs, labels, iteration)
                tf.keras.backend.set_value(
                    self.optimizer.lr, self.optimizer.lr * lr_decay
                )
            return loss
        else:
            writer = PillowWriter(fps=20)
            plt.style.use("classic")
            with writer.saving(plt.figure(dpi=80), save_file, 100):
                for i in range(iteration):
                    loss, outputs = _fit(inputs, labels, 1)
                    tf.keras.backend.set_value(
                        self.optimizer.lr, self.optimizer.lr * lr_decay
                    )
                    plt.clf()
                    plt.title(
                        "step:%d loss:%.6f lr:%.6f"
                        % (i, loss, self.optimizer.lr.numpy())
                    )
                    plt.scatter(inputs[::, 0], inputs[::, 1], color="r", s=1)
                    plt.scatter(outputs[::, 0], outputs[::, 1], color="g", s=1)
                    plt.scatter(labels[::, 0], labels[::, 1], color="b", s=1)
                    plt.legend(["input", "output", "target"])
                    plt.ylim(self.layers[1].image_shape[0], 0)
                    plt.xlim(0, self.layers[1].image_shape[1])
                    try:
                        from IPython import display

                        display.clear_output(wait=True)
                        display.display(plt.gcf())
                    except:
                        pass
                    writer.grab_frame()
            plt.close()
            logger.info("save to %s" % (save_file))
            plt.style.use("default")
            return loss

    def transform(self, inputs):
        """
        模型推理接口，对于训练好的模型，找到了最优转换参数之后，应用于图片或者散点图中
        Args:
            inputs:RGB格式的图片，或者Nx2的点集
        Returns:
            根据求解出来的变换矩阵对于数据进行变换之后的值，大小和原始数据保持一致
        """
        cos_v = tf.math.cos(self.layers[1].theta)
        sin_v = tf.math.sin(self.layers[1].theta)
        cx = self.layers[1].cx
        cy = self.layers[1].cy
        dx = self.layers[1].dx * self.layers[1].image_shape[0]
        dy = self.layers[1].dy * self.layers[1].image_shape[1]
        scale_x = self.layers[1].scale
        scale_y = self.layers[1].scale

        shift_matrix = tf.Variable(
            [[1, 0, 0], [0, 1, 0], [dx, dy, 1]], dtype=tf.float32
        )

        rotate_matrix = tf.Variable(
            [
                [cos_v, sin_v, 0],
                [-sin_v, cos_v, 0],
                [-cx * cos_v + cy * sin_v + cx, -cx * sin_v - cy * cos_v + cy, 1],
            ],
            dtype=tf.float32,
        )

        scale_matrix = tf.Variable(
            [
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [cx * (1 - scale_x), cy * (1 - scale_y), 1],
            ],
            dtype=tf.float32,
        )

        affine_matrix = tf.matmul(tf.matmul(rotate_matrix, scale_matrix), shift_matrix)

        if len(inputs.shape) == 3 or (len(inputs.shape) == 2 and inputs.shape[1] != 3):
            M = affine_matrix.numpy()[::, :2].T
            affined = cv2.warpAffine(inputs, M, inputs.shape[:2])
            return affined
        else:
            dtype = inputs.dtype
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
            return tf.cast(tf.matmul(inputs, affine_matrix), dtype).numpy()
