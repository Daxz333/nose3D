import os
import torch

# 让 onnxruntime-gpu 复用 torch 自带的 CUDA/cuDNN DLL（Windows 未单独安装 CUDA 时）
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if hasattr(os, "add_dll_directory") and os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)

import numpy as np
from PIL import Image
from warper import NoseWarper
from warper import RegionWarper
from warper import REGION_CLASS_MAP
from warper import REGION_LABELS
from region_adapter import ADAPTED_REGIONS, RegionValidationError, enhance_region
from api_catalog import describe_endpoints
from utils import logger
from super_resolution import SuperResolutionModel
import warnings
from flask import Flask, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
from head_reconstruction.api import head_api
import io
import base64
import cv2
import os
from uuid import uuid1
import traceback
from qiniu import Auth, put_file
import time
from datetime import datetime, timezone, timedelta
import requests

warnings.filterwarnings("ignore")

def to_base64(img):
    img_str = cv2.imencode(".jpg", img)[1].tostring()
    b64_code = base64.b64encode(img_str)
    return b64_code


def init():
    # RegionWarper 是 NoseWarper 的泛化版本，region 缺省为 nose，换鼻行为保持不变
    warper = RegionWarper(
        face_detector_path=r"pretrained_model/FaceDetector.onnx",
        face_parsing_path=r"pretrained_model/fp_512.pth",
        face_segmentation_path=r"pretrained_model/rvm_resnet50_fp16.onnx",
        iteration=80,
        prob=0.5,
        learning_rate=0.02,
        decay=0.995,
        threshold=0.6,
        downsample_ratio=0.5,
    )
    warper.set_visualization(
        solve_affine_matrix=False,
        solve_affine_matrix_gif=False,
        result_contrast=True,
        blend=False,
        dpi=200,
    )
    return warper


app = Flask(__name__, template_folder='static', static_folder='static')
app.register_blueprint(head_api)
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(32)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # 或者设置为 'None'
warper = init()
sr_model = SuperResolutionModel("pretrained_model/GFPGANCleanv1-NoCE-C2.pth")
CORS(app, resources=r'/*')

# 需要填写你的 Access Key 和 Secret Key
access_key = os.environ.get('QINIU_ACCESS_KEY')
secret_key = os.environ.get('QINIU_SECRET_KEY')
q = Auth(access_key, secret_key) if access_key and secret_key else None
# 要上传的空间
bucket_name = os.environ.get('QINIU_BUCKET_NAME', '')
new_folder_path = 'final/'
# 保存本地路径
local_save_path = r'final'

# 上传路径
upload_path = local_save_path

tz = timezone(timedelta(hours=+8))
now = datetime.now(tz)
current_time = now.strftime("%H:%M:%S")


def save_result_to_folder(result, save_folder):
    uuid = str(uuid1())
    image_path = os.path.join(save_folder, f"{uuid}.jpg")
    Image.fromarray(result).save(image_path)
    return image_path


def qiniu_upload(localfile, key):
    if q is None or not bucket_name:
        return
    token = q.upload_token(bucket_name, key, 3600)
    ret, info = put_file(token, key, localfile)
    print(info)


def send_note_with_image(filename, region='nose'):
    id = os.environ.get('MIAOTIXING_ID')
    cdn_domain = os.environ.get('QINIU_CDN_DOMAIN')
    if not id or not cdn_domain:
        return
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    label = REGION_LABELS.get(region, region)
    text = '换' + label + '结果' + current_time + '已上传，注意查收' + '图片：' + cdn_domain.rstrip('/') + '/' + filename
    ts = str(time.time())
    type = 'json'

    request_url = "http://miaotixing.com/trigger?"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'
    }

    result = requests.post(request_url + f"id={id}&text={text}&ts={ts}&type={type}", headers=headers)
    print(result)


def save_result_to_folder(result, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    folder_name = "result_1"
    i = 1
    while os.path.exists(os.path.join(folder_path, folder_name + ".jpg")):
        i += 1
        folder_name = f"result_{i}"
    result_image_filename = f"{folder_name}.jpg"
    result_image_path = os.path.join(folder_path, result_image_filename)
    Image.fromarray(result).save(result_image_path)
    print(f'名为"result.jpg"的照片已成功保存到新的文件夹"{folder_name}"中：', result_image_path)
    return result_image_path


@app.route('/', methods=['GET'])
def index():
    return redirect('/app/')


@app.route('/api/endpoints', methods=['GET'])
def api_endpoints():
    """Public API documentation, using the same regions as prediction."""
    return jsonify(describe_endpoints(REGION_CLASS_MAP, REGION_LABELS))


@app.route("/predict", methods=["POST"])
def do_something():
    face_image = None
    nose_image = None
    res = None
    # 替换部位：nose(默认，向后兼容) / eye / lip / brow / mouth / l_eye / r_eye ...
    region = (request.form.get("region") or "nose").strip().lower()
    region_label = REGION_LABELS.get(region, region)
    response = dict(success=True, error=None, result=None, region=region)

    if region not in REGION_CLASS_MAP:
        response['success'] = False
        response['error'] = "不支持的部位:%s，可选:%s" % (region, "/".join(sorted(REGION_CLASS_MAP)))
        return jsonify(response)

    uuid = str(uuid1())
    face_image_path = os.path.join("outputs", uuid, "face_image.jpg")
    nose_image_path = os.path.join("outputs", uuid, "nose_image.jpg")
    result_image_path = os.path.join("outputs", uuid, "result_image.jpg")

    # 创建输出文件夹
    folder_path = os.path.join("outputs", uuid)
    os.makedirs(folder_path, exist_ok=True)

    if request.method == "POST":
        # source_image 为 nose_image 的新名，二者等价，优先取 source_image
        source_file = request.files.get("source_image") or request.files.get("nose_image")
        face_file = request.files.get("face_image")
        face_image = face_file.read() if face_file is not None else None
        nose_image = source_file.read() if source_file is not None else None

        if not face_image or not nose_image:
            response['success'] = False
            response['error'] = "Please upload both face and source images."
        else:
            face_image = Image.open(io.BytesIO(face_image))
            nose_image = Image.open(io.BytesIO(nose_image))

            logger.info("receive face_image:", face_image, "source_image:", nose_image, "region:", region)
            face_image = np.array(face_image.convert("RGB"))
            nose_image = np.array(nose_image.convert("RGB"))

            try:
                # blender / meaner / extension_factor 缺省时按部位配置表取值
                if region == "nose":
                    replaced = warper.warp_nose(face_image, nose_image, return_details=True, add_bg=True)
                    sr = sr_model.inference(replaced.image)
                    if replaced.alpha is not None:
                        res = enhance_region(replaced.image, sr, replaced.alpha)
                        response['nose_pose_corrected'] = True
                    else:
                        sr = cv2.resize(sr, (replaced.image.shape[1], replaced.image.shape[0]))
                        res = (sr * .3 + replaced.image * .7).astype("uint8")
                        response['nose_pose_corrected'] = False
                elif region in ADAPTED_REGIONS:
                    replaced = warper.warp_adapted(face_image, nose_image, region, return_details=True)
                    res = enhance_region(replaced.image, sr_model.inference(replaced.image), replaced.alpha)
                else:
                    res = warper(face_image, nose_image, region=region, add_bg=True)
                    sr = cv2.resize(sr_model.inference(res), (res.shape[1], res.shape[0]))
                    res = (sr * 0.3 + res * 0.7).astype("uint8")

                # 在生成结果图像后保存到本地
                result_image_path = save_result_to_folder(res, local_save_path)

                # 上传结果图片到七牛云
                qiniu_upload(result_image_path, os.path.relpath(result_image_path, local_save_path))

                # 发送提醒
                send_note_with_image(os.path.basename(result_image_path), region)

                response['result'] = to_base64(res[..., ::-1]).decode("utf-8")
            except RegionValidationError as e:
                response['error'] = str(e)
                response['error_code'] = e.code
                response['success'] = False
            except ValueError as e:
                response['error'] = str(e)
                response['success'] = False
            except Exception as e:
                logger.info("result:", "没有检测到人脸或者%s\n%s" % (region_label, traceback.format_exc()))
                response['error'] = "未检测到人脸或者%s" % (region_label)
                response['success'] = False

    logger.info("完成.")
    return jsonify(response)


# ---------------- Vue3 前端（frontend/ 构建产物） ----------------
# 构建：cd frontend && npm install && npm run build
# 访问：http://127.0.0.1:6006/app/
VUE_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")


@app.route('/app')
def vue_app_redirect():
    """补上结尾斜杠，保证相对资源路径正确解析"""
    return redirect('/app/')


@app.route('/app/')
def vue_app_index():
    index_file = os.path.join(VUE_DIST, "index.html")
    if not os.path.exists(index_file):
        return (
            "Vue 前端尚未构建，请在 frontend 目录执行：npm install && npm run build",
            404,
        )
    return send_from_directory(VUE_DIST, "index.html")


@app.route('/app/<path:filename>')
def vue_app_assets(filename):
    return send_from_directory(VUE_DIST, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
