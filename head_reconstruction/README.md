# 单图人脸 → 固定头部模型（粗重建版）

> 当前页面已升级为 HRN 稠密重建与打印文件导出，使用说明见 [DENSE_HEAD.md](DENSE_HEAD.md)。下文保留旧版粗预览实现说明；该旧版不适用于打印目标。

输入一张正面或轻微侧转的人脸照片，输出带纹理的 GLB。所有人共用同一个中性头模，只改变脸部内侧的几何和纹理；脸部边界、后脑、头顶、双耳和颈部保持一致。内置底模是程序生成的简化光头模型，没有头发。

## 使用

当前机器已安装独立环境并提供模型文件。运行 `start_head.bat`，打开 <http://127.0.0.1:6007/app/#/head> 即可独立使用。独立服务仅提供头部功能，换鼻功能仍使用原来的主服务。

主服务 `flask_server.py` 已注册同一套 API。重启主服务后，可在原前端侧栏进入“3D 头部模型”，或在换鼻结果下直接生成。无需额外启动 6007 服务。Vite 默认代理到主服务 6006。

新机器先运行 `setup_env.bat` 建立原项目 Python，再运行 `setup_head.bat`。如修改前端，进入 `frontend` 运行 `npm install` 和 `npm run build`。重建依赖单独安装在 `.venv-head`，不改变原 `.venv` 的 TensorFlow、PyTorch 版本。可用 `AIHB_HEAD_PYTHON` 指定其他兼容的 Python 解释器。

## 方法及边界

- 使用 Google MediaPipe Face Landmarker 的预测三维关键点，取与标准面部拓扑对应的 468 点。
- 用三维相似变换消除拍摄位置、尺度和旋转，保守混合预测深度与标准面部。
- 面部外沿两圈顶点冻结，向内部平滑增加变形权重；头部其他几何与材质固定。
- 将输入照片按三角形投影到统一 UV 图集，边缘渐变到固定底模肤色。耳部、头皮、颈部肤色不随照片变化，因此某些输入仍可看到肤色过渡。
- GLB 包含命名的 `face`、`fixed_scalp`、`fixed_ear_-1`、`fixed_ear_1`、`fixed_neck` 网格、平滑法线与嵌入纹理，坐标单位为米（模板尺度，不是真实测量尺寸）。
- 这是**关键点驱动的粗面部重建**，不是 DECA/FLAME 高精度重建，也不是扫描。鼻深、侧面和遮挡区域是估计；眼睛和嘴唇采用面部网格表面，没有独立眼球、牙齿或动画绑定。固定头模会有意舍弃人物外部头型差异。
- 当前是多部件展示模型，头部和颈部有相交部分，不保证可直接 3D 打印。自定义 OBJ/FBX 底模的面部对应关系尚未实现。

## API

- `GET /api/head/status`：检查解释器、权重和底模文件是否就绪。
- `POST /api/head/reconstruct`：`multipart/form-data`，文件字段 `image`。成功返回 `model_url`、`download_url`、`metadata`。
- `GET /api/head/template.glb`：统一底模。
- `GET /api/head/models/<id>.glb`：生成结果；加 `?download=1` 下载。

每次仅处理一个请求；忙时返回 429。图片最大 12 MB / 2400 万像素，推理前缩小到最长边 1600。拒绝多人、无人脸、脸部裁切、过小及明显侧脸。单次进程超时为 120 秒。上传文件在请求完成后移除，生成 GLB 保存在 `outputs/heads`，不会自动清理。重建过程不上传到外部服务，也不调用原换鼻接口的七牛上传和通知。

## 验证

```powershell
.venv-head/Scripts/python.exe -m unittest discover -s tests -p test_head_reconstruction.py
```

覆盖面部变化、非面部逐顶点固定、脸部边界无裂缝、GLB 纹理/法线、真实图片推理和下载、临时图片清理、格式错误、无人脸、多人脸及忙状态。

## 资源来源

- [Face Landmarker 官方文档](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [版本 1 float16 模型](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- [MediaPipe 标准面部网格](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj)

MediaPipe 标准网格的 Apache 2.0 许可证保留在 `assets/LICENSE.mediapipe`。安装脚本只在文件缺失时下载官方资源；推理不自动下载。
