# HRN 稠密人脸与打印头模

面向 150–250 mm 高的头像验证：HRN 从单张照片预测人脸形状、非线性变形和细节位移，再把这些变化融合到同一个封闭头模上。耳朵、头顶、后脑、颈部和面部边界使用固定几何。页面默认展示无贴图灰模，可单独近看五官。

## 本机使用

运行项目根目录 `start_head.bat`，打开 <http://127.0.0.1:6007/app/#/head>。完整换鼻工作台使用原主服务 6006；重启该服务后，同一套稠密 API 也会生效。

选择一张正面、自然表情、无遮挡的清晰人脸照片，设置模型总高度（含颈部），生成后下载 STL。GLB 用于浏览器或建模工具预览，JSON 报告记录网格与尺寸检查。

## 环境与资源

首次部署先安装原项目 Python（`setup_env.bat`），然后在 PowerShell 执行 `./setup_dense_head.ps1`。需要已有 Ubuntu-22.04 WSL、可用的 NVIDIA GPU 与网络。脚本在项目目录的 `.runtime/hrn` 中安装 Python、CUDA 编译工具和 PyTorch，不安装或替换系统显卡驱动。Windows `.venv-head` 负责输入校验、网格处理和 Web 服务；Linux 独立环境负责 HRN 推理。

脚本从作者的官方 HRN GitHub 和 ModelScope 仓库下载源码、模型与完整头模。文件位于 `third_party` 和 `.runtime`，均排除在版本控制之外。源代码的本地兼容调整由 `tools/hrn_setup/prepare_vendor.py` 重现：CUDA 渲染替代 OpenGL 上下文；CPU 执行 TensorFlow 分割；HRN 的 UV 三角形重心插值统一使用 nvdiffrast，避免另外编译仅训练部分需要的 PyTorch3D。

部署会执行一张本地照片的完整推理与网格检查，成功后才写入 `.runtime/hrn/ready.json` 并启用稠密 API。不会回退成 468 点粗重建。此版本按个人/科研验证使用；HRN 涉及的 BFM、完整头模和预训练资产的适用许可需分别遵守，不能仅按源代码的 Apache 许可证推断所有资产可商用。

## 几何与打印检查

- 修复完整头部模板、封闭颈部并切出平底，只做一次并缓存。
- 对共用头顶和颈部做几何平滑，同时保护耳部、面部和底面；照片发际线与颏下边缘的细节位移逐步衰减，减少可见融合痕迹。
- 在较细的模板表面采样 HRN 的面部变形和实际几何位移。细分只提供采样网格；面部信息来自模型预测。
- 所有照片从同一拓扑和外部几何开始；面部边界采用渐变权重，局部自交修复可改变面部三角形连接。模型尺寸按固定底模缩放，避免不同人物的头模外部比例变化。
- STL 使用毫米数值、Z 轴向上、底部 Z=0；GLB 使用米、Y 轴向上。STL 格式本身不记录单位，切片软件导入时请选择毫米。
- 导出要求单一封闭实体、正体积、法线绕序一致、零边界边、零非流形边，以及 PyMeshFix 三角形自交检测通过。发现局部自交时先修复并重新检查，同时限制固定外侧区域变化不超过 0.005 mm；修复或复检失败则停止输出。检测前将坐标量化到 1/1024 mm 的二进制格点，避免近共面三角形的浮点噪声；量化带来的坐标变化不到 1 微米。

这些是网格拓扑检查，不是人物相似度、扫描精度或实物打印质量保证；当前没有对打印机、材料、最小壁厚、支撑或所有自交情况进行认证。单张照片无法唯一确定鼻部深度和不可见部分。请检查灰模五官与侧面，再在切片软件中确认尺寸和切片；当前按实心封闭头模导出，未自动镂空或生成支撑。

## API

- `GET /api/head/dense/status`：稠密环境就绪状态与模板地址。
- `GET /api/head/dense/template.glb`：固定头模。
- `POST /api/head/dense/reconstruct`：`multipart/form-data`，`image` 文件，`height_mm`（150–250，默认 200）。
- 成功返回 `model_url`、`download_url`、`stl_url`、`report_url`、`metadata`。
- 文件保存在 `outputs/heads`；上传原图与中间预测在请求结束时删除。推理和头模生成不上传照片，不触发换鼻结果的对象存储上传和通知。

## 验证

```powershell
.venv-head/Scripts/python.exe -m unittest discover -s tests -p test_dense_head.py
.venv-head/Scripts/python.exe -m tools.hrn_setup.validate_runtime
```

前一条检查封闭模板、外部顶点冻结、位移确实改变几何，以及 150/250 mm STL 往返读取的尺寸和拓扑；后一条运行真实照片推理。

资源：[HRN 官方源码](https://github.com/youngLBW/HRN)、[稠密人脸模型](https://www.modelscope.cn/models/damo/cv_resnet50_face-reconstruction)、[完整头模资产](https://www.modelscope.cn/models/damo/cv_HRN_head-reconstruction)、[nvdiffrast](https://github.com/NVlabs/nvdiffrast)。
