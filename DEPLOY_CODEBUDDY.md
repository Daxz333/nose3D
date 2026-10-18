# AIHB2 换机部署交接文档（供 CodeBuddy / DeepSeek V4.1 Pro 阅读）

本文依据 2026-10-03 的项目源码编写。目标是在另一台电脑恢复项目的真实推理和 Vue 工作台。本文编写阶段未在目标电脑执行安装；执行者必须按验收步骤验证，不能把本机 README 中的历史验证结果当作目标机器的结果。

## 1. 给部署助手的任务说明

你是接手此项目的 CodeBuddy。先阅读本文，再检查实际文件和目标机器环境，按顺序完成部署、排错和验收。默认目标是 **Windows 64 位 + NVIDIA GPU + 本机/可信局域网使用**。优先使用构建后的前端，由 Flask 在同一个端口托管。

- CodeBuddy 和 DeepSeek 是部署助手，不是项目运行依赖；本项目不需要 DeepSeek API Key。
- 所有命令除特别说明外均在项目根目录的 **PowerShell** 执行。先将工作目录切换到目标电脑实际路径，不要照抄原电脑盘符。
- 保留算法行为、11 个替换部位、现有页面和接口契约。不要为了安装成功随意升级 torch、torchvision、TensorFlow、NumPy 或 ONNX Runtime。
- 不复制旧 `.venv` 和 `node_modules` 作为新电脑的运行环境；重新创建。需要迁移的是代码、模型和必要缓存。
- 首次部署采用本地模式，按第 5 节隔离现有云上传和通知。不要使用代码里的历史密钥，不要发送测试通知。
- 每一阶段检查命令退出码和输出；安装脚本显示“完成”不等于通过验收。失败时定位第一个真实错误，修复后再继续。
- 若目标是 Linux、macOS、无 NVIDIA GPU 或不兼容旧 CUDA 软件栈的显卡，先报告适配差异；本仓库没有完成这些环境的默认部署支持，不能声称直接可运行。

## 2. 项目运行结构

后端入口是 `flask_server.py`，默认监听 `0.0.0.0:6006`。启动时就加载所有模型，尚未收到请求也可能下载缓存。`warper.py` 负责推理主流程，`region_adapter.py` 负责部位适配，`nose_pose.py` 负责侧鼻到正脸校正，`api_catalog.py` 提供接口说明。

前端为 `frontend/` 下的 Vue 3 + Vite 5 工程。`npm run build` 生成 `frontend/dist/`，后端通过 `/app/` 托管。根路由 `/` 重定向到工作台。`GET /api/endpoints` 返回接口说明，`POST /predict` 执行真实推理。接口目前无需登录。

Python 依赖以 `requirements.txt` 为准，核心版本如下：

- Python：优先 3.10.11，与 `setup_env.bat` 一致。
- torch `2.0.1+cu118`，torchvision `0.15.2+cu118`。
- tensorflow `2.13.1`，用于仿射优化；本部署方案在 Windows 上使用 CPU。
- onnxruntime-gpu `1.16.3`、onnx `1.14.1`。
- numpy `1.24.3`、opencv-python `4.8.1.78`。
- basicsr `1.4.2`、realesrgan `0.3.0`、facexlib `0.3.0`。
- Flask `2.3.3`，另按现有脚本补装 `setuptools<81`，保留 pyamg 所需的 `pkg_resources`。

前端优先复用原电脑已使用的 Node 24 / npm 11 主版本并以 `npm ci` 安装锁文件。具体可用版本由目标机器实际构建验证；不要因为 README 写了 Node 18+ 就主动安装已过时的版本。

## 3. 原电脑需要交付的文件

复制项目代码及目录结构，至少包括：根目录所有运行所需 `.py` 文件、`requirements.txt`、两个 `.bat` 脚本、`face_detector/`、`segmentation/`、`super_resolution/`（包含内置 gfpgan 源码）、`pretrained_model/`、`frontend/` 源码及 public 资源、`tests/`、`images/` 和本文。前端 `package-lock.json`、`.env.development`、`.env.production` 也要保留。`static/` 可一并复制，但不要用旧页面作为新部署入口。

可以不迁移 `.venv/`、`frontend/node_modules/`、旧日志、`__pycache__/`、`outputs/` 和 `final/` 的历史结果。`frontend/dist/` 可以携带，但目标电脑仍应重新构建一次。训练标注和 notebook 不是运行必需项。

必需主模型及当前文件字节数（用于初步发现漏传/截断，不替代哈希校验）：

- `pretrained_model/FaceDetector.onnx`：1,755,813。
- `pretrained_model/fp_512.pth`：53,289,081。
- `pretrained_model/rvm_resnet50_fp16.onnx`：53,752,431。
- `pretrained_model/GFPGANCleanv1-NoCE-C2.pth`：348,632,874。

`FaceLandmark.onnx`、`rvm_mobilenetv3_fp16.onnx` 为当前主流程未直接使用的备用模型，可以一起复制。`setup_env.bat` **不会补齐上述四个主模型**，缺失时应从原项目副本补齐，不要任意下载同名替代品。

还需从原电脑单独备份以下缓存文件，放进迁移包中的 `deployment-cache/`（自行创建），重建环境后再放回目标位置：

- `.venv/Lib/site-packages/facexlib/weights/parsing_parsenet.pth`。
- `.venv/Lib/site-packages/facexlib/weights/detection_Resnet50_Final.pth`。
- `.venv/Lib/site-packages/weights/RealESRGAN_x2plus.pth`。
- torch hub 的 `checkpoints/resnet18-5c106cde.pth`。

第四项容易遗漏：`segmentation/resnet.py` 在加载完整解析权重前，仍会调用 `modelzoo.load_url` 加载 ResNet18。仅复制 `pretrained_model/` 不保证离线启动成功。

在原电脑定位 torch hub 缓存：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.hub.get_dir())"
```

迁移前后分别执行以下命令，对比同名模型的 SHA256：

```powershell
Get-ChildItem .\pretrained_model -File | Get-FileHash -Algorithm SHA256
# deployment-cache 已创建且包含缓存时再执行
Get-ChildItem .\deployment-cache -File | Get-FileHash -Algorithm SHA256
```

## 4. 目标电脑环境安装

### 4.1 先检查环境

```powershell
Set-Location 'D:\AIHB2'  # 替换为目标电脑实际项目目录
Get-Location
Test-Path .\flask_server.py
nvidia-smi
node --version
npm.cmd --version
uv --version
Get-ChildItem .\pretrained_model -File | Select-Object Name,Length
```

`nvidia-smi` 应能识别显卡和驱动，但它显示的 CUDA 版本不等于 Python 中 torch 的运行时版本。后面必须运行 CUDA 张量计算和 ONNX Session 检查。

安装缺少的 Node、uv 或 NVIDIA 驱动时使用对应官方渠道。显卡实际兼容性以检测结果为准，尤其是比此旧依赖栈更新的 GPU；不要把“存在 NVIDIA 显卡”当作兼容证明。仓库未给出经过测量的最低显存/内存要求，单请求冒烟测试是必要验收项。

### 4.2 建立独立 Python 环境

如果误带了旧 `.venv`，先移到项目内一个未占用的备份目录，不要复用，也不要自动删除用户已有环境。以下命令以 `.venv` 不存在为前提：

```powershell
uv python install 3.10.11
uv venv --python 3.10.11 --seed .venv
.\.venv\Scripts\python.exe -m pip install "setuptools<81"
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "setuptools<81"
.\.venv\Scripts\python.exe -m pip check
```

每条安装命令成功后再执行下一条。若 uv 无法提供指定补丁版本，先记录错误，可以使用已安装的官方 Python 3.10 创建环境：`py -3.10 -m venv .venv`；记录实际补丁版本并完成全部验收。不要自动换到 Python 3.12/3.14。

始终使用 `.venv\Scripts\python.exe -m pip`，避免把依赖装入系统 Python。无需激活环境。PowerShell 遇到 npm 脚本执行策略限制时使用本文的 `npm.cmd`，不必修改全局执行策略。

### 4.3 恢复附加模型缓存

假设第 3 节四个文件已平铺放在 `deployment-cache/`：

```powershell
New-Item -ItemType Directory -Force .\.venv\Lib\site-packages\facexlib\weights | Out-Null
New-Item -ItemType Directory -Force .\.venv\Lib\site-packages\weights | Out-Null
Copy-Item .\deployment-cache\parsing_parsenet.pth .\.venv\Lib\site-packages\facexlib\weights\
Copy-Item .\deployment-cache\detection_Resnet50_Final.pth .\.venv\Lib\site-packages\facexlib\weights\
Copy-Item .\deployment-cache\RealESRGAN_x2plus.pth .\.venv\Lib\site-packages\weights\
$aihbHub = & .\.venv\Scripts\python.exe -c "import torch; print(torch.hub.get_dir())"
$aihbCheckpoints = Join-Path $aihbHub 'checkpoints'
New-Item -ItemType Directory -Force $aihbCheckpoints | Out-Null
Copy-Item .\deployment-cache\resnet18-5c106cde.pth $aihbCheckpoints
```

若没有原缓存，有网络时相关库会从源码指定的地址自动下载。准确来源可查看 `segmentation/resnet.py`、`super_resolution/gfpgan_model.py` 和已安装的 facexlib detection/parsing 模块。检查下载结果不是 HTML 错误页，不能只检查文件名是否存在。

完全离线安装还需要事先准备匹配 Windows/Python 3.10 的完整 Python wheel 依赖包、Python 安装介质及 npm 离线依赖缓存；只有模型缓存不能完成离线部署。不要宣称现有一键脚本支持完整离线安装。

### 4.4 检查 GPU 与实际 ONNX provider

```powershell
.\.venv\Scripts\python.exe -c "import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print((torch.ones(1,device='cuda')+1).cpu())"
.\.venv\Scripts\python.exe -c "import os,torch; from pathlib import Path; h=os.add_dll_directory(str(Path(torch.__file__).parent/'lib')); import onnxruntime as ort; print(ort.__version__,ort.get_available_providers()); sessions=[ort.InferenceSession(p,providers=['CUDAExecutionProvider']) for p in ['pretrained_model/FaceDetector.onnx','pretrained_model/rvm_resnet50_fp16.onnx']]; print([s.get_providers() for s in sessions]); assert all('CUDAExecutionProvider' in s.get_providers() for s in sessions)"
.\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__); print(tf.reduce_sum(tf.ones([2,2])).numpy())"
```

前两项必须成功；ONNX 包声称支持 CUDA 不足以证明 Session 实际启用了 CUDA，所以这里检查两个模型的 Session。TensorFlow 在此方案中不需要列出 GPU，`ceshi.py` 也不能替代以上检查。

## 5. 首次推理前处理外部服务

**当前源码没有可直接使用的云服务禁用环境变量，也不自动读取后端 `.env`。** 不要只写一个 `ENABLE_UPLOAD=false` 就声称已禁用。

`flask_server.py` 的成功流程在本地保存后依次调用 `qiniu_upload(...)`、`send_note_with_image(...)`，随后才写入响应的 base64 图片。它包含历史硬编码云密钥和通知目标；通知请求未设置 timeout。外部服务报错可能被包装为“未检测到人脸”，也可能造成等待。

本地部署时，由部署助手对目标副本做最小修改，并记录差异：

1. 删除历史 `access_key`、`secret_key` 的字面量和 `q = Auth(...)` 初始化，移除不用的七牛上传/通知实现及调用，或改成默认关闭的显式条件分支。
2. 保留 `save_result_to_folder(...)`、`local_save_path`、`response['result'] = ...` 和原有算法、错误处理、JSON 契约。
3. 若保留可选云功能，新增的配置必须实际由代码读取；仅启用后初始化客户端并校验配置，通知请求设置合理超时。外部服务失败不应让已经完成的本地推理丢失结果。
4. 本地模式验收应确认没有调用上传和通知。历史密钥应由持有人轮换；不要将它们复制进文档、日志或新配置。

如用户确实需要恢复云服务，再接入其提供的新配置。该功能属于独立验收项。本文仅提供部署说明，**并未修改当前仓库的这些业务代码**。

## 6. 构建前端并启动

### 6.1 推荐：单端口运行

确认 `frontend/.env.production` 中 `VITE_API_BASE=` 为空。同源请求会访问当前网站的 `/predict` 和 `/api/endpoints`。

```powershell
Push-Location .\frontend
npm.cmd ci
# 上一步成功后再构建
npm.cmd run build
Pop-Location
Test-Path .\frontend\dist\index.html
.\.venv\Scripts\python.exe flask_server.py
```

让后端保持运行，等待模型加载完毕和监听日志出现，然后用浏览器打开 `http://127.0.0.1:6006/app/`。关闭终端或按 Ctrl+C 会停止服务。不要仅因为进程已启动就认定模型加载完成。

### 6.2 开发模式（需要修改前端时）

保持后端运行，在另一个 PowerShell 终端进入 `frontend/` 后执行 `npm.cmd run dev`。默认访问 `http://127.0.0.1:5173`，以 Vite 实际输出端口为准。

`frontend/.env.development` 的 `VITE_API_TARGET=http://127.0.0.1:6006` 是 Vite 进程访问的后端地址。代理覆盖 `/predict` 和 `/api`。同一台电脑上运行前后端时不需要替换为局域网 IP。

`VITE_API_BASE` 是构建时配置，改完生产配置必须重新构建。若浏览器在别的电脑上，生产接口地址中的 `127.0.0.1` 指向浏览器那台电脑，因此优先保持同源配置为空。

### 6.3 原有批处理脚本的限制

- `setup_env.bat` 会安装 uv、Python、依赖并尝试构建前端，但模型下载使用第三方镜像，部分失败只输出 WARN，且不会下载四个主模型和 ResNet18 缓存。它也可能跳过已存在但失效的环境或 node_modules。优先使用本文分步命令。
- `start.bat` 会启动后端和 Vite，等待约 6 秒后打开页面；不做真实就绪检查。它不是后台常驻服务管理器。
- Flask 目前使用开发服务器。本文的单端口方案适合本机/可信局域网功能部署，不等于已完成公网生产运维部署。

## 7. 验收：不能只验证页面能打开

在项目根目录执行，测试会写入 `outputs/`；真实模型测试可能耗时较长：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/smoke_regions.py --enhance
.\.venv\Scripts\python.exe tests/smoke_nose_pose.py
```

上述冒烟脚本使用本地推理；侧鼻脚本使用替身隔离云上传与通知，并测试路由逻辑和 JPEG 响应。它们不替代真实监听端口的 HTTP 验收。模型缺失时仍可能触发模型下载。

保持后端运行，在另一个终端检查：

```powershell
Invoke-RestMethod http://127.0.0.1:6006/api/endpoints
# 仅在第 5 节本地模式修改完成后调用真实推理接口
curl.exe --fail --max-time 300 -sS -X POST -F "face_image=@images/55.jpg" -F "source_image=@images/761.jpg" -F "region=nose" http://127.0.0.1:6006/predict -o deployment-response.json
$aihbResponse = Get-Content .\deployment-response.json -Raw | ConvertFrom-Json
if (-not $aihbResponse.success -or -not $aihbResponse.result) { throw ($aihbResponse | ConvertTo-Json -Compress) }
[IO.File]::WriteAllBytes((Join-Path (Get-Location) 'deployment-result.jpg'), [Convert]::FromBase64String($aihbResponse.result))
```

先检查 curl 的退出码，再读取响应，避免把旧文件误认为本次结果。HTTP 200 也可能返回 `success=false`，必须检查 JSON。`result` 为不带 data URL 前缀的 base64 JPG。

最终人工检查：

- `/app/` 能显示页面，静态资源无 404；“接口列表”能获取数据和显示部位示例。
- 工作台上传两张人脸图片后获得结果，对比预览和下载正常；打开 `deployment-result.jpg` 检查实际图像。
- 11 个部位选项存在：`nose`、`eye`、`l_eye`、`r_eye`、`lip`、`u_lip`、`l_lip`、`brow`、`l_brow`、`r_brow`、`mouth`。
- 单元测试、部位增强冒烟、侧鼻冒烟分别记录退出码和报告；报告位于 `outputs/region-adapter-smoke/`、`outputs/nose-pose/`。
- 口腔/嘴唇状态不匹配时返回业务错误是设计行为，不要用始终返回成功掩盖问题。
- 重启后仍可运行。记录启动目录、解释器、访问地址、模型位置和所有实际改动。

## 8. 常见故障处理

**找不到模型、路径不存在：** 确认终端位于项目根目录，四个主模型存在且大小/哈希正确；检查附加缓存。不要只把 Python 入口的绝对路径传给解释器却在其他目录运行。

**CUDA 不可用或 no kernel image：** 检查 NVIDIA 驱动、实际 torch 版本及 GPU 兼容性，运行第 4.4 节的张量计算。项目多处写死 `.cuda()`、CUDA provider 和 FP16，不能仅改一处 device 就声称支持 CPU。

**ONNX DLL 加载失败/退回 CPU：** 确认没有同时混装 CPU 版 onnxruntime；核对 torch 的 `lib` 目录和 CUDA/cuDNN DLL 搜索路径。现有服务使用 `os.add_dll_directory`；必要时像第 4.4 节一样保留返回句柄。缺少 MSVC 运行库时再按具体 DLL 错误安装官方 x64 运行库，不要盲目更换整个 CUDA 依赖栈。

**pkg_resources 或 torchvision.transforms.functional_tensor 报错：** 前者检查 `setuptools<81` 是否安装；后者先核对是否错误升级了 torchvision。按锁定版本恢复，运行 `pip check`。

**加载模型时 EOF、invalid load key、下载停滞：** 检查是不是残缺模型或 HTML 页面，按原缓存哈希恢复；先补齐四类附加缓存。不要将镜像请求的 HTTP 错误页当作权重文件。

**页面 404/白屏：** 检查前端构建成功且 `frontend/dist/index.html` 存在，使用带尾斜杠的 `/app/`；检查浏览器资源请求。不要打开 `static/index.html`，其旧接口地址写死。

**页面正常但接口网络异常：** 检查后端加载日志、6006 监听状态和 Vite 代理目标；端口冲突时查明占用进程，不要直接结束无关进程。改后端端口还需同步前端开发代理和所有访问地址。

**长时间无响应或“未检测到人脸”：** 先看后端 traceback，确认云上传和通知已经隔离，再检查输入图片、显存和模型错误。前端请求超时为 300000 毫秒，界面进度是估算，不能作为后端实际进度证据。

**显存不足：** 关闭无关 GPU 程序，先单请求串行测试，避免同时运行服务和多套冒烟脚本导致重复加载模型。不要启动多个模型 worker 试图解决问题。

## 9. 局域网与长期运行边界

局域网使用服务器实际 IPv4 地址访问 `http://服务器IP:6006/app/`。当前服务已绑定 `0.0.0.0`；只有需要其他电脑访问时才配置 Windows 防火墙规则，并限定可信网络。无需对外开放 Vite 的 5173 端口即可使用构建后的前端。

如果另有公网/长期运行要求，需要单独落实 WSGI 托管、HTTPS/反向代理、访问控制、上传限制、请求超时、日志、输出清理和进程恢复。当前全局模型及部分可变状态不应直接多线程/多进程并发使用；需串行推理或经过验证的锁/队列设计。不要仅换一个 Web 启动命令就宣布生产就绪。

## 10. CodeBuddy 最终交付格式

完成后向用户说明：目标系统和 GPU；Python/torch/ONNX/Node 实际版本；项目路径和启动命令；可访问地址；模型和缓存是否完整；本地模式如何关闭外部调用；前端构建、GPU 检查、单元测试、真实推理的实际结果；改动文件；剩余阻塞项。未执行的步骤明确写“未验证”，不得编造测试成功。

可直接粘贴给 CodeBuddy 的任务：

> 请阅读项目根目录 DEPLOY_CODEBUDDY.md，并结合实际源码，在这台电脑完成 AIHB2 的部署。先检查系统、NVIDIA GPU、Python、Node 和模型完整性，再重建环境、恢复缓存、按文档将外部上传/通知设为本地模式、构建前端、启动服务并完成真实推理验收。保留现有算法和全部部位功能，不要随意升级锁定依赖，不要使用历史密钥或发送外部通知。记录每一步实际结果和代码改动，最后给出访问地址、启动方式、验收结果及未解决的问题。
