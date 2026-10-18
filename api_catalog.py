"""Public interface descriptions; no model loading or credentials required."""


def describe_endpoints(region_classes, region_labels):
    return {
        "authentication": "none",
        "regions": [
            {"value": key, "label": region_labels.get(key, key)}
            for key in region_classes
        ],
        "endpoints": [
            {
                "method": "POST", "path": "/api/head/dense/reconstruct",
                "title": "稠密面部与打印头模",
                "description": "HRN 预测面部几何细节并融合固定封闭头模，输出 STL、GLB 和拓扑检查报告。",
                "content_type": "multipart/form-data",
                "parameters": [{"name": "image", "type": "file", "required": True,
                                "description": "单人正脸照片，12 MB 以内，额头与下巴完整可见。"},
                               {"name": "height_mm", "type": "number", "required": False,
                                "default": 200, "description": "总高度（含颈部），150–250 毫米。"}],
                "response_example": {"success": True, "model_url": "/api/head/models/<id>.glb",
                                     "download_url": "/api/head/models/<id>.glb?download=1",
                                     "stl_url": "/api/head/models/<id>.stl?download=1",
                                     "metadata": {"template": "hrn-fixed-head-v2", "approximate_depth": True}},
                "notes": ["使用 setup_dense_head.ps1 安装独立科研重建环境。",
                          "GET /api/head/dense/status 检查环境；GET /api/head/dense/template.glb 获取底模。",
                          "深度为单图估计，非面部固定；建议超时 510 秒；忙时返回 429。",
                          "STL 数值单位为毫米、Z 轴向上；报告中的拓扑检查不替代切片与实体细节检查。"],
            },
            {
                "method": "POST",
                "path": "/predict",
                "title": "人脸部位替换",
                "description": "上传目标人脸和来源人脸，返回替换后的 JPEG 图片。",
                "content_type": "multipart/form-data",
                "parameters": [
                    {"name": "face_image", "type": "file", "required": True,
                     "description": "目标人脸图片，将被替换所选部位。"},
                    {"name": "source_image", "type": "file", "required": True,
                     "description": "部位来源图片；兼容旧字段 nose_image，二者任选其一，source_image 优先。"},
                    {"name": "region", "type": "string", "required": False,
                     "default": "nose", "description": "替换部位，取值见下方支持部位列表。"},
                ],
                "response_example": {
                    "success": True, "error": None, "result": "<base64 JPEG>",
                    "region": "nose", "nose_pose_corrected": True,
                },
                "notes": [
                    "result 为纯 base64 JPEG；预览时加上 data:image/jpeg;base64, 前缀。",
                    "请检查 success 字段；处理失败时 error 返回原因，语义校验失败还会返回 error_code。",
                    "nose_pose_corrected 仅在鼻子替换成功时返回，表示是否启用了侧鼻校正。",
                    "接口为同步推理，建议设置 300 秒超时，避免并发提交。",
                ],
            },
            {
                "method": "GET",
                "path": "/api/endpoints",
                "title": "接口列表",
                "description": "查询接口说明、请求参数和后端实际支持的部位。",
                "content_type": None,
                "parameters": [],
                "response_example": {
                    "authentication": "none",
                    "endpoints": [{"method": "POST", "path": "/predict", "title": "人脸部位替换", "parameters": ["..."]}],
                    "regions": [{"value": "nose", "label": "鼻子"}],
                },
                "notes": [],
            },
        ],
    }
