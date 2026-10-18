/**
 * 推理流水线阶段定义。
 *
 * 顺序严格对齐 README「运行逻辑」中的主链路：
 *   上传校验 → 人脸检测裁剪 → BiSeNet 分割 → 关键点采样
 *   → 仿射对齐求解 → 供体变换与颜色对齐 → alpha 融合 → GFPGAN 超分 → 落盘返回
 *
 * weight 为该阶段在总耗时中的相对占比（合计 100），
 * 用于按预估耗时把整体进度切分成段。权重依据各环节的计算量估算：
 * 仿射对齐（Adam 80 次迭代）与 GFPGAN 超分最重，融合次之，检测/分割较轻。
 *
 * 注意：后端 /predict 是同步接口，不返回真实进度，
 * 因此进度值是基于「历史平均耗时」的估算，仅用于体验反馈。
 */
export const PROCESS_STAGES = [
  {
    key: 'upload',
    label: '上传图片并校验参数',
    en: 'UPLOAD & VALIDATE',
    weight: 2,
    desc: '提交 face_image / source_image / region，校验图片和替换部位',
  },
  {
    key: 'detect',
    label: '人脸检测与方形裁剪',
    en: 'FACE DETECT',
    weight: 6,
    desc: '缩放至 640 → FaceDetector 检测人脸框 → 裁剪为方形并归一化到 512',
  },
  {
    key: 'parse',
    label: '部位语义分割',
    en: 'SEMANTIC PARSE',
    weight: 8,
    desc: 'BiSeNet（fp_512，19 类）推理，取目标类别生成区域掩码',
  },
  {
    key: 'keypoint',
    label: '掩码边缘关键点采样',
    en: 'KEYPOINT SAMPLE',
    weight: 5,
    desc: '拉普拉斯二阶微分求边缘，sigmoid 阈值二值化后随机采样关键点',
  },
  {
    key: 'affine',
    label: '仿射对齐参数优化',
    en: 'AFFINE SOLVING',
    weight: 20,
    desc: 'AffineSolver + Adam 迭代求解旋转 / 缩放 / 平移，最小化最近点匹配损失',
  },
  {
    key: 'align',
    label: '供体变换与颜色对齐',
    en: 'WARP & COLOR MATCH',
    weight: 10,
    desc: '对来源图做仿射变换，并用均值方差一致化对齐肤色与光照',
  },
  {
    key: 'blend',
    label: '图像融合',
    en: 'ALPHA BLEND',
    weight: 17,
    desc: 'alpha_blender_v6：掩码膨胀 / 腐蚀 / 高斯羽化，去眼睛区域并 inpaint 补背景',
  },
  {
    key: 'sr',
    label: '超分增强',
    en: 'SUPER RESOLUTION',
    weight: 20,
    desc: 'GFPGAN 推理后按 sr * 0.3 + res * 0.7 混合回填',
  },
  {
    key: 'save',
    label: '结果保存与返回',
    en: 'SAVE & RESPOND',
    weight: 12,
    desc: '结果落盘 final/、上传对象存储，并以 base64 JPG 返回',
  },
]

/** 权重合计（用于进度分段归一化） */
export const TOTAL_WEIGHT = PROCESS_STAGES.reduce((sum, stage) => sum + stage.weight, 0)

/** 无历史样本时的默认预估耗时（秒） */
export const DEFAULT_ESTIMATED_SECONDS = 30
