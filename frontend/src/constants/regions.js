/**
 * 可替换部位定义。
 *
 * 依据《后续改进.md》：BiSeNet 19 类人脸解析模型已覆盖以下部位，
 * 后端 RegionWarper 把类别 ID 参数化后即可支持（已实测核实类别分布）。
 * supported = true 表示后端已开放该部位的融合参数，false 表示仍走实验参数。
 */
export const REGIONS = [
  {
    value: 'nose',
    label: '鼻子',
    en: 'NOSE',
    classIds: [10],
    desc: '侧鼻到正鼻 · 自动姿态校正',
    sourceHint: '提供鼻子来源的人脸照片',
    targetHint: '将被替换鼻子的人脸照片',
    tip: '侧脸来源换到正脸时自动校正鼻梁、鼻尖表面，保留目标鼻孔与鼻柱结构，避免镜像亮斑；保护眼睛、嘴唇与背景。单张侧脸的遮挡侧采用二维近似。',
    supported: true,
  },
  {
    value: 'eye',
    label: '眼睛',
    en: 'EYE',
    classIds: [4, 5],
    desc: '双眼 · 专用融合参数',
    sourceHint: '提供眼睛来源的人脸照片（建议双眼睁开、正视镜头）',
    targetHint: '将被替换眼睛的人脸照片',
    supported: true,
  },
  {
    value: 'lip',
    label: '嘴唇',
    en: 'LIP',
    classIds: [12, 13],
    desc: '上下唇 · 保留唇色与唇线',
    sourceHint: '提供嘴唇来源的人脸照片（张嘴 / 闭嘴状态与目标一致）',
    targetHint: '将被替换嘴唇的人脸照片',
    tip: '两张照片的张嘴 / 闭嘴状态需一致；融合保留供体唇色并校正亮度，保护目标口腔。',
    supported: true,
  },
  {
    value: 'brow',
    label: '眉毛',
    en: 'BROW',
    classIds: [2, 3],
    desc: '双眉 · 左右分别对齐',
    sourceHint: '提供眉毛来源的人脸照片',
    targetHint: '将被替换眉毛的人脸照片',
    supported: true,
  },
  {
    value: 'mouth',
    label: '口腔',
    en: 'MOUTH',
    classIds: [11],
    desc: '口腔内部 · 张嘴幅度需相近',
    sourceHint: '提供口腔来源的人脸照片（需张嘴，幅度与目标相近）',
    targetHint: '将被替换口腔的人脸照片（需张嘴）',
    tip: '口腔替换仅修改嘴内区域，保留目标嘴唇。两张照片都需张嘴且幅度相近；闭嘴或幅度差异过大时会提示重新选图。',
    supported: true,
  },
  ...[
    ['l_eye', '左眼', [4], '睁眼并正视镜头'],
    ['r_eye', '右眼', [5], '睁眼并正视镜头'],
    ['l_brow', '左眉', [2], '眉毛清晰、无遮挡'],
    ['r_brow', '右眉', [3], '眉毛清晰、无遮挡'],
    ['u_lip', '上唇', [12], '唇线清晰、嘴巴状态相近'],
    ['l_lip', '下唇', [13], '唇线清晰、嘴巴状态相近'],
  ].map(([value, label, classIds, hint]) => ({
    value, label, classIds, en: value.toUpperCase(),
    desc: `${label} · 同名部位对齐`,
    sourceHint: `提供${label}来源的人脸照片（${hint}）`,
    targetHint: `将被替换${label}的人脸照片`,
    tip: value.includes('lip')
      ? '仅替换所选嘴唇，保留另一侧唇和口腔；两张照片的张嘴 / 闭嘴状态需一致。'
      : '左 / 右按人脸解析模型的同名侧匹配，不镜像；请使用方向一致、未做水平翻转的照片。',
    supported: true,
  })),
]

export const REGION_MAP = REGIONS.reduce((acc, item) => {
  acc[item.value] = item
  return acc
}, {})

export function getRegion(value) {
  return REGION_MAP[value] || REGIONS[0]
}
