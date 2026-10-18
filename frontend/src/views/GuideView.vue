<script setup>
import { REGIONS } from '@/constants/regions'

const PIPELINE = [
  { step: '01', title: '人脸检测', desc: 'RetinaFace（FaceDetector.onnx）定位人脸框，裁剪为正方形区域并归一化到 640 / 512。' },
  { step: '02', title: '部位解析', desc: 'BiSeNet 19 类人脸解析模型（fp_512.pth）输出语义掩码，取目标类别作为区域掩码。' },
  { step: '03', title: '掩码关键点', desc: '对掩码做拉普拉斯二阶微分，经 sigmoid / 阈值提取边缘，随机采样得到轮廓关键点。' },
  { step: '04', title: '仿射对齐', desc: 'TensorFlow 自定义 AffineLayer 以 Adam 迭代求解旋转 / 缩放 / 平移共 5 个参数，损失含最近点匹配与多边形直径约束。' },
  { step: '05', title: '颜色对齐', desc: '全局 / 局部均值方差一致化，消除供体与目标之间的肤色与光照差异。' },
  { step: '06', title: 'Alpha 融合', desc: 'alpha_blender_v6：掩码膨胀 / 腐蚀 / 高斯羽化 + 去眼睛区域 + inpaint 补背景。' },
  { step: '07', title: '超分增强', desc: 'GFPGAN 增强后按 sr * 0.3 + res * 0.7 混合，提升细节与清晰度。' },
  { step: '08', title: '结果输出', desc: '结果落盘 final/ 并上传对象存储，接口以 base64 JPG 返回给前端。' },
]

const LIBS = [
  { name: 'PyTorch 2.0.1', note: 'CUDA 11.8 · 分割 / GFPGAN / RealESRGAN' },
  { name: 'TensorFlow 2.13', note: 'CPU · 仿射对齐求解' },
  { name: 'ONNX Runtime GPU', note: 'FaceDetector · RVM 人脸分割' },
  { name: 'OpenCV / NumPy', note: '图像处理与数值计算' },
]
</script>

<template>
  <div>
    <header>
      <h1>算法说明</h1>
      <div class="subtitle">
        <span class="dot" />
        PIPELINE · 检测 → 分割 → 对齐 → 融合 → 超分
      </div>
    </header>

    <div class="divider">
      <span class="divider-line" />
      <span class="divider-text">PROCESSING PIPELINE</span>
      <span class="divider-line" />
    </div>

    <section class="panel corner">
      <ol class="steps">
        <li v-for="item in PIPELINE" :key="item.step" class="step">
          <span class="step-no mono">{{ item.step }}</span>
          <div class="step-body">
            <h3>{{ item.title }}</h3>
            <p>{{ item.desc }}</p>
          </div>
        </li>
      </ol>
    </section>

    <div class="divider">
      <span class="divider-line" />
      <span class="divider-text">REGION EXPANSION ROADMAP</span>
      <span class="divider-line" />
    </div>

    <section class="panel corner">
      <p class="muted">
        19 类解析模型已覆盖眉毛、眼睛、嘴唇、口腔等部位，后端已把类别 ID 参数化（RegionWarper），
        鼻子与眼睛使用原有融合流程，眉毛、嘴唇与口腔使用语义保护和确定性对齐；
        已开放全部部位及单侧选项，口腔要求两张照片张嘴且幅度相近。
        下表为各部位对应的类别与状态。
      </p>

      <div class="table">
        <div class="tr th mono">
          <span>部位</span>
          <span>类别 ID</span>
          <span>状态</span>
        </div>
        <div v-for="item in REGIONS" :key="item.value" class="tr">
          <span class="td-label">
            {{ item.label }}
            <small class="mono">{{ item.en }}</small>
          </span>
          <span class="mono">{{ item.classIds.join(' + ') }}</span>
          <span>
            <span class="tag" :class="item.supported ? 'ok' : 'warn'">
              {{ item.supported ? '已支持' : '规划中' }}
            </span>
          </span>
        </div>
      </div>

      <p class="note">
        注意：不同部位的最优融合参数（膨胀 / 腐蚀 / 羽化 / 扩展系数）差异较大，
        眼睛等高频细节部位使用专门的融合策略：掩码改用小幅膨胀而非纵向拉伸扩展，
        羽化核从 71 降到 15，颜色对齐窗口从 111 降到 31，
        并关闭「去眼睛区域」、改用肤色均值填充掩码外围，避免出现暗环与细节丢失。
      </p>
    </section>

    <div class="divider">
      <span class="divider-line" />
      <span class="divider-text">RUNTIME STACK</span>
      <span class="divider-line" />
    </div>

    <section class="panel corner">
      <div class="libs">
        <div v-for="lib in LIBS" :key="lib.name" class="lib">
          <span class="lib-name">{{ lib.name }}</span>
          <span class="lib-note">{{ lib.note }}</span>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.steps {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}

.step {
  display: flex;
  gap: 14px;
  padding: 14px 16px;
  border-radius: 8px;
  border: 1px solid var(--line-soft);
  background: rgba(5, 10, 18, 0.5);
  transition: 0.25s;
}

.step:hover {
  border-color: var(--line);
  background: rgba(0, 240, 255, 0.05);
}

.step-no {
  font-size: 18px;
  line-height: 1.2;
  color: var(--cyan);
  opacity: 0.85;
  text-shadow: 0 0 12px rgba(0, 240, 255, 0.45);
}

.step-body h3 {
  margin-bottom: 4px;
}

.step-body p {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.75;
  color: var(--muted);
}

/* ---- 表格 ---- */
.table {
  margin-top: 14px;
  border: 1px solid var(--line-soft);
  border-radius: 8px;
  overflow: hidden;
}

.tr {
  display: grid;
  grid-template-columns: 1.4fr 1fr 1fr;
  align-items: center;
  gap: 10px;
  padding: 11px 16px;
  font-size: 13px;
  border-bottom: 1px solid var(--line-soft);
}

.tr:last-child {
  border-bottom: none;
}

.tr.th {
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--cyan);
  background: rgba(0, 240, 255, 0.05);
}

.td-label small {
  margin-left: 8px;
  font-size: 9px;
  letter-spacing: 1.5px;
  color: var(--muted);
}

.note {
  margin: 14px 0 0;
  font-size: 12px;
  line-height: 1.8;
  color: var(--muted);
}

/* ---- 运行环境 ---- */
.libs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
}

.lib {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 12px 14px;
  border-radius: 8px;
  border: 1px solid var(--line-soft);
  background: rgba(5, 10, 18, 0.5);
}

.lib-name {
  color: var(--text-strong);
  letter-spacing: 1px;
  font-size: 13px;
}

.lib-note {
  font-size: 11.5px;
  color: var(--muted);
}
</style>
