<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import HeadViewer from '@/components/HeadViewer.vue'
import UploadSlot from '@/components/UploadSlot.vue'
import { getHeadStatus, reconstructHead } from '@/api'
import { fileToDataURL } from '@/utils/image'

const props = defineProps({ resultImage: { type: String, default: '' } })
const selected = ref(null), preview = ref(''), ready = ref(false), checking = ref(true)
const busy = ref(false), error = ref(''), output = ref(null), templateUrl = ref('')
const showingTemplate = ref(false)
const height = ref(200)
const modelUrl = computed(() => showingTemplate.value || !output.value ? templateUrl.value : output.value.model_url)
let controller, selection = 0, disposed = false

async function check() {
  checking.value = true
  error.value = ''
  try {
    const status = await getHeadStatus()
    if (disposed) return
    ready.value = status.ready
    templateUrl.value = status.template_available ? status.template_url : ''
    if (!status.ready) error.value = status.message
  } catch (e) {
    if (!disposed) error.value = e.message
  } finally { if (!disposed) checking.value = false }
}

function invalidate() {
  selection += 1
  controller?.abort()
  busy.value = false
  output.value = null
  showingTemplate.value = false
  error.value = ''
}

async function select(file) {
  invalidate()
  const token = selection
  selected.value = null
  preview.value = ''
  if (file.size > 12 * 1024 * 1024) { error.value = '请上传 12 MB 以内的图片'; return }
  try {
    const data = await fileToDataURL(file)
    if (disposed || token !== selection) return
    selected.value = file
    preview.value = data
  } catch { if (token === selection) error.value = '无法读取图片，请重新选择' }
}

function clear() { invalidate(); selected.value = null; preview.value = '' }

async function useResult() {
  if (!props.resultImage) return
  const blob = await (await fetch(props.resultImage)).blob()
  if (!disposed) await select(new File([blob], '换鼻结果.jpg', { type: blob.type }))
}

async function generate() {
  if (!selected.value || busy.value || !ready.value) return
  error.value = ''
  output.value = null
  showingTemplate.value = false
  busy.value = true
  const token = selection
  controller = new AbortController()
  try {
    const result = await reconstructHead(selected.value, controller.signal, height.value)
    if (disposed || token !== selection) return
    if (!result.success) throw new Error(result.error || '重建失败')
    output.value = result
  } catch (e) {
    if (!disposed && token === selection) error.value = e.message
  } finally { if (!disposed && token === selection) busy.value = false }
}

watch(() => props.resultImage, () => { clear(); if (props.resultImage) useResult() })
onMounted(() => { check(); if (props.resultImage) useResult() })
onBeforeUnmount(() => { disposed = true; selection += 1; controller?.abort() })
</script>

<template>
  <section class="panel corner head-panel">
    <div class="panel-heading">
      <div>
        <h2>稠密人脸 · 打印头模</h2>
        <p class="muted">从照片重建五官几何，融合到统一头模。支持 15–25 厘米高度和 STL 导出。</p>
      </div>
      <span class="tag">{{ checking ? '检查环境…' : ready ? 'HRN 稠密重建' : '环境未就绪' }}</span>
    </div>
    <div class="head-grid">
      <div>
        <UploadSlot title="重建用人脸照片" tag="SINGLE IMAGE" hint="一张清晰正脸，额头与下巴完整可见；JPG / PNG / WebP，12 MB 内"
          :preview="preview" :file="selected" :disabled="busy" @select="select" @clear="clear" />
        <label class="size-field">模型总高度（含颈部）
          <select v-model.number="height" :disabled="busy" @change="output = null">
            <option :value="150">150 mm</option><option :value="200">200 mm</option><option :value="250">250 mm</option>
          </select>
        </label>
        <div class="buttons">
          <button class="btn" type="button" :disabled="!selected || !ready || busy" @click="generate">
            {{ busy ? '正在重建…' : '生成稠密头模' }}
          </button>
          <button v-if="resultImage" class="btn subtle" type="button" :disabled="busy" @click="useResult">使用换鼻结果</button>
          <button v-if="!ready" class="btn subtle" type="button" :disabled="checking" @click="check">重新检查</button>
        </div>
        <p v-if="busy" class="muted" role="status">正在重建五官、融合头模并检查封闭网格，请稍候。</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
        <p class="note">预览展示实际几何。后脑、耳朵和颈部共用固定底模。单图深度仍是估计值；导出 STL 后请在切片软件中检查尺寸、细节和支撑。</p>
        <div v-if="output" class="output">
          <p>已生成 · {{ output.metadata.faces.toLocaleString() }} 个三角面 · {{ output.metadata.height_mm }} mm</p>
          <p>单一封闭实体 · 边界边 {{ output.metadata.topology.boundary_edges }} · 非流形边 {{ output.metadata.topology.nonmanifold_edges }}</p>
          <p>自交三角形 {{ output.metadata.topology.self_intersection_triangles }} · STL 按毫米导入切片软件</p>
          <div class="buttons">
            <a class="btn" :href="output.stl_url" download="head.stl">下载 STL</a>
            <a class="btn subtle" :href="output.download_url" download="head.glb">下载 GLB</a>
            <a class="btn subtle" :href="output.report_url" download="head-report.json">检查报告</a>
            <button class="btn subtle" type="button" @click="showingTemplate = !showingTemplate">
              {{ showingTemplate ? '查看重建结果' : '对比原始底模' }}
            </button>
          </div>
        </div>
      </div>
      <div>
        <p class="viewer-title">{{ !output || showingTemplate ? '统一头部底模' : '面部重建结果' }}</p>
        <HeadViewer :url="modelUrl" />
      </div>
    </div>
  </section>
</template>

<style scoped>
.head-panel { margin-top: 26px; }
.panel-heading { display: flex; justify-content: space-between; gap: 16px; margin-bottom: 22px; align-items: start; }
.panel-heading p { font-size: 13px; line-height: 1.8; max-width: 650px; }
.head-grid { display: grid; grid-template-columns: minmax(230px, .8fr) minmax(320px, 1.2fr); gap: 26px; }
.buttons { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.size-field { display: flex; justify-content: space-between; align-items: center; margin-top: 18px; gap: 10px; font-size: 13px; }
.size-field select { color: var(--text-strong); background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 6px; padding: 8px 12px; }
.note { font-size: 12px; line-height: 1.9; color: var(--muted); margin-top: 20px; }
.error { color: var(--danger); font-size: 13px; line-height: 1.8; }
.viewer-title { font-size: 13px; margin: 0 0 12px; color: var(--text-strong); }
.output { border-top: 1px solid var(--line-soft); margin-top: 20px; padding-top: 10px; }
.output p { font-size: 13px; color: var(--cyan); }
@media (max-width: 960px) { .head-grid { grid-template-columns: 1fr; } }
</style>
