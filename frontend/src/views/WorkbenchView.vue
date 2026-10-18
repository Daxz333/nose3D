<script setup>
import { computed, defineAsyncComponent, onBeforeUnmount, ref, watch } from 'vue'

import UploadSlot from '@/components/UploadSlot.vue'
import RegionSelector from '@/components/RegionSelector.vue'
import CompareSlider from '@/components/CompareSlider.vue'
import BaseModal from '@/components/BaseModal.vue'
import ProgressPipeline from '@/components/ProgressPipeline.vue'

import { predict } from '@/api'
import { useToastStore } from '@/stores/toast'
import { useHistoryStore } from '@/stores/history'
import { getRegion } from '@/constants/regions'
import { DEFAULT_ESTIMATED_SECONDS, PROCESS_STAGES } from '@/constants/pipeline'
import {
  base64ToDataURL,
  compressDataURL,
  downloadDataURL,
  fileToDataURL,
  timestampedName,
} from '@/utils/image'

const toast = useToastStore()
const HeadPanel = defineAsyncComponent(() => import('@/components/HeadPanel.vue'))
const history = useHistoryStore()

/* ---------------- 输入状态 ---------------- */
const faceFile = ref(null)
const noseFile = ref(null)
const facePreview = ref('')
const nosePreview = ref('')
const region = ref('nose')

/* ---------------- 运行状态 ---------------- */
const processing = ref(false)
const finished = ref(false)
const progress = ref(0)
const elapsedMs = ref(0)
const errorMsg = ref('')
/** 本次处理的真实耗时（秒），由后端请求返回后写入 */
const elapsed = ref(null)

let tickTimer = null

/* ---------------- 输出状态 ---------------- */
const resultFull = ref('')
const resultPreview = ref('')
const viewerOpen = ref(false)

const currentRegion = computed(() => getRegion(region.value))
const canProcess = computed(() => Boolean(faceFile.value && noseFile.value) && !processing.value)
const hasResult = computed(() => Boolean(resultFull.value))

watch(region, () => {
  resultFull.value = ''
  resultPreview.value = ''
  errorMsg.value = ''
  elapsed.value = null
  viewerOpen.value = false
})

/* ---------------- 文件选择 ---------------- */
async function selectFace(file) {
  faceFile.value = file
  facePreview.value = await fileToDataURL(file)
}

async function selectNose(file) {
  noseFile.value = file
  nosePreview.value = await fileToDataURL(file)
}

function clearFace() {
  faceFile.value = null
  facePreview.value = ''
}

function clearNose() {
  noseFile.value = null
  nosePreview.value = ''
}

/* ---------------- 进度推进 ---------------- */
/**
 * 预估总耗时（秒）：优先取同部位的历史平均耗时，无样本时用默认值。
 * 后端 /predict 为同步接口、不返回真实进度，因此以该值作为进度条的推进基准。
 */
const estimatedSeconds = computed(() => {
  const samples = history.records
    .filter((item) => item.region === region.value && Number(item.elapsed) > 0)
    .map((item) => Number(item.elapsed))
  if (!samples.length) return DEFAULT_ESTIMATED_SECONDS

  const average = samples.reduce((sum, value) => sum + value, 0) / samples.length
  return Math.min(180, Math.max(8, average * 1.15))
})

const elapsedSeconds = computed(() => elapsedMs.value / 1000)

/** 当前阶段下标：由进度反推，保证进度条与阶段清单始终一致 */
const activeIndex = computed(() => {
  let acc = 0
  for (let i = 0; i < PROCESS_STAGES.length; i += 1) {
    acc += PROCESS_STAGES[i].weight
    if (progress.value < acc) return i
  }
  return PROCESS_STAGES.length - 1
})

function startProgress() {
  stopProgress()
  const totalMs = estimatedSeconds.value * 1000
  const startedAt = performance.now()
  elapsedMs.value = 0
  progress.value = 0

  tickTimer = window.setInterval(() => {
    const ms = performance.now() - startedAt
    elapsedMs.value = ms
    // 渐近曲线：到达预估耗时时约 89%，之后缓慢逼近 97%，避免提前跑满
    const ratio = 1 - Math.exp(-ms / (totalMs * 0.4))
    progress.value = Math.min(ratio * 97, 97)
  }, 100)
}

function stopProgress() {
  if (tickTimer) {
    clearInterval(tickTimer)
    tickTimer = null
  }
}

function describeError(raw) {
  if (!raw) return '处理失败'
  if (raw.startsWith('Please upload both face')) return '图片上传失败，请重新选择'
  if (raw.startsWith('不支持的部位')) return raw
  return raw
}

/** 压缩后写入本地历史（避免 localStorage 溢出），失败不影响主流程 */
async function saveToHistory(seconds) {
  if (!resultFull.value) return
  try {
    const [faceThumb, noseThumb, resultThumb] = await Promise.all([
      compressDataURL(facePreview.value, 620, 0.76),
      compressDataURL(nosePreview.value, 620, 0.76),
      compressDataURL(resultFull.value, 900, 0.82),
    ])

    resultPreview.value = resultThumb
    history.add({
      region: region.value,
      regionLabel: currentRegion.value.label,
      face: faceThumb,
      nose: noseThumb,
      result: resultThumb,
      elapsed: seconds,
    })
  } catch {
    // 忽略历史写入异常
  }
}

/* ---------------- 主流程 ---------------- */
async function process() {
  if (!faceFile.value || !noseFile.value) {
    errorMsg.value = '请同时选择「人脸图片」与「部位来源图片」'
    toast.error(errorMsg.value)
    return
  }

  errorMsg.value = ''
  resultFull.value = ''
  resultPreview.value = ''
  elapsed.value = null
  finished.value = false
  processing.value = true
  startProgress()

  const requestStart = performance.now()
  let succeeded = false

  try {
    const data = await predict({
      faceFile: faceFile.value,
      noseFile: noseFile.value,
      region: region.value,
    })

    if (!data?.success || !data?.result) {
      errorMsg.value = describeError(data?.error)
      toast.error(errorMsg.value)
      return
    }

    succeeded = true
    const seconds = Number(((performance.now() - requestStart) / 1000).toFixed(1))
    elapsed.value = seconds

    // 先让进度条走满并展示完成态，再收起面板，避免结果突然跳出
    stopProgress()
    elapsedMs.value = seconds * 1000
    progress.value = 100
    finished.value = true
    await new Promise((resolve) => setTimeout(resolve, 520))

    processing.value = false
    resultFull.value = base64ToDataURL(data.result)
    toast.success(`处理完成，耗时 ${seconds}s`)

    await saveToHistory(seconds)
  } catch (error) {
    errorMsg.value = error.message || '请求失败'
    toast.error(errorMsg.value)
  } finally {
    stopProgress()
    if (!succeeded) processing.value = false
  }
}

function resetAll() {
  if (processing.value) return
  clearFace()
  clearNose()
  resultFull.value = ''
  resultPreview.value = ''
  errorMsg.value = ''
  elapsed.value = null
  progress.value = 0
  elapsedMs.value = 0
  finished.value = false
}

function download() {
  if (!resultFull.value) return
  downloadDataURL(resultFull.value, timestampedName(`${region.value}`))
  toast.info('结果已开始下载')
}

onBeforeUnmount(stopProgress)
</script>

<template>
  <div class="workbench">
    <header class="head">
      <h1>AI 换鼻服务系统</h1>
      <div class="subtitle">
        <span class="dot" />
        NEURAL NOSE TRANSFER · SYSTEM ONLINE
      </div>
    </header>

    <!-- 输入模块 -->
    <div class="divider">
      <span class="divider-line" />
      <span class="divider-text">INPUT MODULE</span>
      <span class="divider-line" />
    </div>

    <section class="panel corner">
      <div class="grid-2">
        <UploadSlot
          title="人脸图片（目标）"
          tag="FACE / TARGET"
          :hint="currentRegion.targetHint"
          :preview="facePreview"
          :file="faceFile"
          :disabled="processing"
          @select="selectFace"
          @clear="clearFace"
        />
        <UploadSlot
          title="来源图片（供体）"
          tag="SOURCE / DONOR"
          :hint="currentRegion.sourceHint"
          :preview="nosePreview"
          :file="noseFile"
          :disabled="processing"
          @select="selectNose"
          @clear="clearNose"
        />
      </div>

      <div class="region-block">
        <div class="region-head">
          <span class="region-title">替换部位</span>
          <span class="tag" :class="{ warn: !currentRegion.supported }">
            {{ currentRegion.en }} · 类别 {{ currentRegion.classIds.join('+') }}
          </span>
        </div>
        <RegionSelector v-model="region" :disabled="processing" />
        <p v-if="currentRegion.tip" class="region-tip">
          {{ currentRegion.tip }}
        </p>
      </div>

      <div class="actions">
        <button class="btn" type="button" :disabled="!canProcess" @click="process">
          {{ processing ? '处理中…' : '开始处理' }}
        </button>
        <button class="btn ghost" type="button" :disabled="processing || (!faceFile && !noseFile && !hasResult)" @click="resetAll">
          重置
        </button>
        <span class="actions-hint mono">GPU · CUDA 11.8 · GFPGAN x2</span>
      </div>

      <p v-if="errorMsg" class="error-line">ERROR · {{ errorMsg }}</p>
    </section>

    <!-- 处理中：按运行逻辑顺序展示流水线进度 -->
    <Transition name="fade">
      <ProgressPipeline
        v-if="processing"
        :progress="progress"
        :stages="PROCESS_STAGES"
        :active-index="activeIndex"
        :elapsed="elapsedSeconds"
        :estimated="estimatedSeconds"
        :finished="finished"
      />
    </Transition>

    <!-- 结果模块 -->
    <template v-if="hasResult">
      <div class="divider">
        <span class="divider-line" />
        <span class="divider-text">OUTPUT MODULE</span>
        <span class="divider-line" />
      </div>

      <section class="panel corner fade-up">
        <div class="result-head">
          <div>
            <h2>处理结果</h2>
            <p class="muted mono">
              {{ currentRegion.en }} TRANSFER COMPLETE<template v-if="elapsed"> · {{ elapsed }}s</template>
            </p>
          </div>
          <div class="row">
            <button class="btn subtle" type="button" @click="viewerOpen = true">查看大图</button>
            <button class="btn subtle" type="button" @click="download">下载结果</button>
            <button class="btn" type="button" @click="process">重新处理</button>
          </div>
        </div>

        <CompareSlider
          :before="facePreview"
          :after="resultFull"
          before-label="ORIGINAL"
          after-label="RESULT"
        />
      </section>
    </template>

    <HeadPanel v-if="hasResult" :result-image="resultFull" />

    <BaseModal v-model="viewerOpen" title="结果大图" en="RESULT PREVIEW" width="960px">
      <img v-if="resultFull" class="viewer-img" :src="resultFull" alt="结果大图" />
      <template #footer>
        <button class="btn ghost" type="button" @click="viewerOpen = false">关闭</button>
        <button class="btn" type="button" @click="download">下载结果</button>
      </template>
    </BaseModal>
  </div>
</template>

<style scoped>
.workbench {
  display: block;
}

.head {
  margin-bottom: 4px;
}

/* ---- 部位区 ---- */
.region-block {
  margin-top: 22px;
  padding-top: 20px;
  border-top: 1px solid var(--line-soft);
}

.region-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.region-title {
  font-size: 14px;
  color: var(--text-strong);
  letter-spacing: 1px;
}

.region-tip {
  margin: 12px 0 0;
  font-size: 12px;
  line-height: 1.7;
  color: var(--warn);
  opacity: 0.85;
}

/* ---- 操作区 ---- */
.actions {
  margin-top: 22px;
  padding-top: 20px;
  border-top: 1px solid var(--line-soft);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.actions-hint {
  margin-left: auto;
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--muted);
}

.error-line {
  margin: 16px 0 0;
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 12.5px;
  letter-spacing: 0.5px;
  color: var(--danger);
  background: rgba(255, 77, 109, 0.08);
  border: 1px solid rgba(255, 77, 109, 0.28);
}

/* ---- 结果 ---- */
.result-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 18px;
}

.result-head p {
  margin: 3px 0 0;
  font-size: 11px;
  letter-spacing: 2px;
}

.viewer-img {
  width: 100%;
  max-height: 68vh;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid var(--line-soft);
  background: var(--panel-2);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.25s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 720px) {
  .actions-hint {
    display: none;
  }
}
</style>
