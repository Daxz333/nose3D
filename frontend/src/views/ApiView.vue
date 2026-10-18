<script setup>
import { computed, onMounted, ref } from 'vue'
import { getEndpoints } from '@/api'
import { useToastStore } from '@/stores/toast'
import RegionExamples from '@/components/RegionExamples.vue'
import regionExamples from '@/constants/regionExamples.json'

const toast = useToastStore()
const catalog = ref(null)
const loading = ref(false)
const error = ref('')
const selectedRegion = ref('nose')
const exampleAsset = (filename) => `${import.meta.env.BASE_URL}examples/regions/${filename}`
const baseURL = computed(() => new URL(import.meta.env.VITE_API_BASE || '/', window.location.origin).href.replace(/\/$/, ''))
const curlExample = computed(() => [
  `curl --max-time 300 -X POST "${baseURL.value}/predict"`,
  '  -F "face_image=@target.jpg"',
  '  -F "source_image=@source.jpg"',
  `  -F "region=${selectedRegion.value}"`,
].join(' \\\n'))

function responseExample(endpoint) {
  const example = { ...endpoint.response_example }
  if (endpoint.path === '/predict') {
    example.region = selectedRegion.value
    if (selectedRegion.value !== 'nose') delete example.nose_pose_corrected
  }
  return JSON.stringify(example, null, 2)
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const data = await getEndpoints()
    if (!Array.isArray(data.endpoints) || !Array.isArray(data.regions)) {
      throw new Error('接口列表格式无效，请确认后端已更新')
    }
    catalog.value = data
  } catch (reason) {
    error.value = reason.message || '接口列表加载失败'
  } finally {
    loading.value = false
  }
}

async function copy(text) {
  try {
    await navigator.clipboard.writeText(text)
    toast.success('已复制')
  } catch {
    toast.error('复制失败，请手动选择文本复制')
  }
}

onMounted(load)
</script>

<template>
  <div class="api-page">
    <header>
      <h1>接口列表</h1>
      <div class="subtitle"><span class="dot" />API REFERENCE · 无需注册或登录</div>
    </header>

    <section class="panel corner summary">
      <div>
        <h2>服务地址</h2>
        <code class="base-url">{{ baseURL }}</code>
        <p class="muted">接口列表由当前后端提供，工作台和接口调用均可直接使用。</p>
      </div>
      <button class="btn sm" type="button" :disabled="loading" @click="load">
        {{ loading ? '加载中…' : '刷新列表' }}
      </button>
    </section>

    <section v-if="error" class="panel error-state" role="alert">
      <p class="text-danger">{{ error }}</p>
      <p class="muted">请确认后端服务已启动，再点击刷新列表。</p>
    </section>
    <p v-else-if="loading && !catalog" class="muted" role="status">正在获取接口列表…</p>

    <template v-if="catalog && !error">
      <section v-for="endpoint in catalog.endpoints" :key="endpoint.path" class="panel corner endpoint">
        <div class="endpoint-head">
          <div class="endpoint-title">
            <span class="tag mono" :class="endpoint.method === 'POST' ? 'ok' : ''">{{ endpoint.method }}</span>
            <code>{{ endpoint.path }}</code>
            <h2>{{ endpoint.title }}</h2>
          </div>
          <button class="btn sm" type="button" @click="copy(`${baseURL}${endpoint.path}`)">复制地址</button>
        </div>
        <p class="muted">{{ endpoint.description }}</p>
        <p v-if="endpoint.content_type" class="content-type">Content-Type: <code>{{ endpoint.content_type }}</code></p>

        <div v-if="endpoint.parameters.length" class="parameters">
          <h3>请求参数</h3>
          <dl class="parameter-list">
            <div v-for="field in endpoint.parameters" :key="field.name" class="parameter">
              <dt><code>{{ field.name }}</code><span class="type">{{ field.type }}</span></dt>
              <dd>
                <span class="requirement" :class="{ required: field.required }">{{ field.required ? '必填' : '可选' }}</span>
                {{ field.description }}<span v-if="field.default"> 默认：<code>{{ field.default }}</code></span>
              </dd>
            </div>
          </dl>
        </div>
        <p v-else class="muted">无需请求参数。</p>

        <template v-if="endpoint.path === '/predict'">
          <h3>支持部位</h3>
          <div class="region-list">
            <button v-for="region in catalog.regions" :key="region.value" class="region-option" type="button"
              :class="{ active: selectedRegion === region.value }" :aria-pressed="selectedRegion === region.value"
              @click="selectedRegion = region.value">
              <img v-if="regionExamples[region.value]" :src="exampleAsset(regionExamples[region.value].views.result.detail)"
                :alt="`${region.label}局部示例`" loading="lazy" width="38" height="38" />
              <span>{{ region.label }} <code>{{ region.value }}</code></span>
            </button>
          </div>
          <p class="muted region-help">点击部位，查看对应图片示例并切换调用参数。</p>
          <RegionExamples :region="selectedRegion" />
          <div class="code-title"><h3>调用示例</h3><button class="btn sm" type="button" @click="copy(curlExample)">复制 cURL</button></div>
          <pre><code>{{ curlExample }}</code></pre>
          <p class="muted">将 target.jpg 和 source.jpg 换为实际文件路径；点击部位可切换示例参数。</p>
          <p class="muted">示例使用 Bash 换行语法；在 PowerShell 中请合并为一行并使用 curl.exe。</p>
        </template>
        <div class="code-title"><h3>响应示例</h3></div>
        <pre><code>{{ responseExample(endpoint) }}</code></pre>
        <ul v-if="endpoint.notes.length" class="notes"><li v-for="note in endpoint.notes" :key="note">{{ note }}</li></ul>
      </section>
    </template>
  </div>
</template>

<style scoped>
.api-page { display: flex; flex-direction: column; gap: 20px; }
.subtitle { margin-bottom: 0; }
.summary, .endpoint-head, .endpoint-title, .code-title { display: flex; align-items: center; gap: 14px; }
.summary, .endpoint-head, .code-title { justify-content: space-between; }
.base-url { display: block; margin: 12px 0 8px; color: var(--cyan); overflow-wrap: anywhere; }
.summary p { margin: 0; }
.summary .btn, .endpoint-head .btn { flex-shrink: 0; }
.endpoint-title { flex-wrap: wrap; }
.endpoint-title h2 { font-size: 16px; }
.endpoint-head { margin-bottom: 16px; }
code { font-family: var(--font-mono); color: var(--text-strong); font-size: 13px; }
.content-type { font-size: 12px; color: var(--muted); }
.parameters, .code-title { margin-top: 24px; }
.parameter-list { margin: 12px 0 24px; }
.parameter { display: grid; grid-template-columns: 190px 1fr; gap: 16px; padding: 14px 0; border-bottom: 1px solid var(--line-soft); }
.parameter dt { display: flex; align-items: center; gap: 12px; }
.parameter dd { margin: 0; color: var(--muted); font-size: 13px; }
.type { font-size: 11px; color: var(--muted); }
.requirement { margin-right: 8px; font-size: 11px; }
.requirement.required { color: var(--ok); }
.region-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.region-option { display: flex; align-items: center; gap: 8px; border: 1px solid var(--line-soft); border-radius: 6px; background: var(--bg-2); color: var(--text); padding: 7px 10px; cursor: pointer; font: inherit; font-size: 12px; }
.region-option img { border-radius: 4px; object-fit: cover; }
.region-help { margin: 10px 0 0; font-size: 12px; }
.region-option code { color: var(--muted); font-size: 11px; }
.region-option.active { border-color: var(--cyan); background: rgba(0, 240, 255, .08); }
.region-option.active code { color: var(--cyan); }
pre { padding: 16px; margin: 12px 0; background: var(--bg); border: 1px solid var(--line-soft); border-radius: 8px; overflow-x: auto; line-height: 1.8; }
.notes { padding-left: 18px; margin-bottom: 0; color: var(--muted); font-size: 13px; }
.notes li + li { margin-top: 6px; }
.error-state { border-color: var(--danger); }
@media (max-width: 720px) {
  .summary, .endpoint-head { align-items: flex-start; flex-direction: column; }
  .parameter { grid-template-columns: 1fr; gap: 6px; }
}
</style>
