<script setup>
import { computed, ref, watch } from 'vue'
import BaseModal from '@/components/BaseModal.vue'
import examples from '@/constants/regionExamples.json'
import { getRegion } from '@/constants/regions'

const props = defineProps({ region: { type: String, required: true } })
const example = computed(() => examples[props.region])
const regionInfo = computed(() => getRegion(props.region))
const roles = [
  { key: 'target', label: '目标图片', detail: '替换前', download: '下载目标原图' },
  { key: 'source', label: '来源图片', detail: '供体部位', download: '下载来源原图' },
  { key: 'result', label: '替换结果', detail: '替换后', download: '下载示例结果' },
]
const viewerOpen = ref(false)
const viewer = ref(null)
const asset = (filename) => `${import.meta.env.BASE_URL}examples/regions/${filename}`

function boxStyle(box) {
  return {
    left: `${box[0] / 512 * 100}%`, top: `${box[1] / 512 * 100}%`,
    width: `${(box[2] - box[0]) / 512 * 100}%`, height: `${(box[3] - box[1]) / 512 * 100}%`,
  }
}

function inputFor(role) {
  return role === 'target' ? example.value.targetInput
    : role === 'source' ? example.value.sourceInput : example.value.views.result.image
}

function open(role, detail = false) {
  viewer.value = {
    title: `${example.value.label} · ${role.label}${detail ? '局部' : ''}`,
    image: asset(example.value.views[role.key][detail ? 'detail' : 'image']),
  }
  viewerOpen.value = true
}

watch(() => props.region, () => { viewerOpen.value = false })
</script>

<template>
  <section v-if="example" class="region-examples" :aria-label="`${example.label}图片示例`">
    <div class="example-heading">
      <h3>{{ example.label }} · 图片示例</h3>
      <span class="tag ok">真实替换结果</span>
    </div>
    <p class="muted example-description">{{ regionInfo.sourceHint }}。青色框标出本例对应部位；点击图片可放大。</p>

    <div class="example-grid">
      <figure v-for="role in roles" :key="role.key" class="example-image">
        <figcaption>{{ role.label }}</figcaption>
        <button class="image-button full-face" type="button" :aria-label="`放大查看${example.label}${role.label}`" @click="open(role)">
          <img :src="asset(example.views[role.key].image)" :alt="`${example.label}${role.label}示例`" loading="lazy" width="512" height="512" />
          <span class="region-box" :style="boxStyle(example.views[role.key].box)" aria-hidden="true" />
        </button>
        <a class="image-download" :href="asset(inputFor(role.key))" :download="role.key === 'result' ? `result-${region}.jpg` : `${role.key}.jpg`">{{ role.download }} ↓</a>
      </figure>
    </div>

    <div class="detail-heading"><h4>局部放大对比</h4><span class="muted">目标 / 来源 / 结果</span></div>
    <div class="example-grid detail-grid">
      <figure v-for="role in roles" :key="role.key" class="example-image">
        <button class="image-button" type="button" :aria-label="`放大查看${example.label}${role.detail}局部`" @click="open(role, true)">
          <img :src="asset(example.views[role.key].detail)" :alt="`${example.label}${role.detail}局部示例`" loading="lazy" />
        </button>
        <figcaption>{{ role.detail }}</figcaption>
      </figure>
    </div>
    <p v-if="regionInfo.tip" class="example-tip">{{ regionInfo.tip }}</p>
    <p class="muted example-note">示例通过实际替换和增强流程生成，效果随输入照片而变化。全脸预览统一裁剪，下载文件为实际输入原图；调用接口时需上传完整人脸照片。</p>
  </section>
  <p v-else class="muted">该部位暂未提供图片示例。</p>

  <BaseModal v-model="viewerOpen" :title="viewer?.title || '图片示例'" en="IMAGE EXAMPLE" width="680px">
    <img v-if="viewer" class="viewer-image" :src="viewer.image" :alt="viewer.title" />
  </BaseModal>
</template>

<style scoped>
.region-examples { margin-top: 24px; padding: 20px; border: 1px solid var(--line-soft); border-radius: 8px; background: rgba(0, 240, 255, .025); }
.example-heading, .detail-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.example-description { margin: 10px 0 16px; font-size: 12px; }
.example-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.example-image { margin: 0; min-width: 0; }
figcaption { font-size: 12px; color: var(--text); margin: 0 0 8px; text-align: center; }
.image-button { display: block; position: relative; width: 100%; padding: 0; border: 1px solid var(--line-soft); border-radius: 6px; overflow: hidden; background: var(--bg); cursor: zoom-in; }
.image-button:hover, .image-button:focus-visible { border-color: var(--cyan); }
.image-button img { display: block; width: 100%; height: auto; aspect-ratio: 1; object-fit: contain; }
.region-box { position: absolute; border: 2px solid var(--cyan); border-radius: 3px; box-shadow: 0 0 6px rgba(0, 240, 255, .45); pointer-events: none; }
.image-download { display: block; margin-top: 8px; font-size: 11px; text-align: center; }
.detail-heading { margin: 22px 0 12px; }
.detail-heading span { font-size: 11px; }
.detail-grid .image-button img { max-height: 190px; }
.detail-grid figcaption { margin: 7px 0 0; color: var(--muted); }
.example-tip { margin: 18px 0 8px; padding-left: 10px; border-left: 2px solid var(--cyan); color: var(--text); font-size: 12px; }
.example-note { margin: 16px 0 0; font-size: 11px; }
.viewer-image { display: block; max-width: 100%; max-height: 68vh; margin: auto; object-fit: contain; }
@media (max-width: 520px) {
  .region-examples { padding: 12px; }
  .example-grid { gap: 7px; }
  .image-download { font-size: 10px; }
}
</style>
