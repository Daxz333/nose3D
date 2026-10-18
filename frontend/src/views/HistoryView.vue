<script setup>
import { computed, ref } from 'vue'

import BaseModal from '@/components/BaseModal.vue'
import CompareSlider from '@/components/CompareSlider.vue'

import { useHistoryStore } from '@/stores/history'
import { useToastStore } from '@/stores/toast'
import { getRegion } from '@/constants/regions'
import { downloadDataURL, formatTime, timestampedName } from '@/utils/image'

const history = useHistoryStore()
const toast = useToastStore()

const active = ref(null)

const records = computed(() => history.records)

function open(record) {
  active.value = record
}

function download(record) {
  downloadDataURL(record.result, timestampedName(`${record.region}`))
  toast.info('结果已开始下载')
}

function remove(record) {
  history.remove(record.id)
  if (active.value?.id === record.id) active.value = null
  toast.info('已删除该条记录')
}

function clearAll() {
  if (!records.value.length) return
  if (!window.confirm(`确认清空全部 ${records.value.length} 条本地历史记录？此操作不可恢复。`)) return
  history.clear()
  active.value = null
  toast.info('历史记录已清空')
}

const activeRegion = computed(() => (active.value ? getRegion(active.value.region) : null))
</script>

<template>
  <div>
    <header class="head">
      <div>
        <h1>历史记录</h1>
        <div class="subtitle">
          <span class="dot" />
          LOCAL RECORDS · 保存在当前浏览器
        </div>
      </div>
      <button class="btn danger sm" type="button" :disabled="!records.length" @click="clearAll">
        清空全部
      </button>
    </header>

    <div v-if="!records.length" class="panel corner empty">
      <span class="empty-mark mono">NO DATA</span>
      <p>暂无处理记录，前往工作台完成第一次换鼻后会自动保存在这里。</p>
      <router-link class="btn ghost" :to="{ name: 'workbench' }">前往工作台</router-link>
    </div>

    <div v-else class="grid">
      <article v-for="item in records" :key="item.id" class="card panel">
        <div class="thumb" @click="open(item)">
          <img :src="item.result" alt="处理结果" />
          <span class="thumb-tag mono">{{ getRegion(item.region).en }}</span>
        </div>

        <div class="card-body">
          <div class="card-row">
            <span class="region-name">{{ item.regionLabel || getRegion(item.region).label }}</span>
            <span v-if="item.elapsed" class="tag mono">{{ item.elapsed }}s</span>
          </div>
          <span class="time mono">{{ formatTime(item.createdAt) }}</span>

          <div class="card-actions">
            <button class="btn subtle sm" type="button" @click="open(item)">对比</button>
            <button class="btn subtle sm" type="button" @click="download(item)">下载</button>
            <button class="btn danger sm" type="button" @click="remove(item)">删除</button>
          </div>
        </div>
      </article>
    </div>

    <BaseModal
      :model-value="!!active"
      :title="active ? `${active.regionLabel || getRegion(active.region).label} · 处理对比` : ''"
      en="RECORD COMPARE"
      width="960px"
      @update:model-value="active = null"
    >
      <template v-if="active">
        <CompareSlider :before="active.face" :after="active.result" before-label="ORIGINAL" after-label="RESULT" />
        <div class="detail-row mono">
          <span>SOURCE · 来源图</span>
          <span>TIME · {{ formatTime(active.createdAt) }}</span>
        </div>
        <img class="source-img" :src="active.nose" alt="来源图" />
      </template>

      <template #footer>
        <button class="btn ghost" type="button" @click="active = null">关闭</button>
        <button v-if="active" class="btn" type="button" @click="download(active)">下载结果</button>
      </template>
    </BaseModal>
  </div>
</template>

<style scoped>
.head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 22px;
}

.empty {
  text-align: center;
  padding: 60px 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.empty-mark {
  font-size: 13px;
  letter-spacing: 6px;
  color: var(--muted);
}

.empty p {
  color: var(--muted);
  max-width: 420px;
  margin: 0;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(258px, 1fr));
  gap: 18px;
}

.card {
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  transition: 0.25s;
}

.card:hover {
  border-color: var(--line);
  box-shadow:
    0 0 0 1px rgba(0, 240, 255, 0.05) inset,
    0 18px 40px rgba(0, 0, 0, 0.5),
    0 0 26px rgba(0, 240, 255, 0.1);
}

.thumb {
  position: relative;
  height: 190px;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  border: 1px solid var(--line-soft);
  background: var(--panel-2);
}

.thumb img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  transition: 0.3s;
}

.thumb:hover img {
  transform: scale(1.03);
}

.thumb-tag {
  position: absolute;
  top: 8px;
  left: 8px;
  font-size: 9px;
  letter-spacing: 2px;
  padding: 2px 7px;
  border-radius: 4px;
  color: var(--cyan);
  background: rgba(5, 10, 18, 0.85);
  border: 1px solid var(--line);
}

.card-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.card-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.region-name {
  font-size: 14px;
  color: var(--text-strong);
  letter-spacing: 1px;
}

.time {
  font-size: 10px;
  letter-spacing: 1px;
  color: var(--muted);
}

.card-actions {
  display: flex;
  gap: 8px;
  margin-top: 4px;
}

.card-actions .btn {
  flex: 1;
  padding: 0 6px;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  margin: 18px 0 8px;
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--muted);
}

.source-img {
  width: 200px;
  max-width: 100%;
  border-radius: 8px;
  border: 1px solid var(--line-soft);
  background: var(--panel-2);
  display: block;
}
</style>
