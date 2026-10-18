<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const title = computed(() => route.meta?.title || 'AI 换鼻服务系统')
const en = computed(() => route.meta?.en || 'SYSTEM')
const now = computed(() =>
  new Date().toLocaleTimeString('zh-CN', { hour12: false })
)
</script>

<template>
  <header class="topbar">
    <div class="crumb">
      <span class="mono crumb-en">{{ en }}</span>
      <span class="crumb-sep">/</span>
      <span class="crumb-title">{{ title }}</span>
    </div>

    <div class="right">
      <span class="tag ok"><span class="dot" />{{ route.name === 'head' ? '本地头模工作台' : '推理服务就绪' }}</span>
      <span class="mono clock">{{ now }}</span>
    </div>
  </header>
</template>

<style scoped>
.topbar {
  height: var(--topbar-h);
  flex: 0 0 var(--topbar-h);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  border-bottom: 1px solid var(--line-soft);
  background: rgba(6, 10, 18, 0.65);
  backdrop-filter: blur(8px);
  position: sticky;
  top: 0;
  z-index: 20;
}

.crumb {
  display: flex;
  align-items: center;
  gap: 10px;
}

.crumb-en {
  font-size: 11px;
  letter-spacing: 3px;
  color: var(--cyan);
}

.crumb-sep {
  color: var(--muted);
}

.crumb-title {
  font-size: 14px;
  color: var(--text-strong);
  letter-spacing: 1px;
}

.right {
  display: flex;
  align-items: center;
  gap: 14px;
}

.right .dot {
  width: 6px;
  height: 6px;
  margin: 0;
  border-color: var(--ok);
  background: var(--ok);
  box-shadow: 0 0 8px var(--ok);
}

.clock {
  font-size: 12px;
  color: var(--muted);
  letter-spacing: 2px;
}

@media (max-width: 720px) {
  .topbar {
    padding: 0 16px;
  }
  .right .clock {
    display: none;
  }
}
</style>
