<script setup>
import { computed } from 'vue'
import { useHistoryStore } from '@/stores/history'

const history = useHistoryStore()

const NAV = [
  { name: 'workbench', label: '换鼻工作台', en: 'WORKBENCH', icon: '◈' },
  { name: 'head', label: '3D 头部模型', en: 'HEAD MODEL', icon: '◉' },
  { name: 'history', label: '历史记录', en: 'HISTORY', icon: '▤' },
  { name: 'guide', label: '算法说明', en: 'PIPELINE', icon: '⌁' },
  { name: 'api', label: '接口列表', en: 'API REFERENCE', icon: '⇄' },
  { name: 'about', label: '关于系统', en: 'ABOUT', icon: '◎' },
]

const historyCount = computed(() => history.count)

</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <span class="brand-title">NOSE·AI</span>
      <span class="brand-sub mono">v1.0 / GPU</span>
    </div>

    <nav class="nav">
      <router-link
        v-for="item in NAV"
        :key="item.name"
        class="nav-item"
        :to="{ name: item.name }"
        :title="item.label"
        :aria-label="item.label"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span class="nav-body">
          <span class="nav-label">{{ item.label }}</span>
          <span class="nav-en mono">{{ item.en }}</span>
        </span>
        <span v-if="item.name === 'history' && historyCount" class="nav-count mono">
          {{ historyCount }}
        </span>
      </router-link>
    </nav>

    <div class="bottom">
      <div class="status">
        <span class="dot" />
        <span class="mono">SYSTEM ONLINE</span>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: var(--sidebar-w);
  flex: 0 0 var(--sidebar-w);
  display: flex;
  flex-direction: column;
  background: linear-gradient(180deg, rgba(8, 14, 24, 0.96), rgba(6, 10, 18, 0.96));
  border-right: 1px solid var(--line-soft);
  position: sticky;
  top: 0;
  height: 100vh;
  padding: 22px 0 18px;
}

.brand {
  padding: 0 20px 18px;
  border-bottom: 1px solid var(--line-soft);
  margin-bottom: 14px;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
}

.brand-title {
  font-family: var(--font-mono);
  font-size: 15px;
  letter-spacing: 3px;
  color: var(--cyan);
  text-shadow: 0 0 12px rgba(0, 240, 255, 0.6);
}

.brand-sub {
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 1px;
}

.nav {
  flex: 1;
  overflow-y: auto;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 20px;
  margin: 3px 0;
  color: var(--muted);
  text-decoration: none;
  font-size: 13.5px;
  border-left: 2px solid transparent;
  transition: 0.25s;
}

.nav-item:hover {
  color: var(--cyan);
  border-left-color: rgba(0, 240, 255, 0.5);
  background: linear-gradient(90deg, rgba(0, 240, 255, 0.08), transparent);
}

.nav-item.router-link-active {
  color: var(--cyan);
  border-left-color: var(--cyan);
  background: linear-gradient(90deg, rgba(0, 240, 255, 0.12), transparent);
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
}

.nav-icon {
  font-size: 13px;
  opacity: 0.85;
}

.nav-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  line-height: 1.25;
}

.nav-label {
  letter-spacing: 1px;
}

.nav-en {
  font-size: 9px;
  letter-spacing: 1.5px;
  opacity: 0.55;
}

.nav-count {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 8px;
  color: var(--cyan);
  border: 1px solid var(--line);
  background: rgba(0, 240, 255, 0.08);
}

.bottom {
  padding: 14px 20px 0;
  border-top: 1px solid var(--line-soft);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 2px;
}

@media (max-width: 820px) {
  .sidebar {
    width: 62px;
    flex-basis: 62px;
  }
  .brand-sub,
  .nav-body,
  .nav-count,
  .status {
    display: none;
  }
  .brand {
    justify-content: center;
    padding: 0 0 16px;
  }
  .brand-title {
    font-size: 10px;
    letter-spacing: 1px;
  }
  .nav-item {
    justify-content: center;
    padding: 13px 0;
  }
  .bottom {
    padding: 12px 8px 0;
  }
}
</style>
