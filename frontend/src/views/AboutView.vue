<script setup>
import { computed } from 'vue'
import { useHistoryStore } from '@/stores/history'

const history = useHistoryStore()

const CONTACT = {
  phone: '17828056125',
  email: '1323049452@qq.com',
}

const notes = [
  '后端为 Flask 开发服务器 + 全局模型，GPU 推理不适合高并发，请避免同时提交多个任务。',
  '首次请求需加载检测 / 分割 / 超分模型，可能明显偏慢，属正常现象。',
  '侧脸、遮挡或光线过暗时可能出现「未检测到人脸或者目标部位」，建议更换清晰正脸照。',
  '历史记录仅保存在当前浏览器 localStorage，清理浏览器数据会一并丢失。',
]

const env = computed(() => [
  { k: '访问方式', v: '直接使用' },
  { k: '本地记录', v: `${history.count} 条` },
  { k: '前端', v: 'Vue 3 + Vite + Pinia' },
  { k: '后端', v: 'Flask 2.3 · 0.0.0.0:6006' },
])

function copy(text) {
  navigator.clipboard?.writeText(text)
}
</script>

<template>
  <div>
    <header>
      <h1>关于系统</h1>
      <div class="subtitle">
        <span class="dot" />
        AIHB · AI 换鼻服务系统 v1.0
      </div>
    </header>

    <div class="grid-2">
      <section class="panel corner">
        <h2>运行信息</h2>
        <div class="kv">
          <div v-for="item in env" :key="item.k" class="kv-row">
            <span class="kv-k">{{ item.k }}</span>
            <span class="kv-v mono">{{ item.v }}</span>
          </div>
        </div>
      </section>

      <section class="panel corner">
        <h2>联系客服</h2>
        <p class="muted">使用中遇到问题或需要批量处理，可通过以下方式联系。</p>
        <div class="contact">
          <button class="contact-item" type="button" @click="copy(CONTACT.phone)">
            <span class="ci-icon mono">TEL</span>
            <span class="ci-value">{{ CONTACT.phone }}</span>
            <span class="ci-copy">复制</span>
          </button>
          <button class="contact-item" type="button" @click="copy(CONTACT.email)">
            <span class="ci-icon mono">MAIL</span>
            <span class="ci-value">{{ CONTACT.email }}</span>
            <span class="ci-copy">复制</span>
          </button>
        </div>
      </section>
    </div>

    <div class="divider">
      <span class="divider-line" />
      <span class="divider-text">BACKEND API</span>
      <span class="divider-line" />
    </div>

    <section class="panel corner">
      <p class="muted">请求地址、参数和调用示例已集中到接口列表。</p>
      <router-link class="btn sm" :to="{ name: 'api' }">查看接口列表 →</router-link>
    </section>

    <div class="divider">
      <span class="divider-line" />
      <span class="divider-text">NOTES</span>
      <span class="divider-line" />
    </div>

    <section class="panel corner">
      <ul class="notes">
        <li v-for="n in notes" :key="n">{{ n }}</li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.kv {
  margin-top: 14px;
  display: flex;
  flex-direction: column;
}

.kv-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px dashed var(--line-soft);
}

.kv-row:last-child {
  border-bottom: none;
}

.kv-k {
  font-size: 12.5px;
  color: var(--muted);
  letter-spacing: 1px;
}

.kv-v {
  font-size: 12px;
  color: var(--text-strong);
  letter-spacing: 1px;
}

.contact {
  margin-top: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.contact-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 8px;
  border: 1px solid var(--line-soft);
  background: rgba(5, 10, 18, 0.55);
  color: var(--text);
  font-family: inherit;
  font-size: 13px;
  cursor: pointer;
  text-align: left;
  transition: 0.22s;
}

.contact-item:hover {
  border-color: var(--line);
  background: rgba(0, 240, 255, 0.07);
}

.ci-icon {
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--cyan);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 2px 6px;
}

.ci-value {
  flex: 1;
}

.ci-copy {
  font-size: 11px;
  color: var(--muted);
}

.notes {
  margin: 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.notes li {
  font-size: 13px;
  line-height: 1.8;
  color: var(--muted);
}

.notes li::marker {
  color: var(--cyan);
}

</style>
