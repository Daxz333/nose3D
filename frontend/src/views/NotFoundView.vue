<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()
</script>

<template>
  <div class="notfound">
    <div class="code mono">404</div>
    <h1>页面不存在</h1>
    <p class="muted">请求的路径未在系统中注册，可能链接已失效。</p>
    <div class="row">
      <button class="btn ghost" type="button" @click="router.back()">返回上一页</button>
      <button class="btn" type="button" @click="router.replace({ name: 'workbench' })">回到工作台</button>
    </div>
  </div>
</template>

<style scoped>
.notfound {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  text-align: center;
  padding: 24px;
}

.code {
  font-size: 76px;
  line-height: 1;
  color: var(--cyan);
  letter-spacing: 8px;
  text-shadow: 0 0 34px rgba(0, 240, 255, 0.55);
}

.notfound p {
  margin: 0 0 12px;
}
</style>
