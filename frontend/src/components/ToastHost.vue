<script setup>
import { useToastStore } from '@/stores/toast'

const toast = useToastStore()

const ICONS = { success: 'OK', error: '!', info: 'i' }
</script>

<template>
  <Teleport to="body">
    <div class="toast-host">
      <TransitionGroup name="toast">
        <div
          v-for="item in toast.items"
          :key="item.id"
          class="toast"
          :class="item.type"
          @click="toast.remove(item.id)"
        >
          <span class="icon">{{ ICONS[item.type] || 'i' }}</span>
          <span class="msg">{{ item.message }}</span>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<style scoped>
.toast-host {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 3000;
  display: flex;
  flex-direction: column;
  gap: 10px;
  pointer-events: none;
}

.toast {
  pointer-events: auto;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 240px;
  max-width: 380px;
  padding: 11px 15px;
  border-radius: 8px;
  font-size: 13px;
  letter-spacing: 0.5px;
  color: var(--text-strong);
  background: rgba(8, 14, 24, 0.95);
  border: 1px solid var(--line);
  backdrop-filter: blur(8px);
  box-shadow:
    0 0 0 1px rgba(0, 240, 255, 0.05) inset,
    0 14px 34px rgba(0, 0, 0, 0.5);
}

.toast .icon {
  flex: 0 0 auto;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  color: #02121a;
  background: var(--cyan);
}

.toast.success {
  border-color: rgba(40, 224, 168, 0.45);
}
.toast.success .icon {
  background: var(--ok);
}

.toast.error {
  border-color: rgba(255, 77, 109, 0.5);
}
.toast.error .icon {
  background: var(--danger);
}

.toast.info .icon {
  background: var(--cyan);
}

.toast-enter-active,
.toast-leave-active {
  transition: all 0.26s ease;
}
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateX(24px);
}
</style>
