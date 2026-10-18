<script setup>
import { onBeforeUnmount, watch } from 'vue'

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  title: { type: String, default: '' },
  en: { type: String, default: '' },
  width: { type: String, default: '860px' },
})

const emit = defineEmits(['update:modelValue'])

function close() {
  emit('update:modelValue', false)
}

function onKeydown(event) {
  if (event.key === 'Escape') close()
}

watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      window.addEventListener('keydown', onKeydown)
      document.body.style.overflow = 'hidden'
    } else {
      window.removeEventListener('keydown', onKeydown)
      document.body.style.overflow = ''
    }
  }
)

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  document.body.style.overflow = ''
})
</script>

<template>
  <Teleport to="body">
    <Transition name="modal">
      <div v-if="modelValue" class="mask" @click.self="close">
        <div class="modal panel corner" :style="{ maxWidth: width }">
          <header class="modal-head">
            <div class="row">
              <span v-if="en" class="mono head-en">{{ en }}</span>
              <h2>{{ title }}</h2>
            </div>
            <button class="close" type="button" @click="close">✕</button>
          </header>

          <div class="modal-body">
            <slot />
          </div>

          <footer v-if="$slots.footer" class="modal-foot">
            <slot name="footer" />
          </footer>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.mask {
  position: fixed;
  inset: 0;
  z-index: 2000;
  background: rgba(2, 5, 10, 0.78);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 26px;
}

.modal {
  width: 100%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  background: rgba(8, 14, 24, 0.97);
  border-color: var(--line);
  box-shadow:
    0 0 0 1px rgba(0, 240, 255, 0.06) inset,
    0 28px 70px rgba(0, 0, 0, 0.6);
  animation: fadeUp 0.25s ease both;
}

.modal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--line-soft);
}

.head-en {
  font-size: 10px;
  letter-spacing: 3px;
  color: var(--cyan);
}

.close {
  width: 30px;
  height: 30px;
  border-radius: 6px;
  border: 1px solid var(--line-soft);
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  font-size: 13px;
  transition: 0.2s;
}

.close:hover {
  color: var(--danger);
  border-color: rgba(255, 77, 109, 0.45);
  background: rgba(255, 77, 109, 0.1);
}

.modal-body {
  padding: 18px 0 4px;
  overflow: auto;
}

.modal-foot {
  padding-top: 16px;
  margin-top: 6px;
  border-top: 1px solid var(--line-soft);
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.22s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}
</style>
