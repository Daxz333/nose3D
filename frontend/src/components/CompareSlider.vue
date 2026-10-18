<script setup>
import { onBeforeUnmount, ref } from 'vue'

defineProps({
  before: { type: String, default: '' },
  after: { type: String, default: '' },
  beforeLabel: { type: String, default: 'BEFORE' },
  afterLabel: { type: String, default: 'AFTER' },
})

const root = ref(null)
const pos = ref(50)
let dragging = false

function updateFromEvent(event) {
  const el = root.value
  if (!el) return
  const rect = el.getBoundingClientRect()
  const ratio = ((event.clientX - rect.left) / rect.width) * 100
  pos.value = Math.min(100, Math.max(0, ratio))
}

function onMove(event) {
  if (dragging) updateFromEvent(event)
}

function stopDrag() {
  dragging = false
  window.removeEventListener('pointermove', onMove)
  window.removeEventListener('pointerup', stopDrag)
}

function startDrag(event) {
  dragging = true
  updateFromEvent(event)
  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', stopDrag)
}

onBeforeUnmount(stopDrag)
</script>

<template>
  <div ref="root" class="compare" :style="{ '--pos': pos + '%' }" @pointerdown="startDrag">
    <img class="layer" :src="after" alt="处理结果" draggable="false" />

    <div class="clip">
      <img class="layer" :src="before" alt="原始图片" draggable="false" />
    </div>

    <div class="handle" :style="{ left: pos + '%' }">
      <span class="knob">⇆</span>
    </div>

    <span class="badge left mono">{{ beforeLabel }}</span>
    <span class="badge right mono">{{ afterLabel }}</span>
    <span class="hint mono">DRAG TO COMPARE</span>
  </div>
</template>

<style scoped>
.compare {
  position: relative;
  width: 100%;
  height: clamp(280px, 48vh, 480px);
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--line);
  background: var(--panel-2);
  cursor: ew-resize;
  user-select: none;
  touch-action: none;
  box-shadow: 0 0 34px rgba(0, 240, 255, 0.12);
}

.layer {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.clip {
  position: absolute;
  inset: 0;
  clip-path: inset(0 calc(100% - var(--pos)) 0 0);
}

.handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--cyan);
  box-shadow: 0 0 14px rgba(0, 240, 255, 0.85);
  transform: translateX(-1px);
  pointer-events: none;
}

.knob {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 34px;
  height: 34px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-size: 14px;
  color: #02121a;
  background: linear-gradient(135deg, var(--cyan), var(--blue));
  box-shadow: 0 0 18px rgba(0, 240, 255, 0.7);
}

.badge {
  position: absolute;
  top: 10px;
  font-size: 10px;
  letter-spacing: 2px;
  padding: 3px 9px;
  border-radius: 4px;
  color: var(--cyan);
  background: rgba(5, 10, 18, 0.85);
  border: 1px solid var(--line);
  pointer-events: none;
}

.badge.left {
  left: 10px;
}

.badge.right {
  right: 10px;
}

.hint {
  position: absolute;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 9px;
  letter-spacing: 3px;
  color: var(--muted);
  pointer-events: none;
}
</style>
