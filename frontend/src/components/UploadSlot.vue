<script setup>
import { ref } from 'vue'
import { formatBytes } from '@/utils/image'

const props = defineProps({
  title: { type: String, required: true },
  tag: { type: String, default: '' },
  hint: { type: String, default: '支持 JPG / PNG，拖拽或点击选择' },
  preview: { type: String, default: '' },
  file: { type: Object, default: null },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['select', 'clear'])

const inputRef = ref(null)
const dragging = ref(false)

function openPicker() {
  if (props.disabled) return
  inputRef.value?.click()
}

function handleFiles(files) {
  if (props.disabled) return
  const file = files?.[0]
  if (!file) return
  if (!file.type.startsWith('image/')) return
  emit('select', file)
}

function onChange(event) {
  handleFiles(event.target.files)
  event.target.value = ''
}

function onDrop(event) {
  dragging.value = false
  handleFiles(event.dataTransfer?.files)
}

function onDragOver() {
  if (!props.disabled) dragging.value = true
}

function onDragLeave() {
  dragging.value = false
}

function clear(event) {
  event.stopPropagation()
  emit('clear')
}
</script>

<template>
  <div class="slot" :class="{ dragging, filled: !!preview, disabled }">
    <div class="slot-head">
      <span class="slot-title">{{ title }}</span>
      <span class="tag">{{ tag }}</span>
    </div>

    <div
      class="dropzone"
      role="button"
      tabindex="0"
      @click="openPicker"
      @keydown.enter.prevent="openPicker"
      @keydown.space.prevent="openPicker"
      @dragover.prevent="onDragOver"
      @dragleave="onDragLeave"
      @drop.prevent="onDrop"
    >
      <template v-if="preview">
        <img class="preview" :src="preview" :alt="title" />
        <button v-if="!disabled" class="clear" type="button" @click="clear">✕</button>
        <div class="meta mono">
          <span>{{ file?.name?.slice(0, 22) || 'image' }}</span>
          <span>{{ formatBytes(file?.size) }}</span>
        </div>
      </template>

      <template v-else>
        <div class="placeholder">
          <span class="plus">+</span>
          <span class="ph-title">点击或拖拽上传</span>
          <span class="ph-hint">{{ hint }}</span>
        </div>
      </template>
    </div>

    <input
      ref="inputRef"
      class="native-input"
      type="file"
      accept="image/*"
      @change="onChange"
    />
  </div>
</template>

<style scoped>
.slot {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}

.slot-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.slot-title {
  font-size: 14px;
  color: var(--text-strong);
  letter-spacing: 1px;
}

.dropzone {
  position: relative;
  height: 210px;
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: rgba(0, 240, 255, 0.03);
  cursor: pointer;
  overflow: hidden;
  transition: 0.25s;
  outline: none;
}

.dropzone:hover,
.dropzone:focus-visible {
  border-color: var(--cyan);
  background: rgba(0, 240, 255, 0.07);
  box-shadow: 0 0 22px rgba(0, 240, 255, 0.16) inset;
}

.slot.dragging .dropzone {
  border-color: var(--cyan);
  border-style: solid;
  background: rgba(0, 240, 255, 0.12);
  box-shadow: 0 0 26px rgba(0, 240, 255, 0.28) inset;
}

.slot.disabled .dropzone {
  cursor: not-allowed;
  opacity: 0.55;
}

.placeholder {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  color: var(--muted);
  text-align: center;
  padding: 0 16px;
}

.plus {
  font-size: 30px;
  line-height: 1;
  color: var(--cyan);
  text-shadow: 0 0 14px rgba(0, 240, 255, 0.6);
}

.ph-title {
  font-size: 13px;
  letter-spacing: 1px;
  color: var(--text);
}

.ph-hint {
  font-size: 11px;
  letter-spacing: 0.5px;
}

.preview {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: var(--panel-2);
  display: block;
}

.clear {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 26px;
  height: 26px;
  border-radius: 6px;
  border: 1px solid rgba(255, 77, 109, 0.4);
  background: rgba(5, 10, 18, 0.85);
  color: var(--danger);
  font-size: 12px;
  cursor: pointer;
  display: grid;
  place-items: center;
  transition: 0.2s;
}

.clear:hover {
  background: rgba(255, 77, 109, 0.2);
}

.meta {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 5px 10px;
  font-size: 10px;
  letter-spacing: 0.5px;
  color: var(--cyan);
  background: linear-gradient(180deg, transparent, rgba(5, 10, 18, 0.92));
}

.native-input {
  display: none;
}
</style>
