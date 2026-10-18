<script setup>
import { REGIONS } from '@/constants/regions'

defineProps({
  modelValue: { type: String, default: 'nose' },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue'])

function pick(value) {
  emit('update:modelValue', value)
}
</script>

<template>
  <div class="region">
    <button
      v-for="item in REGIONS"
      :key="item.value"
      type="button"
      class="chip"
      :class="{ active: modelValue === item.value, unsupported: !item.supported }"
      :disabled="disabled"
      @click="pick(item.value)"
    >
      <span class="chip-main">
        <span class="chip-label">{{ item.label }}</span>
        <span class="chip-en mono">{{ item.en }}</span>
      </span>
      <span class="chip-desc">{{ item.desc }}</span>
      <span v-if="!item.supported" class="chip-flag mono">规划中</span>
    </button>
  </div>
</template>

<style scoped>
.region {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
}

.chip {
  position: relative;
  text-align: left;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  background: rgba(5, 10, 18, 0.7);
  border: 1px solid var(--line-soft);
  color: var(--text);
  font-family: inherit;
  transition: 0.22s;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.chip:hover:not(:disabled) {
  border-color: var(--line);
  background: rgba(0, 240, 255, 0.06);
}

.chip.active {
  border-color: var(--cyan);
  background: rgba(0, 240, 255, 0.1);
  box-shadow: 0 0 20px rgba(0, 240, 255, 0.2) inset;
}

.chip:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.chip-main {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.chip-label {
  font-size: 14px;
  color: var(--text-strong);
  letter-spacing: 1px;
}

.chip.active .chip-label {
  color: var(--cyan);
}

.chip-en {
  font-size: 9px;
  letter-spacing: 1.5px;
  color: var(--muted);
}

.chip-desc {
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.5px;
}

.chip-flag {
  position: absolute;
  top: 8px;
  right: 8px;
  font-size: 9px;
  letter-spacing: 1px;
  padding: 1px 5px;
  border-radius: 4px;
  color: var(--warn);
  border: 1px solid rgba(255, 181, 71, 0.3);
  background: rgba(255, 181, 71, 0.08);
}
</style>
