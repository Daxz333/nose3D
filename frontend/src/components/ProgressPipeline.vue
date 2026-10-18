<script setup>
import { computed } from 'vue'

const props = defineProps({
  /** 整体进度 0 - 100 */
  progress: { type: Number, default: 0 },
  /** 阶段定义（含 weight），顺序即执行顺序 */
  stages: { type: Array, default: () => [] },
  /** 当前所处阶段下标 */
  activeIndex: { type: Number, default: 0 },
  /** 已耗时（秒） */
  elapsed: { type: Number, default: 0 },
  /** 预估总耗时（秒） */
  estimated: { type: Number, default: 30 },
  /** 是否已完成 */
  finished: { type: Boolean, default: false },
})

/** 各阶段起始的累计权重（即进度条上的分段位置） */
const offsets = computed(() => {
  const list = []
  let acc = 0
  for (const stage of props.stages) {
    list.push(acc)
    acc += stage.weight
  }
  return list
})

const totalWeight = computed(() => props.stages.reduce((sum, s) => sum + s.weight, 0) || 1)

/** 某个分段内部被填充的比例 0 - 1 */
function segmentFill(index) {
  const weight = props.stages[index]?.weight || 0
  if (weight <= 0) return 0
  const ratio = (props.progress - offsets.value[index]) / weight
  return Math.min(1, Math.max(0, ratio))
}

/** 某个分段的状态：done / active / wait */
function segmentState(index) {
  if (props.finished) return 'done'
  if (index < props.activeIndex) return 'done'
  if (index === props.activeIndex) return 'active'
  return 'wait'
}

const currentStage = computed(() => props.stages[props.activeIndex] || null)

const percentText = computed(() => props.progress.toFixed(1))

const elapsedText = computed(() => props.elapsed.toFixed(1))

const remainingText = computed(() => {
  const left = props.estimated - props.elapsed
  return left > 0 ? left.toFixed(1) : '0.0'
})
</script>

<template>
  <section class="panel corner progress-panel">
    <!-- 头部 -->
    <header class="head">
      <div class="head-left">
        <span v-if="finished" class="badge done">✓</span>
        <span v-else class="spinner" />
        <div class="head-text">
          <h2>{{ finished ? '处理完成' : '推理进行中' }}</h2>
          <p class="mono">
            {{ finished ? 'PIPELINE FINISHED' : 'PIPELINE RUNNING · 请勿关闭页面' }}
          </p>
        </div>
      </div>

      <div class="percent">
        <span class="percent-num mono">{{ percentText }}</span>
        <span class="percent-unit mono">%</span>
      </div>
    </header>

    <!-- 分段进度条 -->
    <div class="bar" role="progressbar" :aria-valuenow="Math.round(progress)" aria-valuemin="0" aria-valuemax="100">
      <div
        v-for="(stage, index) in stages"
        :key="stage.key"
        class="seg"
        :class="segmentState(index)"
        :style="{ flexGrow: stage.weight / totalWeight }"
      >
        <div class="seg-fill" :style="{ width: (segmentFill(index) * 100).toFixed(2) + '%' }" />
      </div>
    </div>

    <!-- 阶段刻度 -->
    <div class="ticks">
      <span
        v-for="(stage, index) in stages"
        :key="stage.key"
        class="tick mono"
        :class="segmentState(index)"
        :style="{ left: ((offsets[index] + stage.weight / 2) / totalWeight * 100).toFixed(2) + '%' }"
      >
        {{ index + 1 }}
      </span>
    </div>

    <!-- 当前阶段 -->
    <div v-if="currentStage" class="current">
      <div class="current-main">
        <span class="current-label">{{ currentStage.label }}</span>
        <span class="tag mono">{{ currentStage.en }}</span>
      </div>
      <p class="current-desc">{{ currentStage.desc }}</p>
      <div class="current-meta mono">
        <span>STEP {{ activeIndex + 1 }} / {{ stages.length }}</span>
        <span>已用 {{ elapsedText }}s · 预估剩余 {{ remainingText }}s</span>
      </div>
    </div>

    <!-- 阶段清单 -->
    <ol class="list">
      <li
        v-for="(stage, index) in stages"
        :key="stage.key"
        class="item"
        :class="segmentState(index)"
      >
        <span class="item-no mono">{{ String(index + 1).padStart(2, '0') }}</span>
        <span class="item-label">{{ stage.label }}</span>
        <span class="item-state mono">
          {{ segmentState(index) === 'done' ? 'DONE' : segmentState(index) === 'active' ? 'RUNNING' : 'WAIT' }}
        </span>
      </li>
    </ol>

    <p class="footnote mono">
      * 后端为同步接口，进度按历史平均耗时（本次预估 {{ estimated.toFixed(1) }}s）估算，非精确上报值
    </p>
  </section>
</template>

<style scoped>
.progress-panel {
  margin-top: 22px;
}

/* ---- 头部 ---- */
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--line-soft);
}

.head-left {
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
}

.head-text p {
  margin: 2px 0 0;
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--muted);
}

.spinner {
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 2px solid rgba(0, 240, 255, 0.18);
  border-top-color: var(--cyan);
  animation: spin 0.9s linear infinite;
  box-shadow: 0 0 16px rgba(0, 240, 255, 0.3);
}

.badge.done {
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-size: 15px;
  color: #02121a;
  background: var(--ok);
  box-shadow: 0 0 20px rgba(40, 224, 168, 0.55);
}

.percent {
  display: flex;
  align-items: baseline;
  gap: 2px;
  flex: 0 0 auto;
}

.percent-num {
  font-size: 32px;
  line-height: 1;
  color: var(--cyan);
  letter-spacing: 1px;
  text-shadow: 0 0 20px rgba(0, 240, 255, 0.55);
}

.percent-unit {
  font-size: 14px;
  color: var(--muted);
}

/* ---- 进度条 ---- */
.bar {
  display: flex;
  gap: 3px;
  margin-top: 20px;
  height: 10px;
}

.seg {
  position: relative;
  flex: 0 1 auto;
  min-width: 4px;
  border-radius: 3px;
  background: rgba(0, 240, 255, 0.07);
  border: 1px solid rgba(0, 240, 255, 0.1);
  overflow: hidden;
}

.seg-fill {
  height: 100%;
  width: 0;
  border-radius: 2px;
  background: linear-gradient(90deg, rgba(0, 240, 255, 0.75), var(--cyan));
  transition: width 0.4s ease-out;
}

.seg.done .seg-fill {
  background: linear-gradient(90deg, rgba(40, 224, 168, 0.7), var(--ok));
}

.seg.active .seg-fill {
  box-shadow: 0 0 12px rgba(0, 240, 255, 0.75);
  background: linear-gradient(90deg, rgba(0, 240, 255, 0.85), var(--cyan) 60%, #9df6ff);
}

/* ---- 刻度 ---- */
.ticks {
  position: relative;
  height: 16px;
  margin-top: 2px;
}

.tick {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  font-size: 9px;
  letter-spacing: 0;
  color: var(--muted);
  opacity: 0.55;
  transition: 0.3s;
}

.tick.done {
  color: var(--ok);
  opacity: 0.9;
}

.tick.active {
  color: var(--cyan);
  opacity: 1;
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.7);
}

/* ---- 当前阶段 ---- */
.current {
  margin-top: 12px;
  padding: 14px 16px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(0, 240, 255, 0.06);
  box-shadow: 0 0 22px rgba(0, 240, 255, 0.1) inset;
  animation: fadeUp 0.3s ease both;
}

.current-main {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.current-label {
  font-size: 15px;
  color: var(--text-strong);
  letter-spacing: 1px;
}

.current-desc {
  margin: 6px 0 0;
  font-size: 12.5px;
  line-height: 1.7;
  color: var(--muted);
}

.current-meta {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 10px;
  padding-top: 9px;
  border-top: 1px dashed var(--line-soft);
  font-size: 10px;
  letter-spacing: 1.5px;
  color: var(--cyan);
  opacity: 0.85;
}

/* ---- 阶段清单 ---- */
.list {
  list-style: none;
  margin: 18px 0 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 8px;
}

.item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12.5px;
  color: var(--muted);
  border: 1px solid var(--line-soft);
  background: rgba(5, 10, 18, 0.55);
  transition: 0.3s;
}

.item.active {
  color: var(--cyan);
  border-color: var(--line);
  background: rgba(0, 240, 255, 0.08);
}

.item.done {
  color: var(--ok);
  border-color: rgba(40, 224, 168, 0.28);
}

.item-no {
  font-size: 10px;
  letter-spacing: 1px;
  opacity: 0.8;
}

.item-label {
  flex: 1;
  min-width: 0;
}

.item-state {
  font-size: 9px;
  letter-spacing: 1.5px;
  opacity: 0.85;
}

.item.active .item-no {
  animation: pulse 1.1s infinite;
}

.footnote {
  margin: 16px 0 0;
  font-size: 10px;
  letter-spacing: 0.5px;
  line-height: 1.7;
  color: var(--muted);
  opacity: 0.75;
}

@media (max-width: 720px) {
  .percent-num {
    font-size: 24px;
  }
}
</style>
