<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'

const props = defineProps({ url: { type: String, default: '' } })
const host = ref(null)
const gray = ref(true)
const loading = ref(false)
const error = ref('')
let renderer, scene, camera, controls, observer, model, frame, generation = 0
const grayMaterial = new THREE.MeshStandardMaterial({ color: 0xb7bec8, roughness: .85, side: THREE.DoubleSide })

function disposeModel(object) {
  object?.traverse((part) => {
    if (!part.isMesh) return
    part.geometry.dispose()
    const material = part.userData.photoMaterial || part.material
    for (const item of Array.isArray(material) ? material : [material]) {
      for (const value of Object.values(item)) if (value?.isTexture) value.dispose()
      item.dispose()
    }
  })
}

function applyMaterial() {
  model?.traverse((part) => {
    if (!part.isMesh) return
    if (!part.userData.photoMaterial) part.userData.photoMaterial = part.material
    part.material = gray.value ? grayMaterial : part.userData.photoMaterial
  })
}

function view(direction) {
  if (!camera) return
  const positions = { front: [0, .015, .47], face: [0, .035, .30], side: [.47, .015, 0], back: [0, .015, -.47] }
  camera.position.set(...positions[direction])
  controls.target.set(0, direction === 'face' ? .035 : -.015, direction === 'face' ? .04 : 0)
  controls.update()
}

function load() {
  if (!scene) return
  const token = ++generation
  if (model) { scene.remove(model); disposeModel(model); model = null }
  error.value = ''
  loading.value = Boolean(props.url)
  if (!props.url) return
  new GLTFLoader().load(props.url, (gltf) => {
    if (token !== generation) { disposeModel(gltf.scene); return }
    model = gltf.scene
    scene.add(model)
    applyMaterial()
    view('front')
    loading.value = false
  }, undefined, () => {
    if (token !== generation) return
    loading.value = false
    error.value = '模型加载失败，请确认服务运行后重新生成'
  })
}

watch(() => props.url, load)
watch(gray, applyMaterial)
onMounted(() => {
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.setClearColor(0x101c2b)
    host.value.appendChild(renderer.domElement)
    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(36, 1, .001, 10)
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.minDistance = .25
    controls.maxDistance = 1.4
    scene.add(new THREE.HemisphereLight(0xffffff, 0x68758a, 2.5))
    const light = new THREE.DirectionalLight(0xffe8d9, 2)
    light.position.set(.5, .8, 1)
    scene.add(light)
    const fill = new THREE.DirectionalLight(0xd3e8ff, 1)
    fill.position.set(-.6, .1, -.4)
    scene.add(fill)
    observer = new ResizeObserver(() => {
      const width = host.value.clientWidth, height = host.value.clientHeight
      renderer.setSize(width, height)
      camera.aspect = width / Math.max(height, 1)
      camera.updateProjectionMatrix()
    })
    observer.observe(host.value)
    function render() {
      frame = requestAnimationFrame(render)
      controls.update()
      renderer.render(scene, camera)
    }
    render()
    load()
  } catch {
    error.value = '当前浏览器无法启用 WebGL，请使用支持硬件加速的浏览器'
  }
})
onBeforeUnmount(() => {
  generation += 1
  cancelAnimationFrame(frame)
  observer?.disconnect()
  controls?.dispose()
  disposeModel(model)
  grayMaterial.dispose()
  renderer?.dispose()
})
</script>

<template>
  <div class="viewer">
    <div ref="host" class="canvas" role="img" aria-label="可旋转的三维头部模型" />
    <p v-if="loading || error" class="overlay" role="status">{{ error || '正在加载头部模型…' }}</p>
    <div class="toolbar">
      <button class="btn subtle" type="button" @click="view('front')">正面</button>
      <button class="btn subtle" type="button" @click="view('face')">五官近看</button>
      <button class="btn subtle" type="button" @click="view('side')">侧面</button>
      <button class="btn subtle" type="button" @click="view('back')">背面</button>
      <label><input v-model="gray" type="checkbox" /> 灰模 · 检查几何</label>
    </div>
    <p class="muted hint">拖动旋转 · 滚轮缩放 · 右键平移</p>
  </div>
</template>

<style scoped>
.viewer { position: relative; min-width: 0; }
.canvas { height: 490px; width: 100%; overflow: hidden; border-radius: 12px; border: 1px solid var(--line-soft); }
.canvas :deep(canvas) { display: block; touch-action: none; }
.toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
.toolbar label { margin-left: auto; font-size: 13px; }
.hint { font-size: 12px; margin: 10px 0 0; }
.overlay { position: absolute; top: 20px; left: 20px; right: 20px; padding: 12px; background: #132339; color: #fff; border-radius: 8px; }
@media (max-width: 720px) { .canvas { height: 380px; } }
</style>
