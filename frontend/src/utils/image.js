/** 读取 File 为 dataURL */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('图片读取失败'))
    reader.readAsDataURL(file)
  })
}

/** 加载图片，返回 HTMLImageElement */
export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('图片解析失败'))
    img.src = src
  })
}

/** 按最长边压缩 dataURL，用于预览与本地历史存储 */
export async function compressDataURL(dataURL, maxSize = 900, quality = 0.82) {
  if (!dataURL) return ''
  try {
    const img = await loadImage(dataURL)
    const scale = Math.min(1, maxSize / Math.max(img.width, img.height))
    const width = Math.max(1, Math.round(img.width * scale))
    const height = Math.max(1, Math.round(img.height * scale))

    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, width, height)
    ctx.drawImage(img, 0, 0, width, height)

    return canvas.toDataURL('image/jpeg', quality)
  } catch {
    return dataURL
  }
}

/** 后端返回的是纯 base64，这里补齐 dataURL 前缀 */
export function base64ToDataURL(base64, mime = 'image/jpeg') {
  if (!base64) return ''
  return base64.startsWith('data:') ? base64 : `data:${mime};base64,${base64}`
}

/** 触发浏览器下载 */
export function downloadDataURL(dataURL, filename) {
  const link = document.createElement('a')
  link.href = dataURL
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/** 生成带时间戳的文件名 */
export function timestampedName(prefix = 'result', ext = 'jpg') {
  const d = new Date()
  const p = (n) => String(n).padStart(2, '0')
  const stamp =
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  return `${prefix}_${stamp}.${ext}`
}

/** 格式化时间 */
export function formatTime(ts) {
  const d = new Date(ts)
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

/** 人类可读文件大小 */
export function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}
