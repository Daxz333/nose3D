import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

// 需要转发到 Flask 后端的路径（后端定义在 flask_server.py）
const API_ROUTES = ['/predict', '/api']

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://127.0.0.1:6006'

  const proxy = {}
  for (const route of API_ROUTES) {
    proxy[route] = { target, changeOrigin: true }
  }

  return {
    plugins: [vue()],
    // 相对路径，便于被 Flask 以 /app 等任意子路径托管
    base: './',
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    server: {
      port: 5173,
      host: true,
      // 同源代理后端推理和接口说明
      proxy,
    },
    build: {
      outDir: 'dist',
      assetsDir: 'assets',
      chunkSizeWarningLimit: 1200,
    },
  }
})
