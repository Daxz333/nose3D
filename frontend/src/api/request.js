import axios from 'axios'

/**
 * 后端为 Flask（flask_server.py），接口无需登录：
 *  - 开发：由 Vite 代理转发到 127.0.0.1:6006
 *  - 生产：与后端同源（Flask 托管）
 */
const baseURL = import.meta.env.VITE_API_BASE || '/'

const request = axios.create({
  baseURL,
  // 换鼻推理（检测 + 分割 + 仿射优化 + GFPGAN 超分）耗时较长
  timeout: 300000,
})

request.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status
    let message = '网络异常，请确认后端服务已启动（127.0.0.1:6006）'
    if (status === 413) {
      message = '上传的图片过大，请压缩后重试'
    } else if (status === 504) {
      message = '处理超时，请稍后重试'
    } else if (status) {
      message = error.response?.data?.error || `请求失败（HTTP ${status}）`
    }
    return Promise.reject(new Error(message))
  }
)

export default request
