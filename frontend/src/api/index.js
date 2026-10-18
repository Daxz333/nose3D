import request from './request'

export async function getHeadStatus() {
  const { data } = await request.get('/api/head/dense/status', { timeout: 10000 })
  return { ...data, template_url: request.getUri({ url: data.template_url }) }
}

export async function reconstructHead(file, signal, height = 200) {
  const form = new FormData()
  form.append('image', file)
  form.append('height_mm', String(height))
  const { data } = await request.post('/api/head/dense/reconstruct', form, { signal, timeout: 510000 })
  if (!data.success) return data
  return {
    ...data,
    model_url: request.getUri({ url: data.model_url }),
    download_url: request.getUri({ url: data.download_url }),
    stl_url: request.getUri({ url: data.stl_url }),
    report_url: request.getUri({ url: data.report_url }),
  }
}

/** 公开接口说明和支持部位。 */
export async function getEndpoints() {
  const { data } = await request.get('/api/endpoints', { timeout: 10000 })
  return data
}

/**
 * 换鼻 / 换部位：POST /predict
 * @param {{ faceFile: File, noseFile: File, region?: string }} payload
 * @returns {Promise<{ success: boolean, error: string|null, result: string|null }>}
 */
export async function predict({ faceFile, noseFile, region = 'nose' }) {
  const form = new FormData()
  form.append('face_image', faceFile)
  form.append('source_image', noseFile)
  form.append('region', region)

  const { data } = await request.post('/predict', form)
  return data
}
