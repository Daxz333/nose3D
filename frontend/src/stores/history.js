import { defineStore } from 'pinia'

const STORAGE_KEY = 'aihb.history.v1'
const MAX_RECORDS = 24

function readStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

/**
 * 处理历史：保存在浏览器 localStorage 中。
 * 仅存压缩后的缩略图，避免超出 localStorage 容量限制。
 */
export const useHistoryStore = defineStore('history', {
  state: () => ({
    records: readStorage(),
  }),

  getters: {
    count: (state) => state.records.length,
  },

  actions: {
    persist() {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(this.records))
      } catch {
        // 容量溢出时逐步裁剪后重试
        this.records = this.records.slice(0, 8)
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(this.records))
        } catch {
          this.records = []
        }
      }
    },

    add(record) {
      this.records.unshift({
        id: `r_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        createdAt: Date.now(),
        ...record,
      })
      if (this.records.length > MAX_RECORDS) {
        this.records.length = MAX_RECORDS
      }
      this.persist()
    },

    remove(id) {
      this.records = this.records.filter((item) => item.id !== id)
      this.persist()
    },

    clear() {
      this.records = []
      this.persist()
    },
  },
})
