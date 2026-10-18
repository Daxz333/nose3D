import { defineStore } from 'pinia'

let seed = 0

export const useToastStore = defineStore('toast', {
  state: () => ({
    items: [],
  }),

  actions: {
    push(message, type = 'info', duration = 3200) {
      const id = ++seed
      this.items.push({ id, message, type })
      if (duration > 0) {
        setTimeout(() => this.remove(id), duration)
      }
      return id
    },

    success(message) {
      return this.push(message, 'success')
    },

    error(message) {
      return this.push(message, 'error', 5000)
    },

    info(message) {
      return this.push(message, 'info')
    },

    remove(id) {
      this.items = this.items.filter((item) => item.id !== id)
    },
  },
})
