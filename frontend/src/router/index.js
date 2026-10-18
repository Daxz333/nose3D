import { createRouter, createWebHashHistory } from 'vue-router'

import AppLayout from '@/layouts/AppLayout.vue'

const routes = [
  {
    path: '/login',
    redirect: { name: 'workbench' },
  },
  { path: '/register', redirect: { name: 'workbench' } },
  {
    path: '/',
    component: AppLayout,
    children: [
      { path: '', redirect: { name: 'workbench' } },
      {
        path: 'workbench',
        name: 'workbench',
        component: () => import('@/views/WorkbenchView.vue'),
        meta: { title: '换鼻工作台', en: 'WORKBENCH' },
      },
      {
        path: 'head',
        name: 'head',
        component: () => import('@/views/HeadView.vue'),
        meta: { title: '3D 头部模型', en: 'HEAD MODEL' },
      },
      {
        path: 'history',
        name: 'history',
        component: () => import('@/views/HistoryView.vue'),
        meta: { title: '历史记录', en: 'HISTORY' },
      },
      {
        path: 'guide',
        name: 'guide',
        component: () => import('@/views/GuideView.vue'),
        meta: { title: '算法说明', en: 'PIPELINE' },
      },
      {
        path: 'api',
        name: 'api',
        component: () => import('@/views/ApiView.vue'),
        meta: { title: '接口列表', en: 'API REFERENCE' },
      },
      {
        path: 'about',
        name: 'about',
        component: () => import('@/views/AboutView.vue'),
        meta: { title: '关于系统', en: 'ABOUT' },
      },
    ],
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/NotFoundView.vue'),
    meta: { title: '页面不存在' },
  },
]

const router = createRouter({
  // 使用 hash 模式，便于被 Flask 托管在任意子路径下而无需额外路由配置
  history: createWebHashHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

router.afterEach((to) => {
  const title = to.meta?.title
  document.title = title ? `${title} · AI 换鼻服务系统` : 'AI 换鼻服务系统'
})

export default router
