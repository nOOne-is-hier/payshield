import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), tailwindcss()],
    server: {
    port: 5173,
    proxy: {
      // 프론트에서 /api 로 호출하면 FastAPI(127.0.0.1:8000)로 프록시
      '/api': {
        target: 'http://127.0.0.1:8000/api',
        changeOrigin: true,
        // 필요시 경로 유지: /api/metrics -> /metrics (rewrite)
        rewrite: (path) => path.replace(/^\/api/, ''),
      }
    }
  }
})
