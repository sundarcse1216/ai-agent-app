import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxies /api/* to the FastAPI backend in dev, so the frontend can
    // use relative paths (no CORS, no hardcoded backend URL) — same
    // approach used for POST /api/chat, GET /api/agents, and the
    // /api/chat/stream SSE endpoint (proxying preserves streaming).
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
