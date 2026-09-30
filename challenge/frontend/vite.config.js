import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the UI calls /api on the Vite origin, proxied to the local challenge API.
const apiTarget = process.env.CHALLENGE_API_PROXY || 'http://127.0.0.1:8001'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': { target: apiTarget, changeOrigin: false } } },
})
