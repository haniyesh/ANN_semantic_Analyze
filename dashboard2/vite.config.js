import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    proxy: {
      // All /api/* requests → FastAPI backend running on port 8000
      // Strips /api prefix: /api/news/all  →  http://localhost:8000/news/all
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // WebSocket proxy for live Binance stream
      // Used by BinanceChart: ws://localhost:5173/proxy/stream/BTCUSDT/15m
      '/proxy': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,   // <-- IMPORTANT: enables WebSocket proxying
      },
    },
  },
})
