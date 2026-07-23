import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/voice': 'http://localhost:8000',
      '/voices': 'http://localhost:8000',
      '/generate-speech': 'http://localhost:8000',
      '/trigger-wakeword': 'http://localhost:8000'
    }
  }
})
