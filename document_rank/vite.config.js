import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // This allows external access (e.g. LAN or tunnels)
    allowedHosts: ['38e9-106-222-199-148.ngrok-free.app'], // This allows Ngrok or Cloudflare public URLs
  },
})
