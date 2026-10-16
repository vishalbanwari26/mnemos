import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // The static GitHub Pages build lives under /mnemos/ (VITE_BASE).
  base: process.env.VITE_BASE ?? '/',
  plugins: [react()],
})
