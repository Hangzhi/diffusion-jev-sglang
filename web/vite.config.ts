import { defineConfig } from "vite";
export default defineConfig({
  build: { outDir: "../src/diffusion_jev/static", emptyOutDir: true },
  server: {
    proxy: {
      "/v1": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
    },
  },
});
