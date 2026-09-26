import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/ui/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  build: { outDir: "../jev_like/agent/static", emptyOutDir: true },
  server: {
    proxy: { "/api": "http://127.0.0.1:8000", "/v1": "http://127.0.0.1:8000" },
  },
});
