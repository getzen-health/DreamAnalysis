import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [
    react(),
    // @replit/vite-plugin-runtime-error-modal only works on Replit — removed
  ],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "client", "src"),
      "@shared": path.resolve(import.meta.dirname, "shared"),
      "@assets": path.resolve(import.meta.dirname, "attached_assets"),
    },
    dedupe: ["react", "react-dom", "@tanstack/react-query"],
  },
  root: path.resolve(import.meta.dirname, "client"),
  build: {
    outDir: path.resolve(import.meta.dirname, "dist/public"),
    emptyOutDir: true,
    // cache-bust: 2026-02-20
    rollupOptions: {
      output: {
        manualChunks: {
          // Core React runtime — cached aggressively by browsers
          "vendor-react": ["react", "react-dom"],
          // TanStack Query — shared state layer
          "vendor-query": ["@tanstack/react-query"],
          // Charting — heaviest third-party dependency
          "vendor-charts": ["recharts"],
          // Animation library
          "vendor-motion": ["framer-motion"],
          // Radix UI primitives (shadcn/ui foundation)
          "vendor-radix": [
            "@radix-ui/react-dialog",
            "@radix-ui/react-dropdown-menu",
            "@radix-ui/react-select",
            "@radix-ui/react-tabs",
            "@radix-ui/react-tooltip",
          ],
          // ONNX Runtime Web — 403KB, lazy-loaded only when EEG inference fires.
          // Named chunk ensures stable cache key across deploys.
          "vendor-onnx": ["onnxruntime-web"],
        },
      },
    },
  },
  server: {
    fs: {
      strict: true,
      deny: ["**/.*"],
    },
  },
});
