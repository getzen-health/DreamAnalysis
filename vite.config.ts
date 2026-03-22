import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import path from "path";

export default defineConfig({
  plugins: [
    react(),
    // Sentry source map upload — only in production builds when auth token is set
    ...(process.env.SENTRY_AUTH_TOKEN
      ? [
          sentryVitePlugin({
            org: process.env.SENTRY_ORG,
            project: process.env.SENTRY_PROJECT,
            authToken: process.env.SENTRY_AUTH_TOKEN,
            sourcemaps: { filesToDeleteAfterUpload: ["./**/*.map"] },
          }),
        ]
      : []),
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
      // @capacitor/camera is dynamically imported only on native — not available in web builds
      external: ["@capacitor/camera"],
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
