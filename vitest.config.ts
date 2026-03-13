import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "client", "src"),
      "@shared": path.resolve(__dirname, "shared"),
      // Stub native-only Capacitor plugins that aren't installed in the web/test env
      "@capacitor/camera": path.resolve(__dirname, "client/src/test/__mocks__/capacitor-camera.ts"),
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./client/src/test/setup.ts"],
    include: ["client/src/**/*.test.{ts,tsx}"],
    globals: true,
  },
});
