import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@shared": path.resolve(__dirname, "..", "..", "shared"),
    },
  },
  test: {
    environment: "node",
    include: ["server/tests/**/*.test.ts"],
    globals: true,
    testTimeout: 30000,
  },
});
