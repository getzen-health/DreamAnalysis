import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { resolve } from "path";

/**
 * Tests for APK size optimization configuration (Issue #515).
 *
 * Verifies that build.gradle has the correct R8/minification settings
 * for release builds to reduce APK size.
 */

describe("APK size optimization (Issue #515)", () => {
  const buildGradlePath = resolve(__dirname, "../../../../android/app/build.gradle");
  let buildGradle: string;

  try {
    buildGradle = readFileSync(buildGradlePath, "utf-8");
  } catch {
    buildGradle = "";
  }

  it("enables R8 minification for release builds", () => {
    // minifyEnabled true must be in the release buildType
    expect(buildGradle).toContain("minifyEnabled true");
  });

  it("enables resource shrinking for release builds", () => {
    expect(buildGradle).toContain("shrinkResources true");
  });

  it("includes proguard configuration", () => {
    expect(buildGradle).toContain("proguard-android-optimize.txt");
  });

  it("strips debug symbols for release builds", () => {
    expect(buildGradle).toContain("debugSymbolLevel");
    expect(buildGradle).toContain("SYMBOL_TABLE");
  });

  it("configures AAB as default for release", () => {
    // bundle block should exist in android config
    expect(buildGradle).toContain("bundle");
    expect(buildGradle).toContain("enableSplit");
  });

  it("has PNG-to-WebP conversion comment", () => {
    // Should document that PNGs should be converted to WebP
    expect(buildGradle).toMatch(/[Ww]eb[Pp]/);
  });
});
