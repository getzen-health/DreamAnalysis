import { describe, it, expect } from "vitest";

describe("useVoiceEmotion types", () => {
  it("exports useVoiceEmotion as a function", async () => {
    const mod = await import("@/hooks/use-voice-emotion");
    expect(typeof mod.useVoiceEmotion).toBe("function");
  });
});
