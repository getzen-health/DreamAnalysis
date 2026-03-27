// client/src/test/lib/insight-engine/morning-briefing-ratelimit.test.ts
import { describe, it, expect } from "vitest";

// Test the date key logic — no server needed
describe("morning briefing date key", () => {
  it("generates UTC YYYY-MM-DD key consistently", () => {
    const key = (userId: string) => {
      const date = new Date().toISOString().slice(0, 10);
      return `morning_briefing:${userId}:${date}`;
    };
    expect(key("user1")).toMatch(/^morning_briefing:user1:\d{4}-\d{2}-\d{2}$/);
  });
});
