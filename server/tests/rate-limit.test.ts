/**
 * Integration tests for IP-based rate limiting on auth endpoints — issue #322.
 *
 * Runs against the live Vercel production API.
 * Tests that repeated requests to /api/auth/login get rate-limited (429)
 * and that the Retry-After header is present on 429 responses.
 *
 * NOTE: These tests hit the login endpoint with bad credentials.
 * The rate limit is 10 attempts per IP per 15 minutes.
 * After 10 requests, the 11th should return 429.
 */
import { describe, it, expect } from "vitest";

const API = "https://dream-analysis.vercel.app";

// ── helpers ─────────────────────────────────────────────────────────────────

async function postLogin(username: string, password: string) {
  const res = await fetch(`${API}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return {
    status: res.status,
    headers: res.headers,
    json: await res.json().catch(() => ({})),
  };
}

// ── Rate limit tests ────────────────────────────────────────────────────────

describe("Issue #322: Rate limiting on auth endpoints", () => {
  // Use a unique fake username per test run to avoid interfering with real users.
  const fakeUser = `ratelimit_test_${Date.now()}`;
  const fakePass = "wrong_password_12345";

  it("should return 429 after exceeding login rate limit", async () => {
    const results: number[] = [];

    // Send 11 login attempts rapidly with bad credentials.
    // The rate limit is 10 per 15 minutes, so attempt #11 should be 429.
    for (let i = 0; i < 11; i++) {
      const res = await postLogin(fakeUser, fakePass);
      results.push(res.status);
    }

    // First 10 should be 401 (bad credentials) or 400 (bad request) — not 429
    const firstTen = results.slice(0, 10);
    for (const status of firstTen) {
      expect(status).not.toBe(429);
    }

    // The 11th should be 429
    expect(results[10]).toBe(429);
  });

  it("should include Retry-After header on 429 responses", async () => {
    // After the previous test, the rate limit is already exceeded for this IP.
    // One more request should still be 429.
    const res = await postLogin(fakeUser, fakePass);

    expect(res.status).toBe(429);
    const retryAfter = res.headers.get("retry-after");
    expect(retryAfter).toBeTruthy();
    expect(Number(retryAfter)).toBeGreaterThan(0);
  });

  it("should include error message in 429 response body", async () => {
    const res = await postLogin(fakeUser, fakePass);

    expect(res.status).toBe(429);
    expect(res.json.error).toBeDefined();
    expect(typeof res.json.error).toBe("string");
  });
});
