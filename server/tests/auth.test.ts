/**
 * Integration tests for auth endpoints — issues #331, #326, #327.
 *
 * Runs against the live Vercel production API.
 * Tests email uniqueness, authorization checks, and admin role gating.
 */
import { describe, it, expect } from "vitest";

const API = "https://dream-analysis.vercel.app";

// ─── helpers ─────────────────────────────────────────────────────────────────

async function post(path: string, body: Record<string, unknown>, token?: string) {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let json: any;
  try { json = JSON.parse(text); } catch { json = { raw: text }; }
  return { status: res.status, json };
}

async function get(path: string, token?: string) {
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API}${path}`, { headers });
  const text = await res.text();
  let json: any;
  try { json = JSON.parse(text); } catch { json = { raw: text }; }
  return { status: res.status, json };
}

function randomUsername() {
  return `test_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

// ─── Issue #331: Email unique constraint ─────────────────────────────────────

describe("Issue #331: Email unique constraint", () => {
  const sharedEmail = `test_${Date.now()}@example.com`;
  let firstToken: string;

  it("should allow registration with a unique email", async () => {
    const res = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
      email: sharedEmail,
    });
    expect(res.status).toBe(201);
    expect(res.json.user).toBeDefined();
    expect(res.json.token).toBeDefined();
    firstToken = res.json.token;
  });

  it("should reject registration with a duplicate email", async () => {
    const res = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
      email: sharedEmail,
    });
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
  });

  it("should reject registration with duplicate email in different case", async () => {
    const res = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
      email: sharedEmail.toUpperCase(),
    });
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
  });

  it("should allow registration without email (null email)", async () => {
    const res = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
    });
    expect(res.status).toBe(201);
    expect(res.json.user).toBeDefined();
  });

  it("should allow multiple registrations without email", async () => {
    const res1 = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
    });
    const res2 = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
    });
    expect(res1.status).toBe(201);
    expect(res2.status).toBe(201);
  });

  it("should reject duplicate username", async () => {
    const username = randomUsername();
    await post("/api/auth/register", { username, password: "testpass123" });
    const res = await post("/api/auth/register", { username, password: "testpass123" });
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
  });
});

// ─── Issue #326: Horizontal privilege escalation ─────────────────────────────

describe("Issue #326: Authorization — users cannot access other users' data", () => {
  let userAToken: string;
  let userAId: string;
  let userBToken: string;
  let userBId: string;

  it("setup: create two test users", async () => {
    const resA = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
    });
    expect(resA.status).toBe(201);
    userAToken = resA.json.token;
    userAId = resA.json.user.id;

    const resB = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
    });
    expect(resB.status).toBe(201);
    userBToken = resB.json.token;
    userBId = resB.json.user.id;

    expect(userAId).not.toBe(userBId);
  });

  it("should allow user A to access their own health metrics", async () => {
    const res = await get(`/api/health-metrics/${userAId}`, userAToken);
    // 200 or 404 (no data yet) is fine — just not 403
    expect([200, 404]).toContain(res.status);
  });

  it("should block user A from accessing user B's health metrics", async () => {
    const res = await get(`/api/health-metrics/${userBId}`, userAToken);
    expect(res.status).toBe(403);
  });

  it("should block user A from accessing user B's dream analysis", async () => {
    const res = await get(`/api/dream-analysis/${userBId}`, userAToken);
    expect(res.status).toBe(403);
  });

  it("should block user A from accessing user B's chat history", async () => {
    const res = await get(`/api/ai-chat/${userBId}`, userAToken);
    expect(res.status).toBe(403);
  });

  it("should block user A from accessing user B's settings", async () => {
    const res = await get(`/api/settings/${userBId}`, userAToken);
    expect(res.status).toBe(403);
  });

  it("should block user A from accessing user B's data export", async () => {
    const res = await get(`/api/export/${userBId}`, userAToken);
    expect(res.status).toBe(403);
  });

  it("should block unauthenticated access to any user's data", async () => {
    const res = await get(`/api/health-metrics/${userAId}`);
    expect(res.status).toBeGreaterThanOrEqual(401);
  });
});

// ─── Issue #327: Admin role check ────────────────────────────────────────────

describe("Issue #327: Admin endpoints require admin role", () => {
  let regularUserToken: string;

  it("setup: create a regular user", async () => {
    const res = await post("/api/auth/register", {
      username: randomUsername(),
      password: "testpass123",
    });
    expect(res.status).toBe(201);
    regularUserToken = res.json.token;
  });

  it("should block regular user from study admin participants", async () => {
    const res = await get("/api/study/admin/participants", regularUserToken);
    expect(res.status).toBe(403);
  });

  it("should block regular user from study admin sessions", async () => {
    const res = await get("/api/study/admin/sessions", regularUserToken);
    expect(res.status).toBe(403);
  });

  it("should block regular user from study admin stats", async () => {
    const res = await get("/api/study/admin/stats", regularUserToken);
    expect(res.status).toBe(403);
  });

  it("should block regular user from study admin CSV export", async () => {
    const res = await get("/api/study/admin/export-csv", regularUserToken);
    expect(res.status).toBe(403);
  });
});
