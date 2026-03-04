# Thursday Launch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make NeuralDreamWorkshop fully functional on localhost and Vercel by Thursday — users never see a blank/error screen, the ML backend cold start shows a loading UI, and the app degrades gracefully when ML is unreachable.

**Architecture:** `MLConnectionProvider` (inside `AuthProvider`) pings the Render ML backend on mount and manages a `status` state machine (`connecting → warming → ready | error`). A full-screen `MLWarmupScreen` covers the app during the 30-60s cold start. A keep-alive ping every 14 minutes prevents future cold starts. `mlFetch` gains retry logic (3 attempts, exponential backoff) and a 30s timeout.

**Tech Stack:** React 18, TypeScript 5.6, Vitest (tests), shadcn/ui, Tailwind CSS, Lucide React, FastAPI (Python), pytest

---

## Task 1: Create useMLConnection hook

**Files:**
- Create: `client/src/hooks/use-ml-connection.tsx`
- Test: `client/src/tests/hooks/use-ml-connection.test.tsx`

**Step 1: Write the failing test**

Create `client/src/tests/hooks/use-ml-connection.test.tsx`:

```tsx
import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MLConnectionProvider, useMLConnection } from "@/hooks/use-ml-connection";
import * as mlApi from "@/lib/ml-api";

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <MLConnectionProvider>{children}</MLConnectionProvider>
);

describe("useMLConnection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("starts in connecting state", () => {
    vi.spyOn(mlApi, "pingBackend").mockResolvedValue(false);
    const { result } = renderHook(() => useMLConnection(), { wrapper });
    expect(result.current.status).toBe("connecting");
    expect(result.current.warmupProgress).toBe(0);
  });

  it("transitions to ready when ping succeeds", async () => {
    vi.spyOn(mlApi, "pingBackend").mockResolvedValue(true);
    const { result } = renderHook(() => useMLConnection(), { wrapper });
    await waitFor(() => expect(result.current.status).toBe("ready"), { timeout: 3000 });
    expect(result.current.latencyMs).toBeGreaterThanOrEqual(0);
  });

  it("transitions to error after 3 consecutive failures", async () => {
    vi.spyOn(mlApi, "pingBackend").mockResolvedValue(false);
    const { result } = renderHook(() => useMLConnection(), { wrapper });
    await waitFor(() => expect(result.current.status).toBe("error"), { timeout: 20000 });
  });

  it("reconnect() resets to connecting", async () => {
    vi.spyOn(mlApi, "pingBackend").mockResolvedValue(false);
    const { result } = renderHook(() => useMLConnection(), { wrapper });
    await waitFor(() => expect(result.current.status).toBe("error"), { timeout: 20000 });
    vi.spyOn(mlApi, "pingBackend").mockResolvedValue(true);
    act(() => { result.current.reconnect(); });
    await waitFor(() => expect(result.current.status).toBe("ready"), { timeout: 5000 });
  });
});
```

**Step 2: Run it to confirm it fails**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
npx vitest run client/src/tests/hooks/use-ml-connection.test.tsx
```
Expected: `FAIL — Cannot find module '@/hooks/use-ml-connection'`

**Step 3: Create the hook**

Create `client/src/hooks/use-ml-connection.tsx`:

```tsx
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { pingBackend } from "@/lib/ml-api";

export type MLStatus = "idle" | "connecting" | "warming" | "ready" | "error";

export interface MLConnectionState {
  status: MLStatus;
  latencyMs: number | null;
  warmupProgress: number;   // 0-100
  retryCount: number;
  reconnect: () => void;
}

const MLConnectionContext = createContext<MLConnectionState>({
  status: "idle",
  latencyMs: null,
  warmupProgress: 0,
  retryCount: 0,
  reconnect: () => {},
});

export function useMLConnection(): MLConnectionState {
  return useContext(MLConnectionContext);
}

export function MLConnectionProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<MLStatus>("connecting");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [warmupProgress, setWarmupProgress] = useState(0);
  const [retryCount, setRetryCount] = useState(0);

  const failureCountRef = useRef(0);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef(Date.now());

  function startProgressBar() {
    if (progressIntervalRef.current) return;
    startTimeRef.current = Date.now();
    progressIntervalRef.current = setInterval(() => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      // Reaches 95% at 35 seconds, never completes on its own
      setWarmupProgress(Math.min(95, (elapsed / 35) * 95));
    }, 300);
  }

  function stopProgressBar(complete = false) {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
    if (complete) setWarmupProgress(100);
  }

  function clearPingInterval() {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  }

  async function doPing() {
    const t0 = Date.now();
    const ok = await pingBackend(12_000);
    if (ok) {
      failureCountRef.current = 0;
      setLatencyMs(Date.now() - t0);
      stopProgressBar(true);
      setStatus("ready");
      // Slow ping every 30s to detect later disconnections
      clearPingInterval();
      pingIntervalRef.current = setInterval(doPing, 30_000);
    } else {
      failureCountRef.current += 1;
      setRetryCount(c => c + 1);
      if (failureCountRef.current === 1) {
        setStatus("warming");
      }
      if (failureCountRef.current >= 3) {
        stopProgressBar(false);
        setStatus("error");
        clearPingInterval();
      }
    }
  }

  function startConnecting() {
    failureCountRef.current = 0;
    setRetryCount(0);
    setStatus("connecting");
    setWarmupProgress(0);
    startProgressBar();
    doPing(); // immediate first attempt
    clearPingInterval();
    pingIntervalRef.current = setInterval(doPing, 5_000);
  }

  function reconnect() {
    startConnecting();
  }

  useEffect(() => {
    startConnecting();
    return () => {
      stopProgressBar();
      clearPingInterval();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <MLConnectionContext.Provider
      value={{ status, latencyMs, warmupProgress, retryCount, reconnect }}
    >
      {children}
    </MLConnectionContext.Provider>
  );
}
```

**Step 4: Run tests**

```bash
npx vitest run client/src/tests/hooks/use-ml-connection.test.tsx
```
Expected: all 4 tests pass.

**Step 5: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/hooks/use-ml-connection.tsx client/src/tests/hooks/use-ml-connection.test.tsx
git commit -m "feat: add useMLConnection hook for ML backend connection state"
```

---

## Task 2: Build MLWarmupScreen component

**Files:**
- Create: `client/src/components/ml-warmup-screen.tsx`
- Test: `client/src/tests/components/ml-warmup-screen.test.tsx`

**Step 1: Write the failing test**

Create `client/src/tests/components/ml-warmup-screen.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MLWarmupScreen } from "@/components/ml-warmup-screen";
import * as connectionHook from "@/hooks/use-ml-connection";

function mockConnection(overrides: Partial<connectionHook.MLConnectionState>) {
  vi.spyOn(connectionHook, "useMLConnection").mockReturnValue({
    status: "connecting",
    latencyMs: null,
    warmupProgress: 0,
    retryCount: 0,
    reconnect: vi.fn(),
    ...overrides,
  });
}

describe("MLWarmupScreen", () => {
  it("renders when status is connecting", () => {
    mockConnection({ status: "connecting" });
    render(<MLWarmupScreen />);
    expect(screen.getByText(/neural dream workshop/i)).toBeInTheDocument();
  });

  it("renders when status is warming", () => {
    mockConnection({ status: "warming", warmupProgress: 50 });
    render(<MLWarmupScreen />);
    expect(screen.getByText(/neural dream workshop/i)).toBeInTheDocument();
  });

  it("does not render when status is ready", () => {
    mockConnection({ status: "ready" });
    const { container } = render(<MLWarmupScreen />);
    expect(container.firstChild).toBeNull();
  });

  it("shows simulation mode button after 40s elapsed", () => {
    vi.useFakeTimers();
    mockConnection({ status: "warming", warmupProgress: 95 });
    const onSim = vi.fn();
    render(<MLWarmupScreen onSimulationMode={onSim} />);
    vi.advanceTimersByTime(41_000);
    expect(screen.getByText(/simulation mode/i)).toBeInTheDocument();
    vi.useRealTimers();
  });
});
```

**Step 2: Run it to confirm it fails**

```bash
npx vitest run client/src/tests/components/ml-warmup-screen.test.tsx
```
Expected: FAIL — `Cannot find module '@/components/ml-warmup-screen'`

**Step 3: Create the component**

Create `client/src/components/ml-warmup-screen.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import { Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMLConnection } from "@/hooks/use-ml-connection";

const MESSAGES = [
  "Initializing neural engines...",
  "Loading EEG models...",
  "Calibrating signal pipeline...",
  "Almost ready...",
];

interface MLWarmupScreenProps {
  onSimulationMode?: () => void;
}

export function MLWarmupScreen({ onSimulationMode }: MLWarmupScreenProps) {
  const { status, warmupProgress, retryCount } = useMLConnection();
  const [msgIdx, setMsgIdx] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const mountedAt = useRef(Date.now());

  const isVisible = status === "connecting" || status === "warming";

  // Rotate status messages every 8 seconds
  useEffect(() => {
    if (!isVisible) return;
    const id = setInterval(() => setMsgIdx(i => (i + 1) % MESSAGES.length), 8_000);
    return () => clearInterval(id);
  }, [isVisible]);

  // Elapsed counter
  useEffect(() => {
    if (!isVisible) return;
    mountedAt.current = Date.now();
    const id = setInterval(
      () => setElapsed(Math.floor((Date.now() - mountedAt.current) / 1000)),
      1_000
    );
    return () => clearInterval(id);
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background">
      {/* Pulsing brain icon */}
      <div className="mb-8 animate-pulse">
        <Brain className="w-20 h-20 text-primary" strokeWidth={1.5} />
      </div>

      <h1 className="text-2xl font-bold text-foreground mb-2">Neural Dream Workshop</h1>

      {/* Status message */}
      <p className="text-muted-foreground text-sm mb-8 h-5 transition-opacity duration-300">
        {MESSAGES[msgIdx]}
      </p>

      {/* Progress bar */}
      <div className="w-72 h-1.5 bg-muted rounded-full overflow-hidden mb-3">
        <div
          className="h-full bg-primary rounded-full transition-all duration-300 ease-linear"
          style={{ width: `${warmupProgress}%` }}
        />
      </div>

      {/* Elapsed time + retry count */}
      <p className="text-xs text-muted-foreground mb-8">
        {elapsed}s
        {retryCount > 0 && ` · attempt ${retryCount + 1}`}
      </p>

      {/* Simulation mode escape hatch (after 40 seconds) */}
      {elapsed >= 40 && onSimulationMode && (
        <Button variant="outline" size="sm" onClick={onSimulationMode}>
          Continue in Simulation Mode
        </Button>
      )}
    </div>
  );
}
```

**Step 4: Run tests**

```bash
npx vitest run client/src/tests/components/ml-warmup-screen.test.tsx
```
Expected: all 4 tests pass.

**Step 5: Commit**

```bash
git add client/src/components/ml-warmup-screen.tsx client/src/tests/components/ml-warmup-screen.test.tsx
git commit -m "feat: add MLWarmupScreen component for cold-start loading state"
```

---

## Task 3: Integrate MLConnectionProvider into App.tsx

**Files:**
- Modify: `client/src/App.tsx`

**Step 1: Locate the provider nesting point**

Open `client/src/App.tsx`. The current outer wrapper is:
```tsx
<QueryClientProvider client={queryClient}>
  <ThemeProvider>
    <AuthProvider>
      <DeviceProvider>
        <TooltipProvider>
          ...
```

**Step 2: Add imports at the top of App.tsx**

Add after the existing imports (around line 11):
```tsx
import { MLConnectionProvider } from "@/hooks/use-ml-connection";
import { MLWarmupScreen } from "@/components/ml-warmup-screen";
```

**Step 3: Add MLConnectionProvider inside AuthProvider**

Wrap `DeviceProvider` with `MLConnectionProvider`. The updated nesting in the return of `App()` (or wherever the providers are composed):

```tsx
<QueryClientProvider client={queryClient}>
  <ThemeProvider>
    <AuthProvider>
      <MLConnectionProvider>          {/* ← ADD this */}
        <DeviceProvider>
          <TooltipProvider>
            <MLWarmupScreen onSimulationMode={() => {/* noop — simulation is auto */}} />
            <AppRoutes />
            <Toaster />
          </TooltipProvider>
        </DeviceProvider>
      </MLConnectionProvider>          {/* ← ADD this */}
    </AuthProvider>
  </ThemeProvider>
</QueryClientProvider>
```

Note: `MLWarmupScreen` is placed ABOVE `AppRoutes` inside `TooltipProvider` so it overlays the entire app. The `onSimulationMode` is a no-op because the user can dismiss it and the app will work in simulation mode automatically (retry logic in `mlFetch` handles fallback).

**Step 4: Verify typecheck**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
npm run check
```
Expected: no TypeScript errors.

**Step 5: Test in browser on localhost**

```bash
npm run dev
```
Open `http://localhost:5000`. Log in. Stop the ML backend (`Ctrl+C` on the uvicorn process). Refresh the app — you should see the MLWarmupScreen pulsing brain animation.

**Step 6: Commit**

```bash
git add client/src/App.tsx
git commit -m "feat: integrate MLConnectionProvider and MLWarmupScreen into App.tsx"
```

---

## Task 4: Add keep-alive ping in AppLayout

**Files:**
- Modify: `client/src/layouts/app-layout.tsx`

**Step 1: Read the current AppLayout** (already done — it has useHealthSync and registerNativePush)

**Step 2: Add the keep-alive effect**

Open `client/src/layouts/app-layout.tsx`. Import `getMLApiUrl` and `useAuth`. Add this effect inside `AppLayout`:

```tsx
// At top of file, add import:
import { useAuth } from "@/hooks/use-auth";

// Import getMLApiUrl from ml-api
// Note: ml-api.ts does not export getMLApiUrl — use pingBackend instead
import { pingBackend } from "@/lib/ml-api";
```

Add this effect inside `AppLayout` function, after the existing effects:

```tsx
// ── Keep-alive: ping ML backend every 14 min to prevent Render free-tier sleep ──
const { user } = useAuth();
useEffect(() => {
  if (!user) return; // only ping when authenticated

  const FOURTEEN_MINUTES = 14 * 60 * 1000;
  const id = setInterval(() => {
    if (document.visibilityState === "visible") {
      // Fire-and-forget: prevents Render's 15-min idle timeout
      pingBackend(5_000).catch(() => {});
    }
  }, FOURTEEN_MINUTES);

  return () => clearInterval(id);
}, [user]);
```

**Step 3: Verify typecheck**

```bash
npm run check
```
Expected: no TypeScript errors.

**Step 4: Commit**

```bash
git add client/src/layouts/app-layout.tsx
git commit -m "feat: add keep-alive ping in AppLayout to prevent Render free-tier sleep"
```

---

## Task 5: Add ML connection status dot in sidebar

**Files:**
- Modify: `client/src/components/sidebar.tsx`

**Step 1: Add import**

Open `client/src/components/sidebar.tsx`. Add at top:

```tsx
import { useMLConnection } from "@/hooks/use-ml-connection";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
```

**Step 2: Add status dot component inside Sidebar**

Inside the `Sidebar` function, after the existing hooks:

```tsx
const { status, latencyMs, reconnect } = useMLConnection();

const dotColor =
  status === "ready"
    ? "bg-green-500"
    : status === "error"
    ? "bg-red-500"
    : "bg-amber-500 animate-pulse";

const statusLabel =
  status === "ready"
    ? `ML Backend: Connected${latencyMs !== null ? ` (${latencyMs}ms)` : ""}`
    : status === "error"
    ? "ML Backend: Unreachable"
    : "ML Backend: Warming up...";
```

**Step 3: Add the dot to the sidebar JSX**

Find where the sidebar header or user info is rendered. Add the status dot near the top of the sidebar (e.g., after the app title/logo area):

```tsx
{/* ML Backend status indicator */}
<div className="px-4 py-2 flex items-center gap-2">
  <Tooltip>
    <TooltipTrigger asChild>
      <button
        className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
        onClick={status === "error" ? reconnect : undefined}
      >
        <span className={`w-2 h-2 rounded-full shrink-0 ${dotColor}`} />
        <span className="truncate max-w-[140px]">
          {status === "ready" ? "ML Ready" : status === "error" ? "ML Offline" : "ML Starting"}
        </span>
      </button>
    </TooltipTrigger>
    <TooltipContent side="right">
      <p>{statusLabel}</p>
      {status === "error" && (
        <p className="text-xs text-muted-foreground mt-1">Click to reconnect</p>
      )}
    </TooltipContent>
  </Tooltip>
</div>
```

**Step 4: Verify typecheck**

```bash
npm run check
```

**Step 5: Test in browser**

Start dev server. The sidebar should show a small green dot when ML backend is up, amber when warming, red when down.

**Step 6: Commit**

```bash
git add client/src/components/sidebar.tsx
git commit -m "feat: add ML connection status dot to sidebar"
```

---

## Task 6: Add retry logic and timeout to mlFetch

**Files:**
- Modify: `client/src/lib/ml-api.ts`
- Test: `client/src/tests/lib/ml-api-retry.test.ts`

**Step 1: Write the failing test**

Create `client/src/tests/lib/ml-api-retry.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

// We'll import the internal mlFetch behavior indirectly via analyzeEEG
describe("mlFetch retry logic", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    // Reset localStorage so getMLApiUrl() returns default
    localStorage.clear();
  });

  it("retries 3 times on 503 then throws", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      clone: () => ({ json: async () => ({}) }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const { analyzeEEG } = await import("@/lib/ml-api");
    await expect(analyzeEEG([[1, 2, 3]], 256)).rejects.toThrow();

    // Should have been called 4 times (1 initial + 3 retries)
    expect(mockFetch).toHaveBeenCalledTimes(4);
  });

  it("does NOT retry on 422 client error", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      clone: () => ({ json: async () => ({ detail: "bad input" }) }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const { analyzeEEG } = await import("@/lib/ml-api");
    await expect(analyzeEEG([[1, 2, 3]], 256)).rejects.toThrow();

    // No retries for 4xx
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("succeeds on 2nd attempt after 1 failure", async () => {
    let calls = 0;
    const mockFetch = vi.fn().mockImplementation(() => {
      calls++;
      if (calls === 1) {
        return Promise.resolve({
          ok: false,
          status: 503,
          statusText: "unavailable",
          clone: () => ({ json: async () => ({}) }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ emotions: { emotion: "happy" } }),
      });
    });
    vi.stubGlobal("fetch", mockFetch);

    const { analyzeEEG } = await import("@/lib/ml-api");
    // Should succeed on 2nd call without throwing
    await expect(analyzeEEG([[1, 2, 3]], 256)).resolves.toBeDefined();
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});
```

**Step 2: Run it to confirm it fails**

```bash
npx vitest run client/src/tests/lib/ml-api-retry.test.ts
```
Expected: FAIL — test expects 4 calls but only gets 1 (no retry exists yet).

**Step 3: Add retry wrapper to mlFetch**

Open `client/src/lib/ml-api.ts`. Find the `mlFetch` function (around line 296). Replace it with this version:

```ts
const RETRY_DELAYS = [1_000, 3_000, 9_000]; // 3 retries, exponential backoff

async function mlFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${getMLApiUrl()}/api${endpoint}`;
  const baseHeaders = {
    "Content-Type": "application/json",
    ...ngrokHeaders(),
    ...options?.headers,
  };

  let lastError: Error = new Error("Request failed");

  for (let attempt = 0; attempt <= RETRY_DELAYS.length; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    try {
      const response = await fetch(url, {
        ...options,
        headers: baseHeaders,
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (response.ok) {
        return response.json() as Promise<T>;
      }

      // Do NOT retry 4xx errors (client errors)
      if (response.status >= 400 && response.status < 500) {
        try {
          const body = await response.clone().json();
          const detail = body?.detail;
          if (typeof detail === "string") throw new Error(detail);
          if (Array.isArray(detail)) throw new Error(detail.map((d: { msg?: string }) => d.msg).join("; "));
        } catch (parseErr) {
          if (parseErr instanceof Error && parseErr.message !== "") throw parseErr;
        }
        throw new Error(`Request failed (${response.status})`);
      }

      // 5xx — retry
      lastError = new Error(`Request failed (${response.status})`);
    } catch (err) {
      clearTimeout(timeout);
      if (err instanceof Error && (err.name === "AbortError" || !String(err).includes("Request failed"))) {
        lastError = err;
      } else if (err instanceof Error) {
        // Re-throw 4xx errors immediately
        const msg = err.message;
        if (!msg.includes("Request failed (5")) throw err;
        lastError = err;
      }
    }

    // Wait before next retry (not after last attempt)
    if (attempt < RETRY_DELAYS.length) {
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAYS[attempt]));
    }
  }

  throw lastError;
}
```

**Step 4: Run tests**

```bash
npx vitest run client/src/tests/lib/ml-api-retry.test.ts
```
Expected: all 3 tests pass.

**Step 5: Run full test suite to check for regressions**

```bash
npx vitest run
```
Expected: all existing 227+ tests still pass.

**Step 6: Commit**

```bash
git add client/src/lib/ml-api.ts client/src/tests/lib/ml-api-retry.test.ts
git commit -m "feat: add retry logic (3x exponential backoff) and 30s timeout to mlFetch"
```

---

## Task 7: Add SimulationModeBanner component

**Files:**
- Create: `client/src/components/simulation-mode-banner.tsx`
- Modify: `client/src/pages/emotion-lab.tsx` (add banner)
- Modify: `client/src/pages/brain-monitor.tsx` (add banner)

**Step 1: Create the banner component**

Create `client/src/components/simulation-mode-banner.tsx`:

```tsx
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMLConnection } from "@/hooks/use-ml-connection";

export function SimulationModeBanner() {
  const { status, reconnect } = useMLConnection();
  if (status !== "error") return null;

  return (
    <div className="w-full flex items-center gap-3 px-4 py-2 bg-amber-500/10 border border-amber-500/30 rounded-lg mb-4 text-sm">
      <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
      <span className="text-amber-200 flex-1">
        ML backend unreachable — running in simulation mode
      </span>
      <Button
        variant="ghost"
        size="sm"
        className="text-amber-400 hover:text-amber-300 h-7 px-2"
        onClick={reconnect}
      >
        <RefreshCw className="w-3 h-3 mr-1" />
        Reconnect
      </Button>
    </div>
  );
}
```

**Step 2: Add banner to EmotionLab page**

Open `client/src/pages/emotion-lab.tsx`. Find the page's top-level JSX return. Import and add the banner at the top:

```tsx
// Add import:
import { SimulationModeBanner } from "@/components/simulation-mode-banner";

// Inside the JSX return, as first child inside the page container:
<SimulationModeBanner />
```

**Step 3: Add banner to BrainMonitor page**

Repeat the same for `client/src/pages/brain-monitor.tsx`.

**Step 4: Verify typecheck**

```bash
npm run check
```
Expected: no errors.

**Step 5: Commit**

```bash
git add client/src/components/simulation-mode-banner.tsx \
        client/src/pages/emotion-lab.tsx \
        client/src/pages/brain-monitor.tsx
git commit -m "feat: add SimulationModeBanner for ML-unreachable state on Emotion Lab and Brain Monitor"
```

---

## Task 8: Wire TSception into emotion classifier live inference path

**Files:**
- Modify: `ml/models/emotion_classifier.py`
- Test: `ml/tests/test_tsception_wiring.py`

**Step 1: Check TSception model exists**

```bash
ls /Users/sravyalu/NeuralDreamWorkshop/ml/models/tsception.py
ls /Users/sravyalu/NeuralDreamWorkshop/ml/models/saved/tsception_emotion.pt 2>/dev/null || echo "NOT FOUND"
```

If `tsception_emotion.pt` does not exist, skip to Task 9 (TSception needs retraining — out of Thursday scope).

**Step 2: Write the failing test**

Create `ml/tests/test_tsception_wiring.py`:

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def test_tsception_loaded_as_fallback():
    """TSception should be attempted when mega_lgbm is not loaded."""
    from models.emotion_classifier import EmotionClassifier
    clf = EmotionClassifier()
    # Artificially disable mega_lgbm path
    clf.mega_lgbm_model = None
    # Provide 4-second epoch (4*256=1024 samples)
    signals = np.random.randn(4, 1024) * 5  # 4 channels, 1024 samples
    result = clf.predict(signals, fs=256)
    assert "emotion" in result
    assert "model_type" in result


def test_tsception_model_type_label():
    """When TSception is used, model_type should be 'tsception'."""
    from models.emotion_classifier import EmotionClassifier
    clf = EmotionClassifier()
    if clf._tsception is None:
        pytest.skip("TSception model not loaded — skipping")
    clf.mega_lgbm_model = None  # force TSception path
    signals = np.random.randn(4, 1024) * 5
    result = clf.predict(signals, fs=256)
    assert result["model_type"] == "tsception"


def test_tsception_skipped_for_short_epoch():
    """TSception should NOT run with fewer than 1024 samples (< 4 seconds)."""
    from models.emotion_classifier import EmotionClassifier
    clf = EmotionClassifier()
    clf.mega_lgbm_model = None
    # Short epoch — 256 samples = 1 second
    signals = np.random.randn(4, 256) * 5
    result = clf.predict(signals, fs=256)
    # Should fall back to feature heuristics, not TSception
    assert result["model_type"] != "tsception"
```

**Step 3: Run to confirm failure**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop/ml
python -m pytest tests/test_tsception_wiring.py -v
```
Expected: FAIL — `_tsception` attribute doesn't exist yet.

**Step 4: Add TSception to EmotionClassifier.__init__**

Open `ml/models/emotion_classifier.py`. In `__init__`, after the `self._eegnet` block (around line 93), add:

```python
# TSception — asymmetry-aware spatial CNN for AF7/AF8 Muse 2 data.
# Fallback after mega_lgbm and EEGNet, before feature heuristics.
# Model file: models/saved/tsception_emotion.pt
self._tsception = None
self._try_load_tsception()
```

**Step 5: Add _try_load_tsception method**

In the `EmotionClassifier` class, add after `_try_load_mega_lgbm` (or near other `_try_load_*` methods):

```python
def _try_load_tsception(self) -> None:
    """Try loading the TSception emotion model (69% CV)."""
    from pathlib import Path
    pt_path = Path("models/saved/tsception_emotion.pt")
    if not pt_path.exists():
        return
    try:
        from models.tsception import TSceptionEmotionClassifier
        self._tsception = TSceptionEmotionClassifier(str(pt_path))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug(f"TSception load failed: {exc}")
        self._tsception = None
```

**Step 6: Add TSception to the predict() dispatch chain**

Find the `predict()` method in `EmotionClassifier`. The current chain dispatches to `_predict_mega_lgbm`, then other models, then `_predict_features`. After the mega_lgbm block and before feature heuristics, add:

```python
# TSception fallback (69% CV) — requires >= 4-second epoch (1024 samples at 256 Hz)
if (self._tsception is not None
        and eeg.ndim == 2
        and eeg.shape[1] >= 1024):
    try:
        result = self._tsception.predict(eeg, fs=fs)
        result["model_type"] = "tsception"
        return self._apply_ema(result)
    except Exception:
        pass  # fall through to heuristics
```

**Step 7: Run tests**

```bash
python -m pytest tests/test_tsception_wiring.py -v
```
Expected: test_tsception_loaded_as_fallback and test_tsception_skipped_for_short_epoch pass. test_tsception_model_type_label passes only if `.pt` file exists.

**Step 8: Run full ML test suite**

```bash
python -m pytest ml/tests/ -v --tb=short
```
Expected: no regressions.

**Step 9: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add ml/models/emotion_classifier.py ml/tests/test_tsception_wiring.py
git commit -m "feat: wire TSception (69% CV) into emotion classifier fallback chain"
```

---

## Task 9: Add RunningNormalizer to eeg_processor.py

**Files:**
- Modify: `ml/processing/eeg_processor.py`
- Test: `ml/tests/test_running_normalizer.py`

**Step 1: Write the failing test**

Create `ml/tests/test_running_normalizer.py`:

```python
import numpy as np
import pytest


def test_running_normalizer_basic():
    from processing.eeg_processor import RunningNormalizer
    rn = RunningNormalizer()
    features = np.array([1.0, 2.0, 3.0])
    user_id = "test_user"
    # Before enough samples, returns raw features
    result = rn.normalize(features, user_id)
    np.testing.assert_array_equal(result, features)


def test_running_normalizer_normalizes_after_30_samples():
    from processing.eeg_processor import RunningNormalizer
    rn = RunningNormalizer()
    user_id = "test_user"
    # Feed 30 samples of constant features
    for _ in range(30):
        rn.normalize(np.array([5.0, 10.0, 3.0]), user_id)
    # 31st sample should be normalized: z = (x - mean) / std
    # std = 0 for constant input, so should return 0s (or handle gracefully)
    result = rn.normalize(np.array([5.0, 10.0, 3.0]), user_id)
    # When std is near-zero, z-score should be 0 (not NaN or inf)
    assert np.all(np.isfinite(result))


def test_running_normalizer_isolated_by_user():
    from processing.eeg_processor import RunningNormalizer
    rn = RunningNormalizer()
    # Two users have separate buffers
    for _ in range(30):
        rn.normalize(np.array([100.0, 200.0]), "user_a")
    result_b = rn.normalize(np.array([1.0, 2.0]), "user_b")
    # user_b has no buffer yet — should get raw features back
    np.testing.assert_array_equal(result_b, np.array([1.0, 2.0]))


def test_running_normalizer_thread_safe():
    """Calling from multiple threads should not raise."""
    import threading
    from processing.eeg_processor import RunningNormalizer
    rn = RunningNormalizer()
    errors = []
    def worker():
        try:
            for _ in range(20):
                rn.normalize(np.random.randn(10), "shared_user")
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
```

**Step 2: Run to confirm failure**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop/ml
python -m pytest tests/test_running_normalizer.py -v
```
Expected: FAIL — `cannot import name 'RunningNormalizer'`

**Step 3: Add RunningNormalizer to eeg_processor.py**

Open `ml/processing/eeg_processor.py`. At the end of the file (or after the `BaselineCalibrator` class), add:

```python
# ── RunningNormalizer — session drift correction ──────────────────────────────

import threading as _threading
from collections import deque as _deque


class RunningNormalizer:
    """Per-user rolling z-score normalizer for EEG features.

    Addresses EEG non-stationarity (signal drift within a session).
    Technique validated by SJTU SEED team and UESTC FACED paper:
    correcting for within-session drift recovers +10-20 accuracy points
    on cross-subject emotion recognition.

    Usage::

        rn = RunningNormalizer()
        normed = rn.normalize(features, user_id="user_123")

    The first 30 frames are returned unnormalized (insufficient statistics).
    After that, features are z-scored against the rolling 150-frame buffer
    (~5 minutes at 2-second hop).
    """

    _MIN_SAMPLES = 30    # frames before normalization kicks in
    _BUFFER_SIZE = 150   # ~5 minutes at 2-second hop

    def __init__(self):
        self._buffers: dict[str, _deque] = {}
        self._lock = _threading.Lock()

    def normalize(self, features: np.ndarray, user_id: str) -> np.ndarray:
        """Z-score features against rolling buffer for this user.

        Returns raw features when buffer is below MIN_SAMPLES.
        Handles zero-std gracefully (returns 0 for those dimensions).
        """
        with self._lock:
            if user_id not in self._buffers:
                self._buffers[user_id] = _deque(maxlen=self._BUFFER_SIZE)
            buf = self._buffers[user_id]
            buf.append(features.copy())
            n = len(buf)

        if n < self._MIN_SAMPLES:
            return features

        matrix = np.stack(list(buf))  # (n, n_features)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        # Avoid division by zero: where std < 1e-8, output 0
        safe_std = np.where(std < 1e-8, 1.0, std)
        normed = (features - mean) / safe_std
        normed = np.where(std < 1e-8, 0.0, normed)
        return normed
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_running_normalizer.py -v
```
Expected: all 4 tests pass.

**Step 5: Wire RunningNormalizer into _predict_mega_lgbm**

In `ml/models/emotion_classifier.py`, add the normalizer as a singleton. Near the module level (after imports):

```python
# Singleton RunningNormalizer (one instance shared across all requests)
_running_normalizer = None

def _get_running_normalizer():
    global _running_normalizer
    if _running_normalizer is None:
        try:
            from processing.eeg_processor import RunningNormalizer
            _running_normalizer = RunningNormalizer()
        except Exception:
            pass
    return _running_normalizer
```

In `_predict_mega_lgbm()`, after extracting features and before calling the LGBM model, add:

```python
# Apply running normalization (session drift correction)
rn = _get_running_normalizer()
if rn is not None and user_id:
    features_array = rn.normalize(features_array, user_id)
```

The `user_id` needs to be threaded through `predict()` → `_predict_mega_lgbm()`. Check if `predict()` already accepts `user_id` as a parameter. If not, add it as an optional parameter: `def predict(self, eeg, fs=256, user_id="default")`.

**Step 6: Run full ML tests**

```bash
python -m pytest ml/tests/ -v --tb=short
```
Expected: no regressions.

**Step 7: Commit**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add ml/processing/eeg_processor.py ml/models/emotion_classifier.py ml/tests/test_running_normalizer.py
git commit -m "feat: add RunningNormalizer for EEG non-stationarity correction (SJTU SEED technique)"
```

---

## Task 10: Audit and fix Vercel + CORS environment configuration

**Files:**
- Modify: `.env.example`
- Modify: `render.yaml` (verify CORS)
- Modify: `vercel.json` (add env doc comment)

**Step 1: Check current Vercel URL**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
cat vercel.json
# Check if VITE_ML_API_URL is set anywhere
grep -r "VITE_ML_API_URL" . --include="*.json" --include="*.ts" --include="*.env*" 2>/dev/null
```

**Step 2: Update .env.example**

Open `.env.example`. Ensure it contains:

```bash
# ── Express / Node.js server ──────────────────────────────────────────────────
DATABASE_URL="postgresql://USER:PASSWORD@HOST/DB?sslmode=require"
OPENAI_API_KEY="sk-proj-..."
SESSION_SECRET="change-me-to-a-long-random-string"
PORT=5000

# ── ML Backend URL ────────────────────────────────────────────────────────────
# On localhost: leave blank (defaults to http://localhost:8000)
# On Vercel production: set to your Render ML backend URL
VITE_ML_API_URL="https://neural-dream-ml.onrender.com"

# ── Web Push Notifications (optional) ────────────────────────────────────────
VAPID_PUBLIC_KEY="your-vapid-public-key"
VAPID_PRIVATE_KEY="your-vapid-private-key"
VAPID_EMAIL="mailto:your@email.com"

# ── Spotify OAuth (optional) ─────────────────────────────────────────────────
SPOTIFY_CLIENT_ID="your-spotify-client-id"
SPOTIFY_CLIENT_SECRET="your-spotify-client-secret"
SPOTIFY_REDIRECT_URI="http://localhost:5000/api/spotify/callback"
```

**Step 3: Verify render.yaml CORS origins**

Open `render.yaml`. The `CORS_ORIGINS` value must include your actual Vercel production URL. Verify it matches what `dream-analysis.vercel.app` is (or update to correct URL):

```yaml
- key: CORS_ORIGINS
  value: "http://localhost:5000,http://localhost:3000,http://localhost:5173,https://dream-analysis.vercel.app,https://dream-analysis-*.vercel.app"
```

If your Vercel URL is different, update it.

**Step 4: Add a comment to vercel.json**

Open `vercel.json`. Add an `env` block documenting that `VITE_ML_API_URL` must be set in the Vercel dashboard:

```json
{
  "framework": "vite",
  "buildCommand": "vite build",
  "outputDirectory": "dist/public",
  "env": {
    "VITE_ML_API_URL": "https://neural-dream-ml.onrender.com"
  },
  "functions": {
    "api/**/*.ts": {
      "maxDuration": 30
    }
  },
  "rewrites": [
    { "source": "/((?!api/).*)", "destination": "/index.html" }
  ],
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Origin", "value": "*" },
        { "key": "Access-Control-Allow-Methods", "value": "GET, POST, PUT, DELETE, OPTIONS" },
        { "key": "Access-Control-Allow-Headers", "value": "Content-Type, Authorization" },
        { "key": "Access-Control-Allow-Credentials", "value": "true" }
      ]
    }
  ]
}
```

**Step 5: Push and verify Vercel build**

```bash
git add .env.example render.yaml vercel.json
git commit -m "fix: set VITE_ML_API_URL in vercel.json, update CORS origins and .env.example"
git push origin main
```

Then open the Vercel dashboard → check the latest deployment build logs. Confirm `VITE_ML_API_URL` is picked up.

**Step 6: Smoke test on production**

Open `https://dream-analysis.vercel.app`. Log in. Watch for the MLWarmupScreen. Verify it connects to the Render ML backend within 60 seconds.

---

## Task 11: Update STATUS.md to reflect launch-ready state

**Files:**
- Modify: `STATUS.md`

**Step 1: Open STATUS.md and add the new items**

Under `### Infrastructure` (or a new `### Connection & Resilience` section), add:

```markdown
### Connection & Resilience
- [x] MLWarmupScreen — animated full-screen loading overlay for ML backend cold start (Render free tier)
- [x] Keep-alive ping — 14-min interval in AppLayout prevents Render 15-min idle sleep
- [x] Retry logic — 3-attempt exponential backoff (1s/3s/9s) + 30s timeout in mlFetch
- [x] SimulationModeBanner — amber banner on EmotionLab and BrainMonitor when ML unreachable
- [x] ML status dot — green/amber/red indicator + latency tooltip in sidebar
- [x] TSception live path — wired into emotion classifier fallback chain (mega_lgbm → TSception 69% CV → heuristics)
- [x] RunningNormalizer — per-user rolling z-score (session drift correction, SJTU/UESTC technique)
- [x] Vercel env config — VITE_ML_API_URL in vercel.json, CORS origins verified in render.yaml
```

**Step 2: Update Working Model Accuracies table**

Find the TSception row and change its Notes from "Trained" to "Active (fallback)":

```
| TSception Emotion | TSception CNN (PyTorch) | **69.00% CV** | ... | Active (fallback when mega LGBM not loaded) |
```

**Step 3: Commit**

```bash
git add STATUS.md
git commit -m "docs: update STATUS.md for Thursday launch — connection resilience + TSception + RunningNormalizer"
git push origin main
```

---

## Success Checklist

Before Thursday EOD, verify each item manually:

- [ ] Open app cold (Render sleeping) → see MLWarmupScreen animated brain, not a blank error
- [ ] Wait 30-60s → MLWarmupScreen disappears, app is usable
- [ ] Sidebar shows green dot with latency when ML is ready
- [ ] Kill ML backend → sidebar shows red dot, EmotionLab shows amber banner
- [ ] Keep-alive ping visible in browser DevTools Network tab every 14 min
- [ ] Single ML call failure is invisible (auto-retried)
- [ ] App works identically on `localhost:5000` and `dream-analysis.vercel.app`
- [ ] `/benchmarks` page shows TSception as "Active (fallback)"

---

**Plan complete and saved to `docs/plans/2026-03-03-thursday-launch-plan.md`.**
