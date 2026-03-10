# Fix GATT Connection, TypeScript Errors, and Failing Tests

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the Muse 2 BLE GATT connection failure, resolve 9 TypeScript errors, and fix 25 failing tests.

**Architecture:** Three independent fixes: (1) improve GATT retry logic with longer delays and stale device cleanup, (2) add Web Bluetooth type definitions, (3) update tests to match current component behavior.

**Tech Stack:** TypeScript, Web Bluetooth API, Vitest, React Testing Library

---

### Task 1: Fix GATT Connection Retry Logic

**Files:**
- Modify: `client/src/lib/muse-ble.ts:318-336`

**Step 1: Update `_connectWebBluetooth()` retry logic**

Replace the single-retry with a multi-retry loop (1.5s, 3s delays) and add `device.forget()` for stale GATT cleanup:

```typescript
// Replace lines 318-336 with:

    // If Muse is already paired at OS level, gatt.connect() may fail.
    // The OS GATT stack needs 1.5-3s to fully release resources after disconnect.
    let server: BluetoothRemoteGATTServer;
    const RETRY_DELAYS = [1500, 3000];
    let lastErr: unknown;

    // First attempt
    try {
      server = await device.gatt!.connect();
      this._webGattServer = server;
    } catch (firstErr) {
      lastErr = firstErr;

      // Clear stale OS-level GATT lock
      try { device.gatt!.disconnect(); } catch {}
      // If available, forget() releases the OS pairing entirely
      if (typeof (device as any).forget === "function") {
        try { await (device as any).forget(); } catch {}
        // Re-request after forget — picker will show again
        try {
          const bt = (navigator as Navigator & { bluetooth: Bluetooth }).bluetooth;
          device = await bt.requestDevice({
            filters: [{ services: [MUSE_SERVICE] }],
            optionalServices: [MUSE_SERVICE],
          });
          this.deviceName = device.name ?? "Muse";
        } catch (e) {
          this.setStatus("error", "Device selection cancelled after GATT reset");
          throw e;
        }
      }

      // Retry with increasing delays
      for (const delay of RETRY_DELAYS) {
        await new Promise((r) => setTimeout(r, delay));
        try {
          server = await device.gatt!.connect();
          this._webGattServer = server;
          lastErr = null;
          break;
        } catch (retryErr) {
          lastErr = retryErr;
          try { device.gatt!.disconnect(); } catch {}
        }
      }

      if (lastErr) {
        const msg = String(lastErr);
        const hint = msg.includes("unknown reason")
          ? "GATT lock held by OS. Open System Bluetooth settings, click (i) next to Muse, tap Forget This Device, then try again."
          : `Connection failed: ${msg}`;
        this.setStatus("error", hint);
        throw lastErr;
      }
    }
```

**Step 2: Verify build still passes**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm run build 2>&1 | tail -5`
Expected: `built in` success message

**Step 3: Commit**

```bash
git add client/src/lib/muse-ble.ts
git commit -m "fix: improve GATT retry with longer delays and stale device cleanup"
```

---

### Task 2: Fix TypeScript Errors — Add Web Bluetooth Types

**Files:**
- Modify: `package.json` (add devDependency)

**Step 1: Install @types/web-bluetooth**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm install --save-dev @types/web-bluetooth`

**Step 2: Verify TypeScript errors are resolved**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit 2>&1 | tail -5`
Expected: No errors (or only unrelated errors, not `BluetoothRemoteGATTCharacteristic` etc.)

**Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "fix: add @types/web-bluetooth to resolve muse-ble.ts type errors"
```

---

### Task 3: Fix ml-warmup-screen Tests

**Files:**
- Modify: `client/src/test/components/ml-warmup-screen.test.tsx`
- Reference: `client/src/components/ml-warmup-screen.tsx`

**Step 1: Read both files, identify the threshold mismatch**

The test advances timers by 40_000ms but the component's `SIMULATION_MODE_THRESHOLD_S` is 10s. The test was written when threshold was 40s and not updated.

**Step 2: Update test to use correct threshold**

Change `vi.advanceTimersByTime(40_000)` to `vi.advanceTimersByTime(10_000)` (or match whatever the component currently uses).

**Step 3: Run the specific test**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/components/ml-warmup-screen.test.tsx 2>&1 | tail -15`
Expected: 2 tests pass

**Step 4: Commit**

```bash
git add client/src/test/components/ml-warmup-screen.test.tsx
git commit -m "fix: update ml-warmup-screen test threshold to match component"
```

---

### Task 4: Fix use-ml-connection Test

**Files:**
- Modify: `client/src/test/hooks/use-ml-connection.test.tsx`
- Reference: `client/src/hooks/use-ml-connection.tsx`

**Step 1: Read both files, identify async timing issue**

The test likely uses `vi.advanceTimersByTimeAsync` incorrectly for the ping interval scheduling.

**Step 2: Fix the async timer advancement to match hook's ping schedule**

Adjust `advanceTimersByTimeAsync` calls to match the actual interval timing in the hook.

**Step 3: Run test**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/hooks/use-ml-connection.test.tsx 2>&1 | tail -15`
Expected: All tests pass

**Step 4: Commit**

```bash
git add client/src/test/hooks/use-ml-connection.test.tsx
git commit -m "fix: update use-ml-connection test async timing"
```

---

### Task 5: Fix ml-api-retry Tests

**Files:**
- Modify: `client/src/test/lib/ml-api-retry.test.ts`
- Reference: `client/src/lib/ml-api.ts`

**Step 1: Read both files, identify retry mock issue**

The test mocks global `fetch` but the retry logic may use a different fetch wrapper or the mock isn't matching the actual retry behavior (AbortController timeout, error class hierarchy).

**Step 2: Fix the fetch mock and retry assertions**

Update mock to properly handle the `NonRetryableError` class and `AbortController` usage.

**Step 3: Run test**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/lib/ml-api-retry.test.ts 2>&1 | tail -15`
Expected: All tests pass

**Step 4: Commit**

```bash
git add client/src/test/lib/ml-api-retry.test.ts
git commit -m "fix: update ml-api-retry test mocks to match current retry logic"
```

---

### Task 6: Fix daily-brain-report Test

**Files:**
- Modify: `client/src/test/pages/daily-brain-report.test.tsx`
- Reference: `client/src/pages/daily-brain-report.tsx`

**Step 1: Read both files, identify the "Right now" card mismatch**

The test expects specific text that may have been renamed or restructured.

**Step 2: Update test selectors to match current component**

**Step 3: Run test**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/pages/daily-brain-report.test.tsx 2>&1 | tail -15`
Expected: All tests pass

**Step 4: Commit**

```bash
git add client/src/test/pages/daily-brain-report.test.tsx
git commit -m "fix: update daily-brain-report test selectors"
```

---

### Task 7: Fix dashboard Test

**Files:**
- Modify: `client/src/test/pages/dashboard.test.tsx`
- Reference: `client/src/pages/dashboard.tsx`

**Step 1: Read both files — test expects "Brain State Now" text which doesn't exist in component**

This is a critical mismatch. The text was likely renamed.

**Step 2: Find the actual heading text in dashboard.tsx and update the test assertion**

**Step 3: Run test**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/pages/dashboard.test.tsx 2>&1 | tail -15`
Expected: All tests pass

**Step 4: Commit**

```bash
git add client/src/test/pages/dashboard.test.tsx
git commit -m "fix: update dashboard test to match current component text"
```

---

### Task 8: Fix emotion-lab Tests

**Files:**
- Modify: `client/src/test/pages/emotion-lab.test.tsx`
- Reference: `client/src/pages/emotion-lab.tsx`

**Step 1: Read both files, identify 6 failing assertions**

Likely conditional rendering issues — test mocks don't match the current prop/context shape.

**Step 2: Update mocks and assertions for all 6 failing tests**

**Step 3: Run tests**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/pages/emotion-lab.test.tsx 2>&1 | tail -20`
Expected: All tests pass

**Step 4: Commit**

```bash
git add client/src/test/pages/emotion-lab.test.tsx
git commit -m "fix: update emotion-lab tests to match current component"
```

---

### Task 9: Fix food-emotion Tests

**Files:**
- Modify: `client/src/test/pages/food-emotion.test.tsx`
- Reference: `client/src/pages/food-emotion.tsx`

**Step 1: Read both files, identify 12 failing assertions**

Likely data loading timing — the mock query data shape may not match what the component now expects.

**Step 2: Update mocks and assertions for all 12 failing tests**

**Step 3: Run tests**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/pages/food-emotion.test.tsx 2>&1 | tail -20`
Expected: All tests pass

**Step 4: Commit**

```bash
git add client/src/test/pages/food-emotion.test.tsx
git commit -m "fix: update food-emotion tests to match current component"
```

---

### Task 10: Final Verification

**Step 1: Run full test suite**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run 2>&1 | tail -10`
Expected: 0 failures

**Step 2: Run TypeScript check**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit 2>&1 | tail -5`
Expected: No errors

**Step 3: Run build**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npm run build 2>&1 | tail -5`
Expected: Build succeeds
