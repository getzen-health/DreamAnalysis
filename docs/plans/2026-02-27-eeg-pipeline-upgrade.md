# EEG Pipeline Upgrade — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix Muse connection for production, activate Datadog observability, and retrain stress/flow models above the 60% activation threshold.

**Architecture:** Replace dead hardcoded ngrok URL with env-var-driven backend URL; add warm-up ping before Muse pairing; wire ddtrace + Browser RUM for full-stack error visibility; retrain stress and flow models using DREAMER dataset.

**Tech Stack:** React/TypeScript/Vite (frontend), FastAPI/Python (ML backend on Render), ddtrace 2.x, Datadog Browser RUM JS SDK, LightGBM, TSception CNN.

---

## MANUAL PREREQUISITE (do before any code tasks)

### Step M1: Get the Render URL
- Go to https://dashboard.render.com → your `neural-dream-ml` service
- Copy the URL (looks like `https://neural-dream-ml.onrender.com`)
- You'll need this for Task 1 and Task 3

### Step M2: Set VITE_ML_API_URL in Vercel
- Go to https://vercel.com → DreamAnalysis project → Settings → Environment Variables
- Add: `VITE_ML_API_URL` = `https://neural-dream-ml.onrender.com` (your Render URL)
- Apply to: Production, Preview, Development
- Click Save — Vercel will redeploy automatically

### Step M3: Create Datadog account (free tier, 5 min)
- Go to https://app.datadoghq.com/signup → sign up free
- After login: go to **Integrations → API Keys** → create key → copy `DD_API_KEY`
- Go to **RUM → Applications → New Application** → Web → copy `applicationId` and `clientToken`
- You'll paste these into Task 4 and Task 5

---

## Task 1: Fix dead ML backend URL fallback

**Files:**
- Modify: `client/src/lib/ml-api.ts` lines 1-3

**Context:** The hardcoded ngrok URL `https://brendan-postanesthetic-soliloquisingly.ngrok-free.dev` is expired. Users without a saved localStorage URL get a silent network error. Fix: default to `http://localhost:8000` for local dev; production uses `VITE_ML_API_URL`.

**Step 1: Edit the URL default**

In `client/src/lib/ml-api.ts`, change lines 1-3 from:
```typescript
const ML_API_URL_DEFAULT =
  import.meta.env.VITE_ML_API_URL ||
  "https://brendan-postanesthetic-soliloquisingly.ngrok-free.dev";
```

To:
```typescript
const ML_API_URL_DEFAULT =
  import.meta.env.VITE_ML_API_URL ||
  "http://localhost:8000";
```

**Step 2: Verify no other files reference the dead URL**

Run:
```bash
grep -r "brendan-postanesthetic" /Users/sravyalu/NeuralDreamWorkshop/
```
Expected: no output (zero matches)

**Step 3: Typecheck**

Run from project root:
```bash
cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit 2>&1 | grep -v "test/"
```
Expected: no output (no errors)

**Step 4: Commit**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/lib/ml-api.ts
git commit -m "fix: replace dead ngrok fallback URL with localhost:8000"
```

---

## Task 2: Add ML backend warm-up before Muse pair

**Files:**
- Modify: `client/src/lib/ml-api.ts` (add `pingBackend` export near bottom)
- Modify: `client/src/pages/study/StudySession.tsx` lines 493-542 (muse-pair phase)

**Context:** Render free tier sleeps after 15 min idle. Cold start takes 10-30 seconds. Currently the "Pair Muse 2" button hits the backend immediately and shows a cryptic error while Render wakes up. Fix: ping `/health` silently when the muse-pair screen loads; show "Waking up ML backend…" if it takes >1s; enable the Pair button only when ready.

**Step 1: Add pingBackend to ml-api.ts**

At the bottom of `client/src/lib/ml-api.ts` (before the last export line), add:

```typescript
/** Pings ML backend /health with a timeout. Returns true if reachable. */
export async function pingBackend(timeoutMs = 12_000): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(`${getMLApiUrl()}/health`, {
      signal: controller.signal,
      headers: { ...ngrokHeaders() },
    });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}
```

**Step 2: Add backendReady state to StudySession.tsx**

In `client/src/pages/study/StudySession.tsx`, find the existing state declarations (around lines 80-100, where `useState` calls are). Add after the existing state declarations:

```typescript
const [backendReady, setBackendReady]   = useState<boolean | null>(null); // null = checking
```

**Step 3: Add warm-up useEffect to StudySession.tsx**

Find the existing `useEffect` blocks in StudySession.tsx. Add this new effect near the top of the component (after the state declarations, before the phase-advance effects):

```typescript
// Warm up ML backend when muse-pair screen appears
useEffect(() => {
  if (phase !== "muse-pair") return;
  setBackendReady(null);
  pingBackend().then((ok) => setBackendReady(ok));
}, [phase]);
```

Also add the import at the top of StudySession.tsx. Find the existing import from `@/lib/ml-api` (search for `import.*ml-api`) and add `pingBackend` to that import.

**Step 4: Update the muse-pair render section**

In `client/src/pages/study/StudySession.tsx` lines 493-542, replace the muse-pair block with:

```typescript
// Muse pair screen
if (phase === "muse-pair") {
  const isConnecting = deviceState === "connecting";
  const isConnected  = deviceState === "connected" || deviceState === "streaming";
  const backendChecking = backendReady === null;

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="max-w-md w-full space-y-6">
        <div className="text-center space-y-2">
          <Bluetooth className="w-10 h-10 mx-auto text-primary" />
          <h1 className="text-2xl font-bold">Connect your Muse 2</h1>
          <p className="text-sm text-muted-foreground">
            Put on the headband, then tap Pair. The session starts automatically.
          </p>
        </div>

        <Card>
          <CardContent className="pt-6 space-y-4">
            {/* Backend warm-up status */}
            {backendChecking && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                Waking up ML backend…
              </div>
            )}
            {backendReady === false && (
              <p className="text-xs text-amber-400">
                ML backend unreachable. Check your backend URL in Settings or start it locally.
              </p>
            )}

            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Device status</span>
              <Badge variant="outline" className={isConnected ? "border-green-500/50 text-green-400" : "border-muted"}>
                {isConnected ? "Connected" : isConnecting ? "Connecting…" : "Not connected"}
              </Badge>
            </div>

            {!isConnected && (
              <Button
                className="w-full"
                disabled={isConnecting || backendChecking}
                onClick={() => connect("muse_2")}
              >
                {isConnecting
                  ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Connecting…</>
                  : <><Bluetooth className="mr-2 h-4 w-4" />Pair Muse 2</>}
              </Button>
            )}

            {isConnected && isStarting && (
              <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Starting session…
              </div>
            )}
          </CardContent>
        </Card>

        <div className="text-center">
          <button
            className="text-xs text-muted-foreground hover:text-foreground underline"
            onClick={() => { setUseSimulation(true); if (blockType) startSession(blockType); }}
          >
            Continue without Muse (simulation mode)
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Step 5: Typecheck**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit 2>&1 | grep -v "test/"
```
Expected: no errors

**Step 6: Commit**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/src/lib/ml-api.ts client/src/pages/study/StudySession.tsx
git commit -m "feat: warm-up ML backend ping before Muse pair screen"
```

---

## Task 3: Fix Render CORS to allow Vercel frontend

**Files:**
- Modify: `render.yaml` line 17 (CORS_ORIGINS value)

**Context:** The Render envVars only allow `localhost:*`. The production Vercel frontend at `https://dream-analysis.vercel.app` is blocked by CORS. This is a silent failure — the Muse connect request reaches Render but the response is blocked by the browser.

**Step 1: Update render.yaml CORS_ORIGINS**

In `render.yaml`, change line 17 from:
```yaml
        value: http://localhost:5000,http://localhost:3000,http://localhost:5173
```

To:
```yaml
        value: http://localhost:5000,http://localhost:3000,http://localhost:5173,https://dream-analysis.vercel.app,https://dream-analysis-*.vercel.app
```

**Step 2: Commit and push** (Render auto-deploys on push)
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add render.yaml
git commit -m "fix: add Vercel production URL to Render CORS_ORIGINS"
git push
```

**Step 3: Verify Render redeploys**

Go to https://dashboard.render.com → `neural-dream-ml` → check deploy log.
Wait ~3 minutes for Docker build to complete.

---

## Task 4: Add Datadog Browser RUM to frontend

**Files:**
- Modify: `client/index.html` (add RUM SDK in `<head>`)

**Context:** Datadog Browser RUM captures every frontend error, network request failure, and page load — the "eyes" for the frontend. Requires the `applicationId` and `clientToken` from your Datadog account (Step M3 above). These are PUBLIC values (not secrets), safe to commit.

**Step 1: Read current index.html**

Run:
```bash
cat /Users/sravyalu/NeuralDreamWorkshop/client/index.html
```

**Step 2: Add RUM snippet to `<head>`**

Find the `<head>` section and add this block **after the `<meta charset>` line** (replacing `YOUR_APPLICATION_ID` and `YOUR_CLIENT_TOKEN` with values from Datadog):

```html
<!-- Datadog Browser RUM -->
<script>
  (function(h,o,u,n,d) {
    h=h[d]=h[d]||{q:[],onReady:function(c){h.q.push(c)}}
    d=o.createElement(u);d.async=1;d.src=n
    n=o.getElementsByTagName(u)[0];n.parentNode.insertBefore(d,n)
  })(window,document,'script','https://www.datadoghq-browser-agent.com/us1/v5/datadog-rum.js','DD_RUM')
  window.DD_RUM.onReady(function() {
    window.DD_RUM.init({
      applicationId: 'YOUR_APPLICATION_ID',
      clientToken: 'YOUR_CLIENT_TOKEN',
      site: 'datadoghq.com',
      service: 'neural-dream-workshop',
      env: 'production',
      version: '1.0.0',
      sessionSampleRate: 100,
      sessionReplaySampleRate: 20,
      trackUserInteractions: true,
      trackResources: true,
      trackLongTasks: true,
      defaultPrivacyLevel: 'mask-user-input',
    });
  });
</script>
```

**Step 3: Commit**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add client/index.html
git commit -m "feat: add Datadog Browser RUM for frontend error visibility"
```

---

## Task 5: Activate Datadog APM on ML backend

**Files:**
- Modify: `render.yaml` (add DD_API_KEY envVar)
- Modify: `ml/main.py` lines 85-110 (wire accuracy metrics in auto-retraining loop)

**Context:** `ddtrace>=2.0.0` is already in `ml/requirements.txt`. The patch_all() is already in `ml/main.py` lines 8-13 — it just needs `DD_API_KEY` set in Render to activate. The auto-retraining loop already has a comment "Report accuracy to Datadog" — we just wire it.

**Step 1: Add DD_API_KEY to render.yaml**

In `render.yaml`, add these lines after the existing envVars block:
```yaml
      - key: DD_API_KEY
        sync: false   # tells Render this is a secret — set manually in dashboard
      - key: DD_SITE
        value: datadoghq.com
      - key: DD_SERVICE
        value: neural-dream-ml
      - key: DD_ENV
        value: production
```

**Step 2: Set DD_API_KEY in Render dashboard (manual)**

- Go to Render dashboard → `neural-dream-ml` → Environment
- Add `DD_API_KEY` = (your key from Step M3)
- Also add `DD_SITE=datadoghq.com`, `DD_SERVICE=neural-dream-ml`, `DD_ENV=production`

**Step 3: Wire accuracy metric in ml/main.py**

Find the `_auto_train_loop` function in `ml/main.py` (around line 85). Find the comment `# Report accuracy to Datadog`. Replace that comment with:

```python
            # Report accuracy to Datadog
            if isinstance(result, dict):
                try:
                    from ddtrace.runtime import RuntimeMetrics
                    RuntimeMetrics.enable()
                    from datadog import statsd
                    acc = result.get("cross_val_accuracy") or result.get("accuracy")
                    if acc is not None:
                        statsd.gauge("eeg.model.retrain_accuracy", acc * 100)
                        statsd.gauge("eeg.model.retrain_timestamp", __import__("time").time())
                except Exception:
                    pass  # non-fatal — metrics are optional
```

**Step 4: Commit**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add render.yaml ml/main.py
git commit -m "feat: activate Datadog APM and wire retrain accuracy metrics"
git push
```

---

## Task 6: Retrain stress model with DREAMER dataset

**Files:**
- Run: `ml/training/train_dreamer.py` (already exists)
- Result: updates `ml/models/saved/stress_model.pkl` and `emotion_mega_lgbm.pkl`

**Context:** Stress model is at 59.64% CV — below the 60% activation threshold, so it's disabled in live inference. DREAMER is a consumer Emotiv EPOC dataset (23 subjects, closest hardware to Muse 2). `train_dreamer.py` already exists. Target: ≥65% CV.

**Step 1: Check if DREAMER data exists**
```bash
ls /Users/sravyalu/NeuralDreamWorkshop/ml/data/ 2>/dev/null || echo "no data dir"
ls ~/.cache/kagglehub/ 2>/dev/null | head -20
```

**Step 2: Run the DREAMER training script**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
python ml/training/train_dreamer.py 2>&1 | tee /tmp/train_dreamer_log.txt
```
Expected: downloads DREAMER if not cached, trains, saves model, prints CV accuracy.
If download fails: script likely uses kagglehub or moabb — check `train_dreamer.py` line 1-30 for download method.

**Step 3: Check accuracy**
```bash
tail -20 /tmp/train_dreamer_log.txt
```
Look for: `Cross-val accuracy: XX.XX%` or similar.
Target: ≥65% for stress. If accuracy < 60%, check if DREAMER data loaded correctly.

**Step 4: Run full mega LGBM retrain to incorporate new data**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
python ml/training/train_mega_lgbm_unified.py 2>&1 | tail -30
```
Expected: new `emotion_mega_lgbm.pkl` with higher accuracy.

**Step 5: Commit updated models**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add ml/models/saved/
git commit -m "feat: retrain stress + mega LGBM with DREAMER dataset"
```

---

## Task 7: Retrain flow state + TSception 4-channel model

**Files:**
- Run: `ml/training/train_tsception.py` (already exists)
- Modify: `ml/training/train_tsception.py` if 4-channel config needs fixing

**Context:** Flow state at 57% CV is below threshold. TSception is an asymmetry-aware CNN designed specifically for 4-channel EEG — it treats AF7/AF8 as a left/right pair. `train_tsception.py` exists. `eegnet_emotion_4ch.pt` is the output. Target: ≥62% CV.

**Step 1: Check train_tsception.py configuration**
```bash
head -60 /Users/sravyalu/NeuralDreamWorkshop/ml/training/train_tsception.py
```
Look for: `n_channels`, `T` (time samples), dataset used. Confirm it's configured for 4 channels.

**Step 2: Run TSception training**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
python ml/training/train_tsception.py 2>&1 | tee /tmp/train_tsception_log.txt
```
Expected: prints training progress, saves `ml/models/saved/eegnet_emotion_4ch.pt`, reports test accuracy.

**Step 3: Check output accuracy**
```bash
tail -20 /tmp/train_tsception_log.txt
```
Target: ≥62% test accuracy (cross-subject).

**Step 4: Run benchmark to update STATUS.md numbers**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
python ml/training/benchmark.py 2>&1 | tail -40
```

**Step 5: Commit**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add ml/models/saved/eegnet_emotion_4ch.pt ml/models/saved/flow_state_model.pkl
git commit -m "feat: retrain TSception 4-channel + flow state model"
git push
```

---

## Task 8: Add FACED dataset support and retrain emotion mega LGBM

**Files:**
- Run: `ml/training/train_faced.py` (already exists)

**Context:** FACED (2023) is the largest EEG emotion dataset: 123 subjects, 9 classes, 32-channel. It's the closest Chinese academic dataset to what will generalize to real users. `train_faced.py` exists. Incorporating it into the mega LGBM should push emotion accuracy from 74.21% toward 77-80%.

**Step 1: Run FACED training**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
python ml/training/train_faced.py 2>&1 | tee /tmp/train_faced_log.txt
```

**Step 2: Check if FACED data exists; follow download instructions if not**
```bash
head -80 /Users/sravyalu/NeuralDreamWorkshop/ml/training/train_faced.py | grep -A5 "download\|kaggle\|zenodo\|url"
```

**Step 3: After FACED trains, re-run mega LGBM unified**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
python ml/training/train_mega_lgbm_unified.py 2>&1 | tail -30
```

**Step 4: Commit**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add ml/models/saved/
git commit -m "feat: incorporate FACED dataset into mega LGBM emotion model"
git push
```

---

## Task 9: Update STATUS.md with new model numbers

**Files:**
- Modify: `STATUS.md` model accuracy table

**Step 1: Get all current benchmark numbers**
```bash
cat /Users/sravyalu/NeuralDreamWorkshop/ml/models/saved/*.json 2>/dev/null | python3 -c "
import json, sys
for line in sys.stdin:
    try:
        d = json.loads(line)
        if 'accuracy' in d:
            print(d.get('model_name','?'), d.get('accuracy'))
    except: pass
"
```

**Step 2: Update STATUS.md**

Open `STATUS.md` and update the model accuracy table with the new numbers from the benchmark runs above.

**Step 3: Commit**
```bash
cd /Users/sravyalu/NeuralDreamWorkshop
git add STATUS.md
git commit -m "docs: update model accuracy numbers after DREAMER + FACED retraining"
git push
```

---

## Verification Checklist (run after all tasks)

```bash
# 1. No dead URL references
grep -r "brendan-postanesthetic" /Users/sravyalu/NeuralDreamWorkshop/ && echo "FAIL" || echo "PASS: no dead URL"

# 2. TypeScript clean
cd /Users/sravyalu/NeuralDreamWorkshop && npx tsc --noEmit 2>&1 | grep -v "test/" && echo "PASS: typecheck clean"

# 3. ML backend health (replace with your Render URL)
curl -s https://neural-dream-ml.onrender.com/health | python3 -m json.tool

# 4. CORS headers from Render to Vercel origin
curl -s -I -H "Origin: https://dream-analysis.vercel.app" https://neural-dream-ml.onrender.com/health | grep -i "access-control"

# 5. Model accuracy check
python3 -c "
import pickle, sys
sys.path.insert(0, 'ml')
with open('ml/models/saved/emotion_mega_lgbm.pkl', 'rb') as f:
    bundle = pickle.load(f)
print('Stress model exists:', 'stress' in str(bundle))
"
```
