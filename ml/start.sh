#!/bin/bash
# ============================================================
#  Neural Dream Workshop — ML Backend Startup
#  One command to start everything: venv + backend + ngrok
#
#  Usage:  chmod +x ml/start.sh   (first time only)
#          ./ml/start.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

print_header() {
  echo ""
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}   🧠  Neural Dream Workshop — ML Backend${RESET}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo ""
}

print_header

# ── Step 1: Python virtual environment ──────────────────────────────────────
echo -e "${CYAN}[1/4] Python environment${RESET}"

if [ ! -d ".venv" ]; then
  echo "  Creating virtual environment..."
  python3 -m venv .venv
  echo -e "  ${GREEN}✓ Created .venv${RESET}"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo -e "  ${GREEN}✓ Activated .venv ($(python3 --version))${RESET}"

# ── Step 2: Install runtime dependencies ────────────────────────────────────
echo ""
echo -e "${CYAN}[2/4] Dependencies${RESET}"

install_pkg() {
  # Silently install, only print on hard errors; never abort the script
  pip install -q "$@" 2>&1 | grep -Ei "^ERROR:" || true
}

echo "  [a] API server..."
install_pkg "fastapi==0.115.0" "uvicorn[standard]==0.30.6" \
  "pydantic==2.9.2" "python-multipart==0.0.12" \
  "httpx==0.27.2" "websockets==12.0"
echo -e "  ${GREEN}✓ API server packages${RESET}"

echo "  [b] EEG hardware (brainflow)..."
install_pkg "brainflow==5.12.1"
echo -e "  ${GREEN}✓ brainflow${RESET}"

echo "  [c] Signal processing..."
install_pkg "numpy>=1.26.4,<2.0" "scipy>=1.13.0" "PyWavelets>=1.7.0" \
  "scikit-learn>=1.5.0" "joblib>=1.3.0" "pandas>=2.0.0" \
  "lightgbm>=4.0.0" "h5py>=3.0.0"
echo -e "  ${GREEN}✓ Signal processing packages${RESET}"

echo "  [d] ONNX inference..."
install_pkg "onnxruntime>=1.19.0" || true
echo -e "  ${GREEN}✓ ONNX runtime${RESET}"

echo "  [e] PyTorch (CPU)..."
# Use --index-url to get CPU wheels; skip gracefully if Python 3.13 wheel missing
pip install -q torch \
  --index-url https://download.pytorch.org/whl/cpu \
  2>&1 | grep -Ei "^ERROR:" || true
echo -e "  ${GREEN}✓ PyTorch (CPU)${RESET}"

echo "  [f] Optional DB packages..."
install_pkg "psycopg2-binary>=2.9.9" "asyncpg>=0.29.0" || true
echo -e "  ${GREEN}✓ Optional packages done${RESET}"

# ── Step 3: Start uvicorn ────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[3/4] ML Backend${RESET}"

# Pick port: prefer 8080, fall back to 8001 if busy
NDW_PORT=8080
if lsof -ti:8080 > /dev/null 2>&1; then
  pid=$(lsof -ti:8080)
  # Only kill if it's our own uvicorn process
  if ps -p "$pid" -o command= 2>/dev/null | grep -q "uvicorn main:app"; then
    echo "  Stopping stale uvicorn on port 8080..."
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
  else
    echo "  Port 8080 busy (not uvicorn) — using 8001"
    NDW_PORT=8001
  fi
fi

echo -e "  Using port ${NDW_PORT}"

# Start uvicorn in background, log to file
LOG_FILE="/tmp/ndw_backend.log"
EXPRESS_URL="http://localhost:4000" nohup python3 -m uvicorn main:app \
  --port "$NDW_PORT" \
  --host 0.0.0.0 \
  --log-level warning \
  > "$LOG_FILE" 2>&1 &
UVICORN_PID=$!

# Wait for health endpoint — model loading can take 20-30s on first run
echo -n "  Waiting for backend (model loading takes ~20s)"
READY=0
for i in {1..60}; do
  if curl -sf "http://localhost:${NDW_PORT}/health" > /dev/null 2>&1; then
    READY=1
    break
  fi
  echo -n "."
  sleep 0.5
done
echo ""

if [ "$READY" -eq 0 ]; then
  echo -e "  ${RED}✗ Backend failed to start — showing last 20 lines of log:${RESET}"
  tail -20 "$LOG_FILE" || true
  echo ""
  echo "  Common fixes:"
  echo "    1. Port 8000 already in use: lsof -ti:8000 | xargs kill"
  echo "    2. Python import error: check $LOG_FILE"
  exit 1
fi
echo -e "  ${GREEN}✓ Backend ready (PID $UVICORN_PID) — logs: $LOG_FILE${RESET}"

# ── Step 4: ngrok ────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[4/4] ngrok tunnel${RESET}"

if ! command -v ngrok &> /dev/null; then
  echo -e "  ${YELLOW}⚠  ngrok not found.${RESET}"
  echo ""
  echo "  Install it with:"
  echo "    brew install ngrok/ngrok/ngrok"
  echo "  Then add your authtoken:"
  echo "    ngrok config add-authtoken <YOUR_TOKEN>  (free at ngrok.com)"
  echo ""
  echo -e "  ${YELLOW}Skipping ngrok. Use the backend locally at:${RESET}"
  echo "  http://localhost:${NDW_PORT}"
  echo ""
  echo "  Paste that into Settings → ML Backend if you are running"
  echo "  the app on the same machine."
  echo ""
  echo "  Press Ctrl+C to stop the backend"
  wait "$UVICORN_PID" 2>/dev/null || true
  exit 0
fi

# Kill stale ngrok on port 8000
pkill -f "ngrok http" 2>/dev/null || true
sleep 1

# Start ngrok
nohup ngrok http "$NDW_PORT" \
  --log=stdout \
  --log-format=json \
  > /tmp/ndw_ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok API to return a URL (up to 10 seconds)
echo -n "  Waiting for ngrok URL"
NGROK_URL=""
for i in {1..20}; do
  NGROK_URL=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null | \
    python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnels = data.get('tunnels', [])
    https_tunnels = [t['public_url'] for t in tunnels if t['public_url'].startswith('https')]
    print(https_tunnels[0] if https_tunnels else '')
except:
    print('')
" 2>/dev/null || true)

  if [ -n "$NGROK_URL" ]; then
    break
  fi
  echo -n "."
  sleep 0.5
done
echo ""

if [ -z "$NGROK_URL" ]; then
  echo -e "  ${RED}✗ Could not get ngrok URL.${RESET}"
  echo "  Check /tmp/ndw_ngrok.log for errors."
  echo ""
  echo "  Common causes:"
  echo "    - Not authenticated: ngrok config add-authtoken <YOUR_TOKEN>"
  echo "    - ngrok account limit reached (free plan allows 1 agent)"
  echo ""
  echo -e "  ${YELLOW}Backend is still running at http://localhost:8000${RESET}"
  echo "  If running the app locally, use that URL in Settings."
  exit 1
fi

# ── Done ─────────────────────────────────────────────────────────────────────
ENCODED_URL=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$NGROK_URL'))")
SETTINGS_LINK="https://dream-analysis.vercel.app/settings?ml_backend=${ENCODED_URL}"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  ✅  ML Backend is LIVE${RESET}"
echo ""
echo -e "  ${BOLD}ngrok URL:${RESET}  ${CYAN}${NGROK_URL}${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# ── Auto-update Vercel + open Settings (runs in background) ──────────────────
(
  # 1. Update VITE_ML_API_URL in Vercel
  if command -v vercel &>/dev/null; then
    vercel env rm VITE_ML_API_URL production --yes 2>/dev/null || true
    vercel env rm VITE_ML_API_URL preview   --yes 2>/dev/null || true
    printf '%s' "$NGROK_URL" | vercel env add VITE_ML_API_URL production 2>/dev/null || true
    printf '%s' "$NGROK_URL" | vercel env add VITE_ML_API_URL preview   2>/dev/null || true
    # 2. Rebuild + deploy
    cd "$PROJECT_DIR"
    npx vercel --prod > /tmp/ndw_vercel.log 2>&1 \
      && echo "[vercel] Deploy complete: $NGROK_URL" >> /tmp/ndw_vercel.log \
      || echo "[vercel] Deploy failed — check /tmp/ndw_vercel.log" >&2
  fi
) &
VERCEL_PID=$!

echo "  Updating Vercel in background (PID $VERCEL_PID, ~60s)..."
echo "  App will auto-reload with the new URL. Log: /tmp/ndw_vercel.log"
echo ""

# Open Settings with URL pre-filled (immediate fix for your browser)
if command -v open &>/dev/null; then
  open "$SETTINGS_LINK" 2>/dev/null || true
fi

echo "  Press Ctrl+C to stop the backend"
echo ""

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "  Stopping backend and ngrok..."
  kill "$UVICORN_PID" 2>/dev/null || true
  kill "$NGROK_PID" 2>/dev/null || true
  kill "$VERCEL_PID" 2>/dev/null || true
  pkill -f "uvicorn main:app" 2>/dev/null || true
  pkill -f "ngrok http 8000" 2>/dev/null || true
  echo -e "  ${GREEN}Stopped.${RESET}"
}

trap cleanup INT TERM

# Keep running until Ctrl+C
wait "$UVICORN_PID" 2>/dev/null || true
