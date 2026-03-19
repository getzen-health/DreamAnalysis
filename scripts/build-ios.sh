#!/usr/bin/env bash
# build-ios.sh — Build Neural Dream Workshop for iOS
#
# Usage: ./scripts/build-ios.sh [--release]
#
# Prerequisites:
#   - Xcode 15+ with command-line tools
#   - CocoaPods (gem install cocoapods)
#   - Node.js + npm (for Capacitor CLI)
#   - Apple Developer account (for signing)
#
# What this script does:
#   1. Validates prerequisites (Xcode, CocoaPods, node_modules)
#   2. Builds the Vite frontend (npm run build:mobile)
#   3. Syncs web assets into ios/ (npx cap sync ios)
#   4. Opens Xcode for archive + signing (or runs xcodebuild for CI)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IOS_DIR="$PROJECT_ROOT/ios/App"
SCHEME="App"
BUNDLE_ID="com.neuraldreamworkshop.app"
RELEASE_MODE=false

if [[ "${1:-}" == "--release" ]]; then
    RELEASE_MODE=true
fi

# ── Colors ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Step 1: Check prerequisites ─────────────────────────────────────────────

info "Checking prerequisites..."

# Xcode
if ! command -v xcodebuild &>/dev/null; then
    error "Xcode command-line tools not found. Install Xcode from the App Store, then run:
    xcode-select --install
    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
fi

XCODE_VERSION=$(xcodebuild -version 2>/dev/null | head -1 || echo "unknown")
info "Xcode: $XCODE_VERSION"

# CocoaPods
if ! command -v pod &>/dev/null; then
    error "CocoaPods not found. Install with:
    sudo gem install cocoapods
    — or —
    brew install cocoapods"
fi

POD_VERSION=$(pod --version 2>/dev/null || echo "unknown")
info "CocoaPods: $POD_VERSION"

# Node.js
if ! command -v node &>/dev/null; then
    error "Node.js not found. Install from https://nodejs.org or via nvm."
fi

NODE_VERSION=$(node --version 2>/dev/null || echo "unknown")
info "Node.js: $NODE_VERSION"

# node_modules
if [[ ! -d "$PROJECT_ROOT/node_modules" ]]; then
    warn "node_modules not found. Running npm install..."
    cd "$PROJECT_ROOT" && npm install
fi

# Capacitor CLI
if ! npx cap --version &>/dev/null; then
    error "@capacitor/cli not found in node_modules. Run: npm install @capacitor/cli"
fi

CAP_VERSION=$(npx cap --version 2>/dev/null || echo "unknown")
info "Capacitor CLI: $CAP_VERSION"

# ios/ directory
if [[ ! -d "$PROJECT_ROOT/ios" ]]; then
    warn "ios/ directory not found. Adding iOS platform..."
    cd "$PROJECT_ROOT" && npx cap add ios
fi

echo ""
info "All prerequisites met."

# ── Step 2: Build the web app ────────────────────────────────────────────────

info "Building Vite frontend (npm run build:mobile)..."
cd "$PROJECT_ROOT"
npm run build:mobile

if [[ ! -d "$PROJECT_ROOT/dist/public" ]]; then
    error "Build output not found at dist/public. Check vite config and build:mobile script."
fi

info "Frontend build complete. Output: dist/public/"

# ── Step 3: Sync to iOS ─────────────────────────────────────────────────────

info "Syncing web assets to iOS (npx cap sync ios)..."
cd "$PROJECT_ROOT"
npx cap sync ios

info "Capacitor sync complete."

# ── Step 4: Install CocoaPods dependencies ───────────────────────────────────

if [[ -f "$IOS_DIR/Podfile" ]]; then
    info "Installing CocoaPods dependencies..."
    cd "$IOS_DIR" && pod install
    info "Pod install complete."
fi

# ── Step 5: Build or open Xcode ─────────────────────────────────────────────

if [[ "$RELEASE_MODE" == true ]]; then
    echo ""
    echo "========================================"
    info "RELEASE BUILD — Archive for App Store"
    echo "========================================"
    echo ""
    echo "To build an archive for App Store submission, run from Xcode:"
    echo ""
    echo "  1. Open the workspace:"
    echo "     open $IOS_DIR/App.xcworkspace"
    echo ""
    echo "  2. Select 'Any iOS Device (arm64)' as the build target"
    echo ""
    echo "  3. Product > Archive"
    echo ""
    echo "  4. In the Organizer, click 'Distribute App' > 'App Store Connect'"
    echo ""
    echo "Or use xcodebuild for CI:"
    echo ""
    echo "  xcodebuild archive \\"
    echo "    -workspace $IOS_DIR/App.xcworkspace \\"
    echo "    -scheme $SCHEME \\"
    echo "    -archivePath build/NeuralDream.xcarchive \\"
    echo "    -destination 'generic/platform=iOS' \\"
    echo "    CODE_SIGN_IDENTITY=\"Apple Distribution\" \\"
    echo "    DEVELOPMENT_TEAM=\"YOUR_TEAM_ID\" \\"
    echo "    PRODUCT_BUNDLE_IDENTIFIER=\"$BUNDLE_ID\""
    echo ""
    echo "  xcodebuild -exportArchive \\"
    echo "    -archivePath build/NeuralDream.xcarchive \\"
    echo "    -exportPath build/export \\"
    echo "    -exportOptionsPlist ExportOptions.plist"
    echo ""
    echo "========================================"
    echo "SIGNING REQUIREMENTS"
    echo "========================================"
    echo ""
    echo "  Team ID:             Set in Xcode > Signing & Capabilities"
    echo "  Bundle ID:           $BUNDLE_ID"
    echo "  Provisioning:        Automatic signing recommended for first submission"
    echo "  Distribution Cert:   Apple Distribution certificate in Keychain"
    echo "  Capabilities:        HealthKit, Push Notifications, Background Modes"
    echo ""
else
    echo ""
    info "Opening Xcode workspace for development build..."
    echo ""
    echo "  Workspace: $IOS_DIR/App.xcworkspace"
    echo "  Scheme:    $SCHEME"
    echo "  Bundle ID: $BUNDLE_ID"
    echo ""

    if command -v open &>/dev/null; then
        open "$IOS_DIR/App.xcworkspace"
        info "Xcode opened. Select a simulator or device and press Cmd+R to run."
    else
        warn "Could not open Xcode automatically. Open manually:"
        echo "  open $IOS_DIR/App.xcworkspace"
    fi
fi

echo ""
info "Done."
