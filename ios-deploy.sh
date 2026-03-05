#!/bin/bash
# ── iOS Deploy Script for Neural Dream Workshop ──────────────────────────────
# Run this AFTER Xcode is installed and xcode-select is set.
#
# Usage:
#   chmod +x ios-deploy.sh
#   ./ios-deploy.sh
#
# Requirements:
#   - Xcode installed at /Applications/Xcode.app
#   - Apple Developer account signed into Xcode
#   - iPhone connected via USB (for device build) OR use simulator
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=== Neural Dream Workshop — iOS Deploy ==="

# Step 1: Set Xcode developer path
echo "[1/6] Setting Xcode developer path..."
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept 2>/dev/null || true
echo "  ✓ Xcode active: $(xcode-select -p)"

# Step 2: Build the React/Vite frontend
echo "[2/6] Building frontend..."
npm run build
echo "  ✓ Frontend built to dist/public"

# Step 3: Sync Capacitor (copies web assets to iOS native project)
echo "[3/6] Syncing Capacitor to iOS..."
npx cap sync ios
echo "  ✓ Capacitor synced"

# Step 4: Install CocoaPods dependencies
echo "[4/6] Installing CocoaPods..."
cd ios/App
pod install --repo-update
cd ../..
echo "  ✓ Pods installed"

# Step 5: Build for device (archive)
echo "[5/6] Building iOS app..."
echo ""
echo "  ⚡ Opening Xcode — complete these steps in Xcode:"
echo "  1. Click on 'App' in the project navigator"
echo "  2. Go to Signing & Capabilities tab"
echo "  3. Select your Team (lakshmisravya.vedantham@gmail.com)"
echo "  4. Change bundle ID if needed (currently: com.neuraldreamworkshop.app)"
echo "  5. Product → Archive"
echo "  6. In Organizer: Distribute App → App Store Connect → Upload"
echo ""

# Open Xcode with the project
npx cap open ios

echo "[6/6] Xcode opened — complete signing and upload in Xcode."
echo ""
echo "=== Done! Follow the Xcode steps above to submit to App Store. ==="
