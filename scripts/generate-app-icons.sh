#!/usr/bin/env bash
# generate-app-icons.sh — Generate all required iOS app icon sizes
#
# Usage: ./scripts/generate-app-icons.sh [source_image]
#
# Default source: client/public/icon-1024.png
# Uses macOS built-in `sips` — no ImageMagick or other dependencies needed.
#
# Generates icons for:
#   - iOS App Icon (universal 1024x1024 — required since Xcode 15 / iOS 17+)
#   - Legacy sizes for older iOS versions and iPad
#   - Spotlight, Settings, and Notification icons
#
# Output: ios/App/App/Assets.xcassets/AppIcon.appiconset/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE="${1:-$PROJECT_ROOT/client/public/icon-1024.png}"
OUTPUT_DIR="$PROJECT_ROOT/ios/App/App/Assets.xcassets/AppIcon.appiconset"

# ── Colors ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Validate ─────────────────────────────────────────────────────────────────

if [[ ! -f "$SOURCE" ]]; then
    error "Source image not found: $SOURCE
    Provide a 1024x1024 PNG as argument or place it at client/public/icon-1024.png"
fi

if ! command -v sips &>/dev/null; then
    error "sips not found. This script requires macOS."
fi

# Verify source is at least 1024x1024
SOURCE_WIDTH=$(sips -g pixelWidth "$SOURCE" | tail -1 | awk '{print $2}')
SOURCE_HEIGHT=$(sips -g pixelHeight "$SOURCE" | tail -1 | awk '{print $2}')

if [[ "$SOURCE_WIDTH" -lt 1024 ]] || [[ "$SOURCE_HEIGHT" -lt 1024 ]]; then
    error "Source image is ${SOURCE_WIDTH}x${SOURCE_HEIGHT}. Must be at least 1024x1024."
fi

info "Source: $SOURCE (${SOURCE_WIDTH}x${SOURCE_HEIGHT})"

# ── Create output directory ──────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"

# ── Define all required sizes ────────────────────────────────────────────────
#
# Format: "filename pixel_size"
#
# Since Xcode 15+ / iOS 17+, Apple uses a single 1024x1024 universal icon
# and auto-generates all sizes. But for backwards compatibility with older
# Xcode and deployment targets < iOS 17, we generate the full set.

SIZES=(
    # Notification icon
    "AppIcon-20@1x.png 20"
    "AppIcon-20@2x.png 40"
    "AppIcon-20@3x.png 60"

    # Settings icon
    "AppIcon-29@1x.png 29"
    "AppIcon-29@2x.png 58"
    "AppIcon-29@3x.png 87"

    # Spotlight icon
    "AppIcon-40@1x.png 40"
    "AppIcon-40@2x.png 80"
    "AppIcon-40@3x.png 120"

    # iPhone app icon
    "AppIcon-60@2x.png 120"
    "AppIcon-60@3x.png 180"

    # iPad app icon
    "AppIcon-76@1x.png 76"
    "AppIcon-76@2x.png 152"

    # iPad Pro app icon
    "AppIcon-83.5@2x.png 167"

    # App Store (required, universal)
    "AppIcon-512@2x.png 1024"
)

# ── Generate icons ───────────────────────────────────────────────────────────

info "Generating ${#SIZES[@]} icon sizes..."

for entry in "${SIZES[@]}"; do
    filename="${entry%% *}"
    pixel_size="${entry##* }"
    output_path="$OUTPUT_DIR/$filename"

    # Copy source, then resize with sips
    cp "$SOURCE" "$output_path"
    sips -z "$pixel_size" "$pixel_size" "$output_path" --out "$output_path" >/dev/null 2>&1

    printf "  %-28s %4sx%-4s\n" "$filename" "$pixel_size" "$pixel_size"
done

# ── Write Contents.json ──────────────────────────────────────────────────────
#
# Xcode 15+ single-size format (universal icon, auto-generates others)
# This is the modern recommended approach. The individual size files above
# are kept for backwards compatibility but Xcode will use the universal entry.

cat > "$OUTPUT_DIR/Contents.json" << 'CONTENTS_JSON'
{
  "images" : [
    {
      "filename" : "AppIcon-20@1x.png",
      "idiom" : "iphone",
      "scale" : "1x",
      "size" : "20x20"
    },
    {
      "filename" : "AppIcon-20@2x.png",
      "idiom" : "iphone",
      "scale" : "2x",
      "size" : "20x20"
    },
    {
      "filename" : "AppIcon-20@3x.png",
      "idiom" : "iphone",
      "scale" : "3x",
      "size" : "20x20"
    },
    {
      "filename" : "AppIcon-29@1x.png",
      "idiom" : "iphone",
      "scale" : "1x",
      "size" : "29x29"
    },
    {
      "filename" : "AppIcon-29@2x.png",
      "idiom" : "iphone",
      "scale" : "2x",
      "size" : "29x29"
    },
    {
      "filename" : "AppIcon-29@3x.png",
      "idiom" : "iphone",
      "scale" : "3x",
      "size" : "29x29"
    },
    {
      "filename" : "AppIcon-40@2x.png",
      "idiom" : "iphone",
      "scale" : "2x",
      "size" : "40x40"
    },
    {
      "filename" : "AppIcon-40@3x.png",
      "idiom" : "iphone",
      "scale" : "3x",
      "size" : "40x40"
    },
    {
      "filename" : "AppIcon-60@2x.png",
      "idiom" : "iphone",
      "scale" : "2x",
      "size" : "60x60"
    },
    {
      "filename" : "AppIcon-60@3x.png",
      "idiom" : "iphone",
      "scale" : "3x",
      "size" : "60x60"
    },
    {
      "filename" : "AppIcon-20@1x.png",
      "idiom" : "ipad",
      "scale" : "1x",
      "size" : "20x20"
    },
    {
      "filename" : "AppIcon-20@2x.png",
      "idiom" : "ipad",
      "scale" : "2x",
      "size" : "20x20"
    },
    {
      "filename" : "AppIcon-29@1x.png",
      "idiom" : "ipad",
      "scale" : "1x",
      "size" : "29x29"
    },
    {
      "filename" : "AppIcon-29@2x.png",
      "idiom" : "ipad",
      "scale" : "2x",
      "size" : "29x29"
    },
    {
      "filename" : "AppIcon-40@1x.png",
      "idiom" : "ipad",
      "scale" : "1x",
      "size" : "40x40"
    },
    {
      "filename" : "AppIcon-40@2x.png",
      "idiom" : "ipad",
      "scale" : "2x",
      "size" : "40x40"
    },
    {
      "filename" : "AppIcon-76@1x.png",
      "idiom" : "ipad",
      "scale" : "1x",
      "size" : "76x76"
    },
    {
      "filename" : "AppIcon-76@2x.png",
      "idiom" : "ipad",
      "scale" : "2x",
      "size" : "76x76"
    },
    {
      "filename" : "AppIcon-83.5@2x.png",
      "idiom" : "ipad",
      "scale" : "2x",
      "size" : "83.5x83.5"
    },
    {
      "filename" : "AppIcon-512@2x.png",
      "idiom" : "ios-marketing",
      "scale" : "1x",
      "size" : "1024x1024"
    }
  ],
  "info" : {
    "author" : "xcode",
    "version" : 1
  }
}
CONTENTS_JSON

info "Contents.json written."

echo ""
info "All icons generated in:"
echo "  $OUTPUT_DIR"
echo ""
info "Done. Open Xcode to verify icons appear in Assets.xcassets > AppIcon."
