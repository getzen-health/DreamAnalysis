/**
 * Share utilities — platform-aware image sharing, saving, and clipboard copy.
 *
 * Tries Capacitor native APIs first (iOS/Android), falls back to Web APIs,
 * then finally to download-link fallback.
 *
 * Capacitor plugins are dynamically imported so the web build never fails
 * if native modules are absent.
 */

// ── Helpers ────────────────────────────────────────────────────────────────

/** Convert a data URL (e.g. from canvas.toDataURL) to a Blob. */
export function dataUrlToBlob(dataUrl: string): Blob {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)?.[1] ?? "image/png";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}

/** Trigger a download via an invisible anchor element. */
function downloadFallback(dataUrl: string, filename: string): void {
  const link = document.createElement("a");
  link.download = filename;
  link.href = dataUrl;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ── Share ───────────────────────────────────────────────────────────────────

/**
 * Share an image via the best available API:
 * 1. Capacitor Share plugin (native share sheet)
 * 2. Web Share API (with file support)
 * 3. Download fallback
 */
export async function shareImage(dataUrl: string, filename: string): Promise<void> {
  // Try Capacitor first
  try {
    const { Filesystem, Directory } = await import("@capacitor/filesystem");
    const { Share } = await import("@capacitor/share");

    const base64Data = dataUrl.split(",")[1];
    const written = await Filesystem.writeFile({
      path: filename,
      data: base64Data,
      directory: Directory.Cache,
    });

    await Share.share({
      title: "Share Card",
      url: written.uri,
    });
    return;
  } catch {
    // Not on native — fall through
  }

  // Try Web Share API with file
  try {
    if (navigator.share && navigator.canShare) {
      const blob = dataUrlToBlob(dataUrl);
      const file = new File([blob], filename, { type: blob.type });
      const shareData = { files: [file] };

      if (navigator.canShare(shareData)) {
        await navigator.share(shareData);
        return;
      }
    }
  } catch {
    // User cancelled or not supported — fall through
  }

  // Fallback: download
  downloadFallback(dataUrl, filename);
}

// ── Copy to Clipboard ──────────────────────────────────────────────────────

/**
 * Copy an image to the clipboard.
 * Returns true if successful, false otherwise.
 */
export async function copyImageToClipboard(dataUrl: string): Promise<boolean> {
  try {
    if (!navigator.clipboard?.write) return false;

    const blob = dataUrlToBlob(dataUrl);
    // Clipboard API requires image/png
    const pngBlob =
      blob.type === "image/png"
        ? blob
        : await convertToPngBlob(dataUrl);

    await navigator.clipboard.write([
      new ClipboardItem({ "image/png": pngBlob }),
    ]);
    return true;
  } catch {
    return false;
  }
}

/** Re-render a data URL as PNG blob (for clipboard compatibility). */
async function convertToPngBlob(dataUrl: string): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("No 2d context"));
      ctx.drawImage(img, 0, 0);
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
        "image/png",
      );
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

// ── Save to Photos ─────────────────────────────────────────────────────────

/**
 * Save image to device photo library (Capacitor) or download.
 */
export async function saveImageToPhotos(
  dataUrl: string,
  filename: string,
): Promise<void> {
  // Try Capacitor Filesystem
  try {
    const { Filesystem, Directory } = await import("@capacitor/filesystem");

    const base64Data = dataUrl.split(",")[1];
    await Filesystem.writeFile({
      path: `DCIM/${filename}`,
      data: base64Data,
      directory: Directory.ExternalStorage,
      recursive: true,
    });
    return;
  } catch {
    // Not on native — fall through
  }

  // Fallback: browser download
  downloadFallback(dataUrl, filename);
}
