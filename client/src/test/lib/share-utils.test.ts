import { describe, it, expect } from "vitest";
import { dataUrlToBlob } from "@/lib/share-utils";

// A tiny 1x1 red PNG as a base64 data URL
const TINY_PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8D4HwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

describe("dataUrlToBlob", () => {
  it("converts a data URL to a Blob with the correct MIME type", () => {
    const blob = dataUrlToBlob(TINY_PNG_DATA_URL);
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe("image/png");
    expect(blob.size).toBeGreaterThan(0);
  });

  it("handles jpeg data URLs", () => {
    // Minimal valid jpeg header as base64
    const jpegUrl = "data:image/jpeg;base64,/9j/4AAQSkZJRg==";
    const blob = dataUrlToBlob(jpegUrl);
    expect(blob.type).toBe("image/jpeg");
  });

  it("returns non-zero size blob for valid input", () => {
    const blob = dataUrlToBlob(TINY_PNG_DATA_URL);
    expect(blob.size).toBeGreaterThan(50); // PNG header alone is ~67 bytes
  });
});
