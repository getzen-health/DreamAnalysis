/** Stub for @capacitor/camera — used in vitest (jsdom) environment only. */

export const CameraResultType = {
  Base64: "base64",
  DataUrl: "dataUrl",
  Uri: "uri",
};

export const CameraSource = {
  Camera: "CAMERA",
  Photos: "PHOTOS",
  Prompt: "PROMPT",
};

export const Camera = {
  getPhoto: vi.fn().mockResolvedValue({ base64String: "", format: "jpeg" }),
  checkPermissions: vi.fn().mockResolvedValue({ camera: "granted" }),
  requestPermissions: vi.fn().mockResolvedValue({ camera: "granted" }),
};
