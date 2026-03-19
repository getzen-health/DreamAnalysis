import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.neuraldreamworkshop.app",
  appName: "Neural Dream Workshop",
  // Vite builds the React app to dist/public
  webDir: "dist/public",
  server: {
    // Production: Vercel handles all /api routes; static assets served natively.
    // For local device testing, uncomment and set to your machine's local IP:
    //   url: "http://192.168.x.x:5173",
    //   cleartext: true,
  },
  ios: {
    // Minimum iOS version — 14+ covers all iPhones since 2017
    minVersion: "14.0",
    // URL scheme for deep linking (neuraldream://)
    scheme: "Neural Dream Workshop",
    // Allow inline media without user gesture (for biofeedback animations)
    allowsInlineMediaPlayback: true,
    backgroundColor: "#0d0f14",
    // Content inset behavior for safe areas (iPhone notch/Dynamic Island)
    contentInset: "automatic",
  },
  android: {
    // Minimum Android API level 24 (Android 7, 2016+)
    minWebViewVersion: 55,
    backgroundColor: "#0d0f14",
    // Allow clear text for local dev server (not production)
    allowMixedContent: false,
  },
  plugins: {
    SplashScreen: {
      // Show Heart-Brain logo splash for 2s, then fade to app
      launchShowDuration: 2000,
      launchAutoHide: true,
      launchFadeOutDuration: 300,
      backgroundColor: "#13111a",
      showSpinner: false,
      splashImmersive: true,
      splashFullScreen: true,
    },
    StatusBar: {
      style: "LIGHT",
      backgroundColor: "#13111a",
    },
    PushNotifications: {
      presentationOptions: ["badge", "sound", "alert"],
    },
    BackgroundRunner: {
      // Path relative to webDir (dist/public)
      label: "com.neuraldreamworkshop.eeg-flush",
      src: "background-runner.js",
      event: "eegFlush",
      repeat: true,
      interval: 15,   // minutes
      autoStart: false,
    },
    BluetoothLe: {
      // Shown in iOS permission prompt when scanning for Muse
      displayStrings: {
        scanning: "Scanning for Muse headbands...",
        cancel: "Cancel",
        availableDevices: "Available Muse Devices",
        noDeviceFound: "No Muse headband found",
      },
    },
  },
};

export default config;
