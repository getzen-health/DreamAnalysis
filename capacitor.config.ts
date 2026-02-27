import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.neuraldreamworkshop.app",
  appName: "Neural Dream",
  // Vite builds the React app to dist/public
  webDir: "dist/public",
  server: {
    // In development, point to the local Express server so you can
    // run `npm run dev` and `npx cap run ios` simultaneously.
    // Comment this out for production builds.
    // url: "http://localhost:5000",
    // cleartext: true,
  },
  ios: {
    // Minimum iOS version — 14+ covers all iPhones since 2017
    minVersion: "14.0",
    // Allow inline media without user gesture (for biofeedback animations)
    allowsInlineMediaPlayback: true,
    backgroundColor: "#0d0f14",
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
      launchShowDuration: 2000,
      launchAutoHide: true,
      backgroundColor: "#0d0f14",
      iosSpinnerStyle: "small",
      spinnerColor: "#6366f1",
      showSpinner: false,
      androidScaleType: "CENTER_CROP",
      splashFullScreen: true,
      splashImmersive: true,
    },
    StatusBar: {
      style: "DARK",
      backgroundColor: "#0d0f14",
    },
    PushNotifications: {
      presentationOptions: ["badge", "sound", "alert"],
    },
  },
};

export default config;
