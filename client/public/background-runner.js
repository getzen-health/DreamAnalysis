/**
 * background-runner.js — runs in Capacitor BackgroundRunner isolated JS context.
 *
 * This file runs separately from the main app JS. It cannot access the DOM,
 * React state, or window. It can use CapacitorNotifications and fetch().
 *
 * Receives events via BackgroundRunner.dispatchEvent() from the main thread.
 */

// Show an ongoing notification when sleep EEG recording starts
addEventListener("eegSessionStart", async (resolve, reject, args) => {
  try {
    await CapacitorNotifications.schedule([{
      id: 1001,
      title: "Neural Dream — Sleep Recording",
      body: "EEG monitoring active. Tap to stop.",
      sound: null,
      smallIcon: null,
      iconColor: "#6366f1",
      schedule: { at: new Date() },
    }]);
    resolve();
  } catch (e) {
    reject(String(e));
  }
});

// Cancel the notification when recording stops
addEventListener("eegSessionStop", async (resolve, reject) => {
  try {
    await CapacitorNotifications.cancel([{ id: 1001 }]);
    resolve();
  } catch (e) {
    reject(String(e));
  }
});

// Periodic flush — called every 15 min by iOS BackgroundFetch
addEventListener("eegFlush", async (resolve, reject, args) => {
  try {
    const mlUrl = (args && args.mlApiUrl) ? args.mlApiUrl : "http://localhost:8000";
    const res = await fetch(mlUrl + "/api/health");
    resolve({ ok: res.ok });
  } catch (e) {
    reject(String(e));
  }
});
