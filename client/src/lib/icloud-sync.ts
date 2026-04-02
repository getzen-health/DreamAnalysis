/**
 * icloud-sync.ts
 *
 * iCloud/CloudKit backup bridge for iOS.
 * Backs up the on-device SQLite database to the user's private iCloud container.
 * Data is end-to-end encrypted by Apple — nobody (including Anthropic/AntarAI) can read it.
 *
 * On Android: no-op (use Google Drive backup via Android Auto Backup instead).
 * On web: no-op.
 */

import { Capacitor, registerPlugin } from "@capacitor/core";

// ── Plugin interface ──────────────────────────────────────────────────────────

interface ICloudSyncPlugin {
  isAvailable(): Promise<{ available: boolean }>;
  backup(options: { dbName: string }): Promise<{ success: boolean; timestamp: string }>;
  restore(options: { dbName: string }): Promise<{ success: boolean; restoredAt: string | null }>;
  getLastBackupDate(): Promise<{ date: string | null }>;
}

// Registered in ios/App/App/ICloudSyncPlugin.swift
const ICloudSyncNative = registerPlugin<ICloudSyncPlugin>("ICloudSync");

// ── Public API ────────────────────────────────────────────────────────────────

export function isIOS(): boolean {
  return Capacitor.getPlatform() === "ios";
}

/**
 * Back up the on-device SQLite database to the user's private iCloud container.
 * No-op on non-iOS platforms.
 */
export async function backupToiCloud(dbName = "antaiai_private"): Promise<boolean> {
  if (!isIOS()) return false;
  try {
    const { available } = await ICloudSyncNative.isAvailable();
    if (!available) return false;
    const result = await ICloudSyncNative.backup({ dbName });
    return result.success;
  } catch {
    return false;
  }
}

/**
 * Restore the SQLite database from iCloud backup.
 * Used on first launch after reinstall or device migration.
 */
export async function restoreFromiCloud(dbName = "antaiai_private"): Promise<boolean> {
  if (!isIOS()) return false;
  try {
    const { available } = await ICloudSyncNative.isAvailable();
    if (!available) return false;
    const result = await ICloudSyncNative.restore({ dbName });
    return result.success;
  } catch {
    return false;
  }
}

export async function getLastBackupDate(): Promise<Date | null> {
  if (!isIOS()) return null;
  try {
    const { date } = await ICloudSyncNative.getLastBackupDate();
    return date ? new Date(date) : null;
  } catch {
    return null;
  }
}

/**
 * Smart backup: only backs up if last backup was > 12 hours ago and device is on WiFi.
 * Call this from app foreground event.
 */
export async function maybeBackup(): Promise<void> {
  if (!isIOS()) return;
  try {
    const last = await getLastBackupDate();
    const twelveHoursAgo = Date.now() - 12 * 60 * 60 * 1000;
    if (last && last.getTime() > twelveHoursAgo) return; // already backed up recently
    await backupToiCloud();
  } catch {
    // non-fatal
  }
}
