/**
 * Participant identity — persists across sessions via localStorage.
 *
 * Each device/browser gets one UUID on first visit. That UUID is the
 * userId sent to every API call. After enrollment, the study code is
 * also stored here and shown in the UI.
 *
 * If a different person clears localStorage (or uses a different browser),
 * they get a fresh UUID and their data is completely separate.
 */

const KEY_USER_ID   = "ndw_participant_id";
const KEY_STUDY_CODE = "ndw_study_code";

function generateId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older browsers
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

/** Returns the persistent participant UUID for this device/browser. */
export function getParticipantId(): string {
  let id = localStorage.getItem(KEY_USER_ID);
  if (!id) {
    id = generateId();
    localStorage.setItem(KEY_USER_ID, id);
  }
  return id;
}

/** Call this after successful enrollment to persist the study code. */
export function saveStudyCode(code: string): void {
  localStorage.setItem(KEY_STUDY_CODE, code);
}

/** Returns the stored study code, or null if not yet enrolled. */
export function getStudyCode(): string | null {
  return localStorage.getItem(KEY_STUDY_CODE);
}

/** Clear identity — use on withdraw or explicit sign-out. */
export function clearParticipantIdentity(): void {
  localStorage.removeItem(KEY_USER_ID);
  localStorage.removeItem(KEY_STUDY_CODE);
}
