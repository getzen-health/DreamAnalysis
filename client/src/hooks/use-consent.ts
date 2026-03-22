/**
 * useConsent — React hook for biometric consent state.
 *
 * Reads consent state on mount, provides toggle/acceptAll/rejectAll actions.
 * Components that collect biometric data should call `consent.eeg` (etc.)
 * before proceeding.
 */

import { useState, useCallback } from "react";
import {
  getConsentState,
  saveConsentState,
  DEFAULT_CONSENT_STATE,
  type BiometricConsentState,
  type ConsentModality,
} from "@/lib/consent-store";

export function useConsent() {
  const [state, setState] = useState<BiometricConsentState>(getConsentState);

  const toggle = useCallback((modality: ConsentModality, value: boolean) => {
    setState((prev) => {
      const next = { ...prev, [modality]: value };
      saveConsentState(next);
      return next;
    });
  }, []);

  const acceptAll = useCallback(() => {
    const allOn: BiometricConsentState = {
      eeg: true,
      voice: true,
      health: true,
      nutrition: true,
      location: true,
    };
    saveConsentState(allOn);
    setState(allOn);
  }, []);

  const rejectAll = useCallback(() => {
    const allOff = { ...DEFAULT_CONSENT_STATE };
    saveConsentState(allOff);
    setState(allOff);
  }, []);

  return {
    ...state,
    toggle,
    acceptAll,
    rejectAll,
  };
}
