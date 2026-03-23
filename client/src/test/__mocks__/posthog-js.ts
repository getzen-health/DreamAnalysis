/**
 * Mock for posthog-js — used in test environment since posthog-js
 * is not installed (it's a placeholder dependency).
 */
const posthog = {
  init: () => {},
  capture: () => {},
  identify: () => {},
  reset: () => {},
};

export default posthog;
