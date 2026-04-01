import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement scrollIntoView
Element.prototype.scrollIntoView = () => {};

// vitest v4 + jsdom: --localstorage-file path issue disables localStorage.
// Proxy-based mock so that:
//   • Object.keys(localStorage)  → returns only data keys (not method names)
//   • vi.spyOn(localStorage, 'setItem') → correctly intercepts calls
//   • localStorage.clear() / removeItem() work between tests
const _lsData = new Map<string, string>();
const _lsMethods = {
  getItem(key: string): string | null { return _lsData.get(key) ?? null; },
  setItem(key: string, value: string): void { _lsData.set(key, String(value)); },
  removeItem(key: string): void { _lsData.delete(key); },
  clear(): void { _lsData.clear(); },
  key(i: number): string | null { return Array.from(_lsData.keys())[i] ?? null; },
  get length(): number { return _lsData.size; },
};
const _localStorageMock = new Proxy(_lsMethods, {
  get(target, prop: string | symbol) {
    // Data values shadow methods (avoids naming collisions with real key names)
    if (typeof prop === "string" && _lsData.has(prop)) return _lsData.get(prop);
    return (target as Record<string, unknown>)[prop as string];
  },
  set(target, prop: string | symbol, value: unknown) {
    if (typeof prop === "string" && prop in target) {
      // Allow vi.spyOn to replace method properties
      (target as Record<string, unknown>)[prop] = value;
    } else if (typeof prop === "string") {
      _lsData.set(prop, String(value));
    }
    return true;
  },
  // Object.keys(localStorage) returns only data keys, not method names
  ownKeys() { return Array.from(_lsData.keys()); },
  has(target, prop) {
    return (typeof prop === "string" && _lsData.has(prop)) || prop in target;
  },
  getOwnPropertyDescriptor(target, prop) {
    if (typeof prop === "string" && _lsData.has(prop)) {
      return { value: _lsData.get(prop), enumerable: true, configurable: true, writable: true };
    }
    // Also expose method properties so vi.spyOn can find and replace them
    if (typeof prop === "string" && prop in target) {
      return { value: (target as Record<string, unknown>)[prop], enumerable: false, configurable: true, writable: true };
    }
    return undefined;
  },
});

Object.defineProperty(window, "localStorage", {
  value: _localStorageMock,
  writable: true,
});
