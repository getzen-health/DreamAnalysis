# Lib — Utilities

Shared utilities, API clients, and helper modules used across the frontend.

## All Files

| File | Purpose |
|------|---------|
| `ml-api.ts` | API client for FastAPI ML backend (:8000) — EEG analysis, models, devices |
| `queryClient.ts` | TanStack Query client for Express backend (:5000) — DB, auth, AI chat |
| `data-simulation.ts` | Generates simulated EEG/health data for demo mode |
| `eeg-features.ts` | Client-side EEG feature extraction (band powers, Hjorth params) |
| `ml-local.ts` | Local ML inference using ONNX Runtime in the browser |
| `offline-store.ts` | IndexedDB offline data storage for PWA mode |
| `openai.ts` | OpenAI API integration helpers |
| `utils.ts` | General utilities — `cn()` for Tailwind class merging |

## Two API Paths

The frontend talks to two separate backends:

| Need | Client | Backend | Examples |
|------|--------|---------|----------|
| DB, auth, AI chat | `queryClient.ts` | Express (:5000) | Save dream, fetch settings, chat with AI |
| ML inference | `ml-api.ts` | FastAPI (:8000) | Analyze EEG, get emotions, connect device |

`queryClient.ts` uses TanStack Query for caching and automatic refetching. `ml-api.ts` uses direct fetch calls.
