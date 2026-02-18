# Hooks

Custom React hooks providing shared state and functionality across pages.

## All Hooks

| Hook | File | Purpose | Data Source |
|------|------|---------|-------------|
| `useAuth()` | `use-auth.tsx` | User authentication state, login/logout | Express API |
| `useDevice()` | `use-device.tsx` | EEG device connection, streaming status | FastAPI + BrainFlow |
| `useInference()` | `use-inference.ts` | Client-side ONNX model inference | Local ONNX Runtime |
| `useMetrics()` | `use-metrics.tsx` | Health metric data fetching + caching | Express API |
| `useMobile()` | `use-mobile.tsx` | Responsive breakpoint detection | Window resize |
| `useTheme()` | `use-theme.tsx` | Light/dark theme toggle + persistence | localStorage |
| `useToast()` | `use-toast.ts` | Toast notification dispatch | Local state |

## Pattern

Hooks that manage global state use React Context:

```tsx
// Provider wraps the app in App.tsx
<AuthProvider>
  <DeviceProvider>
    <ThemeProvider>
      ...
    </ThemeProvider>
  </DeviceProvider>
</AuthProvider>

// Any component can consume
const { user, login, logout } = useAuth();
const { connected, startStream } = useDevice();
```
