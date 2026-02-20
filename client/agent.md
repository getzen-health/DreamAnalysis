# Agent Guide — Client

## File Naming

- Pages: `kebab-case.tsx` (e.g., `brain-monitor.tsx`)
- Components: `kebab-case.tsx` (e.g., `emotion-wheel.tsx`)
- Hooks: `use-kebab-case.tsx` (e.g., `use-auth.tsx`)
- Lib: `kebab-case.ts` (e.g., `ml-api.ts`)

## Import Patterns

Always use `@/` alias (maps to `client/src/`). No relative imports.

```tsx
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { mlApi } from "@/lib/ml-api";
```

## Component Patterns

- Default exports for pages
- Context + hook pattern for shared state (auth, device, theme)
- Every authenticated page wraps with `<AppLayout>` in `App.tsx`
- shadcn/ui components in `components/ui/` — don't edit these directly

## State Management

- **Server state**: TanStack Query (`useQuery`, `useMutation`) via `lib/queryClient.ts`
- **Device state**: `useDevice()` context (connection, streaming status)
- **Auth state**: `useAuth()` context (user, login, logout)
- **No Redux** — all state is either server-cached or context-based

## Styling

- Tailwind CSS only — no CSS modules, no styled-components
- Dark theme by default (`ThemeProvider` in `App.tsx`)
- Use `cn()` utility from `lib/utils.ts` for conditional classes
- Colors from shadcn/ui theme tokens (e.g., `bg-background`, `text-foreground`)

## API Calls

Two separate API backends:

| Client | Talks To | Use For |
|--------|----------|---------|
| `lib/queryClient.ts` | Express (:5000) | DB operations, AI chat, settings |
| `lib/ml-api.ts` | FastAPI (:8000) | EEG analysis, model inference, devices |

## Charts

- **Recharts** — preferred for new charts
- **Chart.js** — used in some existing pages
- Chart components live in `components/charts/`

## Data Rules (Non-Negotiable)

- **Never hardcode, mock, or seed fake data** into any chart, component, or hook.
- All chart values must come from live `useDevice()` frames or real API responses (`useQuery`).
- If no real data exists for a time period, render the empty state (message + icon). Never fill it with placeholder values.
