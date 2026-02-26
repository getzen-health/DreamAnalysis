# Agent Rules — Neural Dream Workshop

These rules apply to ALL agents and Claude sessions working on this codebase.
They are non-negotiable and override any default behavior.

---

## Rule 1: No Redundancy

**Each metric, label, or data point must appear in exactly ONE place in the UI.**

- If stress is shown on the Dashboard (live EEG), it must NOT also appear on Health Analytics as a primary stat — or vice versa.
- If a chart exists on one page, do not duplicate it on another page with the same data.
- Before adding any new display of a metric, search the codebase to confirm it is not already shown elsewhere.
- If the same value is shown in two places and one is redundant, REMOVE the redundant one.

**Canonical locations:**
| Metric | Primary Page | Acceptable Secondary |
|--------|-------------|----------------------|
| Live stress / focus / relaxation | Dashboard (live EEG tab only) | None |
| Historical avg stress / focus | Health Analytics | None |
| Emotion readings | Emotion Lab | None |
| Dream analysis | Dream Journal | None |
| Food logs | Food Log | None |

---

## Rule 2: Consistent Units

**All numeric values must use the same unit everywhere they appear.**

- Percentage values → always append `%` (e.g., `Stress 85%`, not `Stress 85`)
- Index values (0–1) → always convert to 0–100% before display, then append `%`
- Never show the same metric as `85` in one place and `85%` in another
- Never mix raw scores with percentages for the same metric

---

## Rule 3: Live vs Historical — Clearly Labeled

- Live EEG values (from current session) must be labeled with context (e.g., "Live", "Now", or session timestamp)
- Historical values (from DB averages) must be labeled "Avg", "7-day avg", or similar
- A user should never be confused whether a number is "right now" or "over time"

---

## Rule 4: No Compensation Language

- Do NOT add any references to compensation, payment, gift cards, or monetary rewards anywhere in the codebase, UI, or documents.
- This includes: "$5/day", "$150", "$25 bonus", "gift card", "payment", "stipend"
- This is a volunteer study.

---

## Rule 5: No Claude Co-Author in Git

- NEVER add `Co-Authored-By: Claude` or any AI co-author line to commit messages.
- Claude must not appear as a contributor in git history.

---

## Rule 6: Vercel Route Coverage

- Every new API route added to `server/routes.ts` (Express) MUST also be added to `api/[...path].ts` (Vercel catch-all).
- The Vercel catch-all is the ONLY serverless function for all study, food, and custom routes.
- Auth routes have dedicated files in `api/auth/*.ts` — do not duplicate them in the catch-all.

---

## Rule 7: Study Routes Architecture

- All `/api/study/*` routes live in `api/[...path].ts` only
- All `/api/food/*` routes live in `api/[...path].ts` only
- The Vercel SPA rewrite must EXCLUDE `/api/` prefix: `{ "source": "/((?!api/).*)", "destination": "/index.html" }`

---

*Last updated: Feb 2026*
