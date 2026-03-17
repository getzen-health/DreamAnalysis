# Onboarding Flow Design

**Goal:** Give new users a quick orientation before they explore the app on their own.

**Target users:** Wellness users (no hardware) and EEG hobbyists (Muse headband owners).

## Design

3 swipeable welcome cards shown once after first registration/login.

### Card 1: Welcome
- "Welcome to AntarAI"
- Tagline: track emotions, stress, and brain activity
- App logo/icon

### Card 2: How It Works
- "Voice, EEG, or Both"
- Works with just your voice for mood/stress analysis
- Connect a Muse headband for full brain monitoring

### Card 3: Get Started
- "Let's Explore"
- "Your dashboard is ready. Start by trying a voice check-in or recording a dream."
- Get Started button

## Behavior
- Shown once after first registration/login
- `localStorage` flag: `onboarding_complete=true`
- Swipeable left/right with dot indicators
- Skip button on every card
- "Get Started" on last card navigates to dashboard
- Dashboard unchanged — empty sections show "No data yet"

## Scope
- Frontend only — no backend/API changes
- New file: `client/src/pages/onboarding.tsx`
- Modify: `client/src/App.tsx` (add route)
- Modify: `client/src/pages/auth.tsx` (redirect to /onboarding after register)
