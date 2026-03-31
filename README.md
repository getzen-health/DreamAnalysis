# GetZen Health — Marketing Website

The official landing page for [getzen.health](https://getzen.health) — a unified mind–body intelligence app that reads your voice, decodes your dreams, tracks nutrition, and maps your emotions every day.

## Tech Stack

- **Framework:** Next.js 16.2.1 (App Router)
- **Language:** TypeScript
- **Styling:** Tailwind CSS + custom CSS (glass morphism design system)
- **Fonts:** Orbitron (headings) + Inter (body) via `next/font/google`
- **Deployment:** Vercel → getzen.health

## Project Structure

```
app/
├── page.tsx        # Full landing page (single-page)
├── layout.tsx      # Root layout with font config and metadata
└── globals.css     # Design system: glass cards, buttons, animations, gradients

public/
└── logo.png        # GetZen brand logo
```

## Design System

| Class | Purpose |
|---|---|
| `.font-display` | Orbitron font for headings |
| `.glass` | Glass morphism card with hover effect |
| `.glass-bright` | Brighter glass panel |
| `.btn` | Primary gradient button |
| `.btn-outline` | Ghost button |
| `.tag` | Small category pill |
| `.g` | Green–cyan gradient text |
| `.gp` | Purple–green gradient text |
| `.gc` | Tri-color gradient text |
| `.a-float` | Floating animation |
| `.a-marquee` | Scrolling marquee animation |

## Sections

1. **Hero** — Logo, headline, CTA buttons, stats bar, phone mockup with floating metric chips
2. **Marquee** — Scrolling feature list
3. **Features** — 9-card grid covering all 7 biometric layers
4. **Dream Deep-dive** — Dream journal + AI interpretation detail
5. **Nutrition Deep-dive** — Food–mood correlation detail
6. **How It Works** — 3-step morning ritual
7. **Built for Real Life** — 5 lifestyle images (voice, exercise, eating, breathing, mindfulness)
8. **Compare** — GetZen vs Calm / Headspace / Oura / Muse
9. **Waitlist** — Email capture
10. **Footer**

## Running Locally

```bash
npm install
npm run dev
# → http://localhost:3000
```

## Deploying

```bash
vercel --prod
```

Domain is configured at Spaceship.com with:
- `A @ → 76.76.21.21`
- `CNAME www → cname.vercel-dns.com`

## About GetZen

GetZen Health is a mobile app (iOS + Android) powered by 16 ML models that unifies:
- **Voice emotion analysis** — 30-second morning voice check-in
- **Dream decoding** — AI interprets dream fragments
- **Sleep staging** — 92.98% accurate REM/deep/light detection
- **Nutrition + food–mood correlation** — 6 eating states
- **Emotion & mood tracking** — daily and longitudinal
- **Breathwork** — 7 guided exercises with HRV guidance
- **Peak focus forecasting** — circadian-aware performance windows
- **EEG neurofeedback** — Muse 2 / Muse S integration
