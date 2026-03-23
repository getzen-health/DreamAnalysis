# AntarAI Business Roadmap

Consolidated action items from issues #539, #520, #519, #518, #516, #509, #503, #502, #501, #488.
Each issue is documented below with findings and next steps. Close all as "documented in roadmap."

---

## #539: Revenue Model & Monetization Strategy

### Findings
- Consumer EEG wellness is a growing but niche market
- Competitors: Muse (meditation subscription), Emotiv (B2B licensing), Neurable (productivity)
- Key differentiator: multi-model emotion tracking + dream analysis + health correlations

### Action Items
- [ ] Define free tier vs premium tier feature split
- [ ] Evaluate subscription pricing ($9.99/mo, $49.99/yr comparable to Muse)
- [ ] Explore B2B licensing for research institutions (anonymized data access)
- [ ] Consider one-time purchase model for privacy-conscious users
- [ ] Draft pricing page copy and A/B test with early users

---

## #520: Partnership Strategy — Wearable Manufacturers

### Findings
- Muse 2/S is primary hardware, but user base is limited (~200K active devices)
- Emotiv EPOC Flex, NeuroSky MindWave, OpenBCI are potential targets
- Hardware partnerships increase addressable market significantly

### Action Items
- [ ] Reach out to InteraXon (Muse) for official app listing partnership
- [ ] Evaluate Emotiv SDK integration (14-channel, broader research community)
- [ ] Draft partnership pitch deck (what we offer: ML pipeline, what we need: hardware access)
- [ ] Investigate OpenBCI Cyton board support via BrainFlow (already supported)
- [ ] Create hardware compatibility matrix for marketing materials

---

## #519: Clinical Validation Roadmap

### Findings
- Current accuracy (feature-based heuristics): 55-65% discrete emotions, 65-75% binary valence
- IRB approval needed for any clinical claims
- Regulatory path documented in `docs/regulatory-strategy.md`

### Action Items
- [ ] Identify university partner for IRB-approved validation study
- [ ] Define study protocol: N=50+ subjects, controlled emotion induction, Muse 2
- [ ] Budget: $15-25K for a basic validation study (participant compensation + equipment)
- [ ] Target publications: IEEE EMBC, Frontiers in Neuroscience, JMIR
- [ ] Milestone: peer-reviewed paper within 12 months of study start

---

## #518: Marketing & Launch Strategy

### Findings
- Target audience: quantified-self enthusiasts, meditation practitioners, researchers
- Key channels: Product Hunt, Hacker News, Reddit r/Muse r/EEG, Twitter/X neurotech community
- Content marketing: blog posts on EEG science, dream analysis, emotion tracking

### Action Items
- [ ] Prepare Product Hunt launch (title, tagline, screenshots, maker story)
- [ ] Write 3 launch blog posts: (1) How EEG emotion tracking works, (2) Dream detection science, (3) Building a BCI app
- [ ] Create demo video (2 min): headband setup, live EEG, emotion classification
- [ ] Build email waitlist landing page
- [ ] Plan beta program: 50 users, 4 weeks, feedback surveys

---

## #516: User Onboarding Flow

### Findings
- Current onboarding: none (lands directly on dashboard)
- First-time users need: (1) hardware setup, (2) baseline calibration, (3) feature tour
- Calibration is critical for accuracy (+15-29% improvement)

### Action Items
- [ ] Design 5-step onboarding wizard:
  1. Welcome + value proposition
  2. Connect Muse headband (BLE pairing)
  3. Run 2-minute baseline calibration
  4. Quick tour of key features (emotions, dreams, analytics)
  5. Set notification preferences
- [ ] Implement onboarding as a gated first-run experience
- [ ] Track onboarding completion rate and drop-off points
- [ ] Add "re-run calibration" prompt after 7 days

---

## #509: Data Retention & Compliance Audit

### Findings
- Current retention: indefinite while account is active
- GDPR requires clear retention policy + right to erasure (Article 17)
- CCPA requires disclosure of data categories collected
- Privacy policy already covers these, but enforcement is manual

### Action Items
- [ ] Implement automated data expiry: 90-day default for raw session data
- [ ] Add server-side cron job to purge expired data
- [ ] Build admin dashboard for data deletion request tracking
- [ ] Conduct annual privacy audit checklist
- [ ] Document data flow diagram (collection -> processing -> storage -> deletion)

---

## #503: Competitive Analysis Update

### Findings
- Documented in `docs/competitive-analysis.md`
- Key competitors: Muse (meditation), Emotiv (research), Neurable (productivity), BrainCo (education)
- Our unique angle: multi-modal (EEG + voice + health), dream analysis, open science approach

### Action Items
- [ ] Update competitive analysis quarterly
- [ ] Track competitor feature releases and pricing changes
- [ ] Identify feature gaps vs top 3 competitors
- [ ] Document our unique differentiators for pitch materials
- [ ] Monitor app store reviews of competitor apps for user pain points

---

## #502: Investor Pitch Materials

### Findings
- Need: pitch deck, one-pager, financial projections
- Market: consumer neurotechnology projected at $3.4B by 2027 (Grand View Research)
- Traction needed: user count, session metrics, retention rates

### Action Items
- [ ] Create 12-slide pitch deck:
  1. Problem (emotional self-awareness gap)
  2. Solution (AntarAI multi-modal brain wellness)
  3. Demo
  4. Market size
  5. Business model
  6. Traction/metrics
  7. Technology (16 ML models, 9 datasets)
  8. Team
  9. Competition
  10. Go-to-market
  11. Financials
  12. Ask
- [ ] Build analytics dashboard for key metrics (DAU, sessions/day, retention)
- [ ] Draft one-pager for cold outreach
- [ ] Identify 10 target investors in health-tech / neurotech space

---

## #501: Community & Open Source Strategy

### Findings
- Research module is open-source ready (IRB protocol, consent flow, data export)
- Community interest in: EEG preprocessing pipelines, emotion datasets, BCI tutorials
- Open source builds trust and attracts contributors

### Action Items
- [ ] Open source the ML preprocessing pipeline (eeg_processor.py, signal processing)
- [ ] Create a "Contributing" guide for researchers
- [ ] Build a public benchmark leaderboard for emotion classification
- [ ] Host monthly virtual meetups for neurotech community
- [ ] Publish training scripts and model weights (non-proprietary models)

---

## #488: Accessibility & Internationalization

### Findings
- App is English-only, no accessibility features beyond basic Radix/shadcn defaults
- EEG terminology is jargon-heavy for general users
- Target markets: US, EU, India, Japan (significant meditation/wellness markets)

### Action Items
- [ ] Audit app for WCAG 2.1 AA compliance
- [ ] Add screen reader labels to all chart components
- [ ] Implement i18n framework (react-intl or i18next)
- [ ] Prioritize translations: English, Spanish, Hindi, Japanese
- [ ] Add "plain language" mode for non-technical users (replace EEG jargon)
- [ ] Ensure color contrast meets AA standards in both themes

---

## Timeline Summary

### Q2 2026 (Now)
- Onboarding flow (#516)
- Data retention automation (#509)
- Competitive analysis update (#503)
- ASO optimization (completed — #517)

### Q3 2026
- Beta launch + marketing (#518)
- Pitch materials (#502)
- Partnership outreach (#520)
- Accessibility audit (#488)

### Q4 2026
- Revenue model implementation (#539)
- Clinical validation study start (#519)
- Community/open-source launch (#501)

---

*Last updated: 2026-03-23*
