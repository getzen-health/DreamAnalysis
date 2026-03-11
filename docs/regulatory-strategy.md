# Regulatory Strategy — AntarAI (2026)

## Decision: Stay Wellness, Build Toward DTx

**Current classification**: General Wellness device (FDA-exempt)
**Recommended path**: Wellness now → collect evidence → target De Novo DTx in 24–36 months
**Decision date**: March 2026

---

## The Two Paths

### Path A: Remain Wellness (Current)

Under FDA's January 2026 General Wellness guidance, noninvasive wearables that make no disease claims and promote general wellness are **not regulated as medical devices**.

**Requirements to stay wellness:**
- No disease claims ("screens for depression," "treats anxiety")
- No clinical thresholds ("your stress is clinically elevated")
- No treatment prompts ("take this medication")
- Data shown for "awareness" and "general wellness" only

**What we can say:**
- "Track your emotional patterns over time"
- "See how your brain responds to different activities"
- "Understand your stress and focus trends"

**What we cannot say:**
- "Diagnose," "treat," "prevent," "cure" any condition
- "Clinically validated" without IRB study + peer-reviewed paper

**Timeline**: Already compliant. Ship now.
**Cost**: $0 regulatory.

---

### Path B: DTx Clearance (Future Option)

**DTx market status (2026):**
- 50+ FDA-cleared DTx apps
- Medicare HCPCS billing codes for Digital Mental Health Treatment Devices (Nov 2024)
- CMS expanded codes to include ADHD DTx (Nov 2025)
- FDA TEMPO pilot (Dec 2025): prescribe before full clearance

**What DTx clearance enables:**
- "Adjunct treatment for generalized anxiety disorder" (like DaylightRx)
- Insurance reimbursement (Medicare, Medicaid, private)
- B2B sales to healthcare systems and employers
- Prescribable by clinicians

**Cost:** $500K–2M for clinical trial + De Novo submission
**Timeline:** 12–24 months after RCT completion

---

## Recommended Hybrid Roadmap

### Phase 1 — Now (Wellness, ship immediately)
- Deploy as wellness app, no clinical claims
- Collect real-world usage data with opt-in consent
- Run food-emotion + supplement pilot (n=10–15)
- Submit research paper with honest accuracy numbers

### Phase 2 — 6–12 months (Evidence building)
- Publish peer-reviewed paper on emotion detection accuracy
- Complete 10-person feasibility pilot (see issue #200)
- Partner with university IRB for supervised data collection
- Document intervention effectiveness (breathing → stress reduction delta)

### Phase 3 — 12–24 months (DTx evaluation)
- If pilot data shows measurable intervention efficacy: initiate small RCT (n=30–50)
- Target De Novo classification as "emotion awareness wellness device with behavioral intervention"
- Consider FDA TEMPO pilot for early prescribability
- Explore employer wellness plan reimbursement (no FDA clearance required)

### Phase 4 — 24–36 months (Optional DTx path)
- If RCT positive + paper published: file De Novo submission
- Target: "adjunct support for stress and anxiety management" (not treatment)
- Explore Medicare DMHT billing codes

---

## Decision Matrix

| Factor | Wellness (Now) | DTx (Future) |
|---|---|---|
| Time to market | ✅ Immediate | ❌ 12–24 months |
| Regulatory cost | ✅ $0 | ❌ $500K–2M |
| Claims allowed | Awareness only | "Treats anxiety" etc. |
| Insurance reimbursement | ❌ No | ✅ Medicare + private |
| Revenue model | B2C subscription | B2B + insurance billing |
| Competitive moat | Moderate | Very high |
| Required for launch | No | No |
| Required for clinical partnerships | No | Yes |

**Verdict**: Wellness path now. Build evidence. Decide on DTx after pilot data.

---

## Compliance Checklist (Wellness, Current)

- [x] No disease diagnosis claims in UI
- [x] No clinical threshold alerts ("your stress is dangerously high")
- [x] No treatment recommendations ("you should take medication")
- [x] "For wellness and general awareness purposes" disclaimer in settings
- [x] User consent for EEG/voice data collection
- [x] Data stored locally or encrypted in transit (Neon PostgreSQL, HTTPS)
- [ ] HIPAA: not required for wellness apps (we handle no PHI as a covered entity), but BAA may be needed if partnering with healthcare providers
- [ ] IRB approval needed before publishing any clinical claims

---

## Claims Language Guide

| Claim | Allowed? | Alternative |
|---|---|---|
| "Detects depression" | ❌ No | "Track mood patterns over time" |
| "Treats anxiety" | ❌ No | "Breathing exercises for relaxation" |
| "Clinically validated" | ❌ (without IRB) | "Research-backed methodology" |
| "Medical-grade accuracy" | ❌ No | "Consumer EEG accuracy: 65–75% cross-subject" |
| "Reduces stress by 30%" | ❌ (without RCT) | "Users report feeling calmer after breathing sessions" |
| "Screens for ADHD" | ❌ No | "Tracks attention and focus patterns" |
| "FDA approved" | ❌ No | "General wellness device, FDA-exempt" |

---

## Sources

- [FDA General Wellness Guidance 2026](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-wellness-policy-low-risk-devices)
- [DTx Complete Guide 2026](https://worldclassblogs.com/digital-therapeutics-complete-guide-prescription-apps-medicine/)
- [Medicare DTx Reimbursement Codes (APA)](https://www.apaservices.org/practice/business/technology/tech-talk/reimbursement-pathways-digital-therapeutics)
- [FDA TEMPO Pilot (Healthcare Brew 2026)](https://www.healthcare-brew.com/stories/2026/01/29/new-digital-therapeutics-pilot)
