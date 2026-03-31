"use client";
import { useState, useEffect, useRef } from "react";
import {
  ArrowRight, Mic, Brain, Moon, Wind, Activity, Zap,
  TrendingUp, Sparkles, ChevronRight, Apple,
  Eye, BarChart2, Check, X, Smartphone, Play
} from "lucide-react";

/* ─── Animated counter ─── */
function Counter({ target, suffix = "" }: { target: number; suffix?: string }) {
  const [val, setVal] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      obs.disconnect();
      let start = 0;
      const step = target / 40;
      const t = setInterval(() => {
        start += step;
        if (start >= target) { setVal(target); clearInterval(t); }
        else setVal(Math.floor(start));
      }, 30);
    }, { threshold: 0.5 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, [target]);
  return <span ref={ref}>{val}{suffix}</span>;
}

/* ─── Phone mockup ─── */
function Phone({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      width: 264, borderRadius: 46, padding: "10px 7px",
      background: "linear-gradient(160deg,#1a1e35,#0b0d18)",
      border: "1.5px solid rgba(255,255,255,0.12)",
      boxShadow: "0 70px 140px rgba(0,0,0,.85), 0 0 0 1px rgba(255,255,255,.04), inset 0 1px 0 rgba(255,255,255,.08)",
      flexShrink: 0, position: "relative",
    }}>
      {/* Reflection */}
      <div style={{ position: "absolute", inset: 0, borderRadius: 46, background: "linear-gradient(135deg,rgba(255,255,255,.04) 0%,transparent 50%)", pointerEvents: "none" }} />
      <div style={{ width: 88, height: 24, background: "#0b0d18", borderRadius: 999, margin: "0 auto 8px", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#1a1d2e" }} />
      </div>
      <div style={{ borderRadius: 36, overflow: "hidden", background: "#070811", minHeight: 490 }}>
        {children}
      </div>
    </div>
  );
}

/* ─── Screen: Brain Report ─── */
function ScreenBrain() {
  return (
    <div style={{ padding: "20px 16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
        <div>
          <p style={{ fontSize: 8, color: "rgba(255,255,255,.3)", letterSpacing: ".1em", textTransform: "uppercase" }}>Daily Brain Report</p>
          <p style={{ fontSize: 13, fontWeight: 700, marginTop: 2 }}>Good morning</p>
        </div>
        <div style={{ width: 30, height: 30, borderRadius: 9, background: "rgba(52,211,153,.12)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Brain size={14} color="#34D399" />
        </div>
      </div>
      <div style={{ textAlign: "center", padding: "16px 0 18px", marginBottom: 14, background: "rgba(255,255,255,.025)", borderRadius: 16, border: "1px solid rgba(255,255,255,.05)" }}>
        <p style={{ fontSize: 8, color: "rgba(255,255,255,.3)", letterSpacing: ".08em", textTransform: "uppercase", marginBottom: 4 }}>Readiness Score</p>
        <p style={{ fontSize: 58, fontWeight: 800, lineHeight: 1, background: "linear-gradient(135deg,#34D399,#22D3EE)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>84</p>
        <p style={{ fontSize: 9, color: "rgba(255,255,255,.3)", marginTop: 4 }}>Peak focus · 9 am – 12 pm</p>
        <div style={{ display: "flex", justifyContent: "center", gap: 4, marginTop: 8 }}>
          {["Calm","Focused","Rested"].map(t => (
            <span key={t} style={{ fontSize: 8, padding: "2px 7px", borderRadius: 999, background: "rgba(52,211,153,.1)", color: "#34D399", border: "1px solid rgba(52,211,153,.2)" }}>{t}</span>
          ))}
        </div>
      </div>
      {[{l:"Focus",p:82,c:"#34D399"},{l:"Stress",p:22,c:"#22D3EE"},{l:"Energy",p:74,c:"#a78bfa"},{l:"Recovery",p:91,c:"#fb923c"}].map(b => (
        <div key={b.l} style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 9, color: "rgba(255,255,255,.5)" }}>{b.l}</span>
            <span style={{ fontSize: 9, color: b.c, fontWeight: 600 }}>{b.p}</span>
          </div>
          <div style={{ height: 5, borderRadius: 99, background: "rgba(255,255,255,.06)", overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${b.p}%`, borderRadius: 99, background: b.c, opacity: .85 }} />
          </div>
        </div>
      ))}
      <div style={{ marginTop: 14, padding: "10px 12px", borderRadius: 12, background: "rgba(52,211,153,.06)", border: "1px solid rgba(52,211,153,.15)", display: "flex", gap: 8, alignItems: "flex-start" }}>
        <Mic size={11} color="#34D399" style={{ marginTop: 1, flexShrink: 0 }} />
        <p style={{ fontSize: 8.5, color: "rgba(255,255,255,.55)", lineHeight: 1.5 }}>Voice tone: confident · Stress markers below baseline · Mood trending positive</p>
      </div>
    </div>
  );
}

/* ─── Screen: Dream Journal ─── */
function ScreenDream() {
  return (
    <div style={{ padding: "20px 16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <p style={{ fontSize: 8, color: "rgba(255,255,255,.3)", letterSpacing: ".1em", textTransform: "uppercase" }}>Dream Journal</p>
          <p style={{ fontSize: 13, fontWeight: 700, marginTop: 2 }}>Last Night</p>
        </div>
        <div style={{ width: 30, height: 30, borderRadius: 9, background: "rgba(167,139,250,.12)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Moon size={14} color="#a78bfa" />
        </div>
      </div>
      <div style={{ padding: "12px 14px", borderRadius: 14, background: "rgba(167,139,250,.06)", border: "1px solid rgba(167,139,250,.15)", marginBottom: 12 }}>
        <p style={{ fontSize: 8.5, color: "rgba(255,255,255,.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: 6 }}>Your words</p>
        <p style={{ fontSize: 9, color: "rgba(255,255,255,.65)", lineHeight: 1.6, fontStyle: "italic" }}>"ocean... falling... old house... someone calling my name... bright light..."</p>
      </div>
      <div style={{ padding: "12px 14px", borderRadius: 14, background: "rgba(34,211,238,.05)", border: "1px solid rgba(34,211,238,.15)", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 6 }}>
          <Sparkles size={10} color="#22D3EE" />
          <p style={{ fontSize: 8, color: "#22D3EE", letterSpacing: ".08em", textTransform: "uppercase" }}>AI Interpretation</p>
        </div>
        <p style={{ fontSize: 9, color: "rgba(255,255,255,.6)", lineHeight: 1.55 }}>Themes of transition and nostalgia. The bright light suggests clarity-seeking during a period of change. Emotional tone: reflective.</p>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        {[{l:"REM",v:"1h 42m",c:"#a78bfa"},{l:"Deep",v:"1h 18m",c:"#34D399"},{l:"Quality",v:"87%",c:"#22D3EE"}].map(s => (
          <div key={s.l} style={{ flex: 1, padding: "8px 6px", borderRadius: 10, background: "rgba(255,255,255,.03)", border: "1px solid rgba(255,255,255,.06)", textAlign: "center" }}>
            <p style={{ fontSize: 11, fontWeight: 700, color: s.c }}>{s.v}</p>
            <p style={{ fontSize: 7.5, color: "rgba(255,255,255,.3)", marginTop: 2 }}>{s.l}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Screen: Nutrition ─── */
function ScreenNutrition() {
  return (
    <div style={{ padding: "20px 16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <p style={{ fontSize: 8, color: "rgba(255,255,255,.3)", letterSpacing: ".1em", textTransform: "uppercase" }}>Nutrition + Mood</p>
          <p style={{ fontSize: 13, fontWeight: 700, marginTop: 2 }}>Today</p>
        </div>
        <div style={{ width: 30, height: 30, borderRadius: 9, background: "rgba(251,146,60,.12)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Apple size={14} color="#fb923c" />
        </div>
      </div>
      <div style={{ padding: "10px 12px", borderRadius: 12, background: "rgba(52,211,153,.06)", border: "1px solid rgba(52,211,153,.18)", marginBottom: 12 }}>
        <p style={{ fontSize: 8, color: "rgba(255,255,255,.4)", marginBottom: 3 }}>Detected eating state</p>
        <p style={{ fontSize: 11, fontWeight: 600, color: "#34D399" }}>Mindful Eating</p>
        <p style={{ fontSize: 8.5, color: "rgba(255,255,255,.45)", marginTop: 3 }}>Mood elevated 18% after lunch · Energy sustained</p>
      </div>
      <p style={{ fontSize: 8, color: "rgba(255,255,255,.3)", letterSpacing: ".08em", textTransform: "uppercase", marginBottom: 8 }}>Food-Emotion Correlation</p>
      {[{l:"Omega-3",e:"Focus +21%",c:"#22D3EE",p:72},{l:"Magnesium",e:"Sleep +14%",c:"#a78bfa",p:55},{l:"Vitamin D",e:"Mood +17%",c:"#fb923c",p:40}].map(n => (
        <div key={n.l} style={{ marginBottom: 9 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
            <span style={{ fontSize: 9, color: "rgba(255,255,255,.55)" }}>{n.l}</span>
            <span style={{ fontSize: 8.5, color: n.c }}>{n.e}</span>
          </div>
          <div style={{ height: 4, borderRadius: 99, background: "rgba(255,255,255,.06)" }}>
            <div style={{ height: "100%", width: `${n.p}%`, borderRadius: 99, background: n.c, opacity: .75 }} />
          </div>
        </div>
      ))}
    </div>
  );
}

/* ─── Life card ─── */
function LifeCard({ id, alt, h, icon, label, caption, accent }: {
  id: string; alt: string; h: number; icon: React.ReactNode; label: string; caption: string; accent: string;
}) {
  return (
    <div style={{ position: "relative", height: h, borderRadius: 20, overflow: "hidden" }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`https://images.unsplash.com/${id}?w=900&q=80&fit=crop&auto=format`} alt={alt}
        style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", transition: "transform .5s ease" }}
        onMouseEnter={e => (e.currentTarget.style.transform = "scale(1.04)")}
        onMouseLeave={e => (e.currentTarget.style.transform = "scale(1)")}
      />
      <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to top, rgba(7,8,15,.9) 0%, rgba(7,8,15,.3) 55%, transparent 100%)" }} />
      <div style={{ position: "absolute", bottom: 18, left: 18, right: 18 }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: 999,
          background: `${accent}18`, border: `1px solid ${accent}35`, marginBottom: 8 }}>
          {icon}
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase", color: accent }}>{label}</span>
        </div>
        <p style={{ fontSize: 13, color: "rgba(255,255,255,.8)", lineHeight: 1.45, fontWeight: 400 }}>{caption}</p>
      </div>
    </div>
  );
}

/* ─── Waitlist form ─── */
function WaitlistForm() {
  const [email, setEmail] = useState("");
  const [done, setDone] = useState(false);
  return done ? (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "14px 22px", borderRadius: 999, background: "rgba(52,211,153,.1)", border: "1px solid rgba(52,211,153,.3)" }}>
      <div style={{ width: 22, height: 22, borderRadius: "50%", background: "#34D399", display: "flex", alignItems: "center", justifyContent: "center" }}><Check size={13} color="#07080f" strokeWidth={3} /></div>
      <span style={{ fontSize: 14, color: "#34D399", fontWeight: 500 }}>You&apos;re on the list. We&apos;ll be in touch.</span>
    </div>
  ) : (
    <form onSubmit={e => { e.preventDefault(); if (email) setDone(true); }} style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
      <input
        type="email" required value={email} onChange={e => setEmail(e.target.value)}
        placeholder="Enter your email"
        style={{
          flex: "1 1 220px", padding: "13px 22px", borderRadius: 999,
          background: "rgba(255,255,255,.05)", border: "1px solid rgba(255,255,255,.1)",
          color: "#e2e8f0", fontSize: 14, outline: "none", fontFamily: "var(--font-inter)",
        }}
      />
      <button type="submit" className="btn">
        Join waitlist <ArrowRight size={15} />
      </button>
    </form>
  );
}

/* ─── Floating metric chip ─── */
function Chip({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string; color: string }) {
  return (
    <div className="a-border" style={{
      display: "flex", alignItems: "center", gap: 9, padding: "10px 14px",
      background: "rgba(7,8,15,.88)", backdropFilter: "blur(24px)",
      border: `1px solid ${color}30`, borderRadius: 14,
      boxShadow: "0 8px 32px rgba(0,0,0,.5)",
    }}>
      <div style={{ width: 30, height: 30, borderRadius: 9, background: `${color}18`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
        {icon}
      </div>
      <div>
        <p style={{ fontSize: 7.5, color: "rgba(255,255,255,.3)", letterSpacing: ".08em", textTransform: "uppercase", marginBottom: 2 }}>{label}</p>
        <p style={{ fontSize: 12.5, fontWeight: 700, color }}>{value}</p>
      </div>
    </div>
  );
}

/* ─── Compare row ─── */
function CRow({ label, us, them }: { label: string; us: boolean; them: boolean }) {
  const tick = (v: boolean) => v
    ? <div style={{ width: 24, height: 24, borderRadius: "50%", background: "rgba(52,211,153,.15)", display: "flex", alignItems: "center", justifyContent: "center" }}><Check size={13} color="#34D399" strokeWidth={2.5} /></div>
    : <div style={{ width: 24, height: 24, borderRadius: "50%", background: "rgba(255,255,255,.04)", display: "flex", alignItems: "center", justifyContent: "center" }}><X size={11} color="rgba(255,255,255,.18)" strokeWidth={2} /></div>;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 16, alignItems: "center", padding: "13px 0", borderBottom: "1px solid rgba(255,255,255,.04)" }}>
      <span style={{ fontSize: 14, color: "rgba(226,232,240,.65)" }}>{label}</span>
      <div style={{ display: "flex", justifyContent: "center", width: 80 }}>{tick(us)}</div>
      <div style={{ display: "flex", justifyContent: "center", width: 80 }}>{tick(them)}</div>
    </div>
  );
}

/* ─── AI Quote card ─── */
function AIQuote({ screen, quote, color }: { screen: string; quote: string; color: string }) {
  return (
    <div className="bento" style={{ padding: "22px 24px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
        <Sparkles size={13} color={color} />
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".1em", textTransform: "uppercase", color }}>{screen}</span>
      </div>
      <p style={{ fontSize: 14, color: "rgba(226,232,240,.7)", lineHeight: 1.65, fontStyle: "italic" }}>&ldquo;{quote}&rdquo;</p>
    </div>
  );
}

/* ═══════════════════════════════════════
   PAGE
═══════════════════════════════════════ */
export default function Page() {
  const screens = [
    { id: "brain", label: "Brain Report", icon: <Brain size={13} /> },
    { id: "dream", label: "Dream", icon: <Moon size={13} /> },
    { id: "nutrition", label: "Nutrition", icon: <Apple size={13} /> },
  ] as const;
  const [activeScreen, setActiveScreen] = useState<"brain"|"dream"|"nutrition">("brain");

  return (
    <div style={{ position: "relative", overflowX: "hidden" }}>

      {/* ── Nav ── */}
      <nav style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
        padding: "0 32px", height: 64,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "rgba(7,8,15,.75)", backdropFilter: "blur(24px)",
        borderBottom: "1px solid rgba(255,255,255,.05)",
      }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/logo.png" alt="GetZen" style={{ width: 44, height: 44, objectFit: "cover", borderRadius: "50%", border: "2px solid rgba(52,211,153,.35)" }} />
        <div style={{ display: "flex", gap: 32, alignItems: "center" }}>
          {["Features","Science","Compare"].map(l => (
            <a key={l} href={`#${l.toLowerCase()}`} style={{ fontSize: 13, color: "rgba(226,232,240,.45)", textDecoration: "none", transition: "color .2s" }}
              onMouseEnter={e => (e.currentTarget.style.color = "#e2e8f0")}
              onMouseLeave={e => (e.currentTarget.style.color = "rgba(226,232,240,.45)")}
            >{l}</a>
          ))}
          <a href="#waitlist" className="btn" style={{ padding: "9px 20px", fontSize: 13 }}>Early access <ChevronRight size={13} /></a>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "110px 32px 80px", position: "relative", textAlign: "center" }}>
        <div className="a-mesh1" style={{ position: "absolute", top: "5%", left: "5%", width: 700, height: 700, borderRadius: "50%", background: "radial-gradient(circle,rgba(52,211,153,.07) 0%,transparent 70%)", pointerEvents: "none" }} />
        <div className="a-mesh2" style={{ position: "absolute", bottom: "5%", right: "5%", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle,rgba(167,139,250,.07) 0%,transparent 70%)", pointerEvents: "none" }} />
        <div className="a-mesh3" style={{ position: "absolute", top: "40%", left: "40%", width: 400, height: 400, borderRadius: "50%", background: "radial-gradient(circle,rgba(34,211,238,.05) 0%,transparent 70%)", pointerEvents: "none" }} />

        {/* Logo with glow ring */}
        <div className="a-float" style={{ position: "relative", marginBottom: 28 }}>
          <div className="a-spin" style={{ position: "absolute", inset: -8, borderRadius: "50%", background: "conic-gradient(from 0deg,#34D399,#22D3EE,#a78bfa,transparent,#34D399)", opacity: .35, pointerEvents: "none" }} />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="GetZen Health" style={{ width: 280, height: 280, objectFit: "cover", borderRadius: "50%", position: "relative", zIndex: 1 }} />
        </div>

        <div className="tag" style={{ marginBottom: 24 }}>
          <Sparkles size={10} />
          Unified mind–body intelligence
        </div>

        <h1 className="font-display" style={{ fontSize: "clamp(40px,6vw,78px)", fontWeight: 900, lineHeight: 1.0, letterSpacing: "-.03em", marginBottom: 22, maxWidth: 820 }}>
          Know Your <span className="shimmer">Mind</span><br />Every Morning
        </h1>

        <p style={{ fontSize: 18, color: "rgba(226,232,240,.5)", lineHeight: 1.7, maxWidth: 500, marginBottom: 40 }}>
          GetZen reads your voice, decodes your dreams, tracks what you eat, and maps how you feel — one daily picture of your entire inner world.
        </p>

        <div style={{ display: "flex", gap: 14, justifyContent: "center", marginBottom: 56 }}>
          <a href="#waitlist" className="btn">Join the waitlist <ArrowRight size={16} /></a>
          <a href="#features" className="btn-outline">See how it works</a>
        </div>

        {/* Animated stats */}
        <div style={{ display: "flex", gap: 0, marginBottom: 72, background: "rgba(255,255,255,.03)", border: "1px solid rgba(255,255,255,.07)", borderRadius: 18, overflow: "hidden" }}>
          {[{v:16,s:"",l:"AI Models"},{v:92,s:".98%",l:"Sleep Accuracy"},{v:7,s:"",l:"Biometric Layers"}].map((s, i) => (
            <div key={s.l} style={{ padding: "20px 40px", borderRight: i < 2 ? "1px solid rgba(255,255,255,.06)" : "none", textAlign: "center" }}>
              <p className="font-display" style={{ fontSize: 30, fontWeight: 800, background: "linear-gradient(135deg,#34D399,#22D3EE)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", lineHeight: 1 }}>
                <Counter target={s.v} />{s.s}
              </p>
              <p style={{ fontSize: 11, color: "rgba(226,232,240,.3)", marginTop: 6, letterSpacing: ".08em", textTransform: "uppercase" }}>{s.l}</p>
            </div>
          ))}
        </div>

        {/* Interactive screen switcher + phone */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
          {/* Tab switcher */}
          <div style={{ display: "flex", gap: 6, padding: "6px", background: "rgba(255,255,255,.04)", borderRadius: 999, border: "1px solid rgba(255,255,255,.07)" }}>
            {screens.map(s => (
              <button key={s.id} onClick={() => setActiveScreen(s.id)}
                style={{
                  display: "flex", alignItems: "center", gap: 6, padding: "8px 18px", borderRadius: 999,
                  border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600, fontFamily: "var(--font-inter)",
                  background: activeScreen === s.id ? "linear-gradient(135deg,#34D399,#22D3EE)" : "transparent",
                  color: activeScreen === s.id ? "#07080f" : "rgba(255,255,255,.4)",
                  transition: "all .2s",
                }}
              >
                {s.icon} {s.label}
              </button>
            ))}
          </div>

          {/* Phone with chips */}
          <div style={{ position: "relative", width: 560, height: 540, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div className="a-breathe" style={{ position: "absolute", inset: 40, borderRadius: "50%", background: "radial-gradient(circle,rgba(52,211,153,.1) 0%,transparent 65%)", pointerEvents: "none" }} />
            <div style={{ position: "absolute", top: 50, left: 0 }}>
              <Chip icon={<Mic size={13} color="#34D399" />} label="Voice" value="Confident" color="#34D399" />
            </div>
            <div style={{ position: "absolute", top: 50, right: 0 }}>
              <Chip icon={<Moon size={13} color="#a78bfa" />} label="REM sleep" value="1h 42m" color="#a78bfa" />
            </div>
            <div style={{ position: "absolute", bottom: 90, left: 0 }}>
              <Chip icon={<Apple size={13} color="#fb923c" />} label="Eating state" value="Mindful" color="#fb923c" />
            </div>
            <div style={{ position: "absolute", bottom: 90, right: 0 }}>
              <Chip icon={<Wind size={13} color="#22D3EE" />} label="Stress" value="Below avg" color="#22D3EE" />
            </div>
            <div style={{ transition: "opacity .3s", opacity: 1 }}>
              <Phone>
                {activeScreen === "brain" && <ScreenBrain />}
                {activeScreen === "dream" && <ScreenDream />}
                {activeScreen === "nutrition" && <ScreenNutrition />}
              </Phone>
            </div>
          </div>
        </div>
      </section>

      <hr className="divider" />

      {/* ── Marquee ── */}
      <div style={{ overflow: "hidden", padding: "14px 0", background: "rgba(255,255,255,.01)" }}>
        <div className="a-marquee" style={{ display: "flex", gap: 60, whiteSpace: "nowrap" }}>
          {["Voice Emotion Analysis","Dream Interpretation","Sleep Staging","Food–Mood Correlation","EEG Neurofeedback","Breathing Exercises","Peak Focus Forecasting","Cycle Awareness","Emotion Tracking","AI Companion","16 ML Models","Daily Brain Report",
            "Voice Emotion Analysis","Dream Interpretation","Sleep Staging","Food–Mood Correlation","EEG Neurofeedback","Breathing Exercises","Peak Focus Forecasting","Cycle Awareness","Emotion Tracking","AI Companion","16 ML Models","Daily Brain Report"
          ].map((t, i) => (
            <span key={i} style={{ fontSize: 11, color: "rgba(226,232,240,.2)", letterSpacing: ".14em", textTransform: "uppercase" }}>
              <span style={{ color: "rgba(52,211,153,.35)", marginRight: 60 }}>✦</span>{t}
            </span>
          ))}
        </div>
      </div>

      <hr className="divider" />

      {/* ── Bento features ── */}
      <section id="features" style={{ padding: "100px 32px", maxWidth: 1140, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 64 }}>
          <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><Activity size={10} />What GetZen tracks</div>
          <h2 className="font-display" style={{ fontSize: "clamp(28px,4vw,50px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.05, marginBottom: 16 }}>
            Seven layers of <span className="gc">self-knowledge</span>
          </h2>
          <p style={{ fontSize: 16, color: "rgba(226,232,240,.4)", maxWidth: 480, margin: "0 auto" }}>
            Most apps track one signal. GetZen layers seven — then connects the dots.
          </p>
        </div>

        {/* Bento grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>

          {/* Large — Voice */}
          <div className="bento" style={{ gridColumn: "span 2", padding: "36px 32px", display: "flex", gap: 32, alignItems: "center", background: "linear-gradient(135deg,rgba(52,211,153,.06),rgba(34,211,238,.04))" }}>
            <div style={{ flex: 1 }}>
              <div style={{ width: 52, height: 52, borderRadius: 16, background: "rgba(52,211,153,.12)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20 }}>
                <Mic size={24} color="#34D399" />
              </div>
              <h3 className="font-display" style={{ fontSize: 20, fontWeight: 800, marginBottom: 10, letterSpacing: "-.02em" }}>Voice Emotion Analysis</h3>
              <p style={{ fontSize: 14.5, color: "rgba(226,232,240,.5)", lineHeight: 1.65 }}>Speak 30 seconds each morning. GetZen extracts stress markers, emotional tone, and energy — no wearable required. Your voice knows things your mind hasn&apos;t admitted yet.</p>
            </div>
            <div style={{ padding: "20px 18px", background: "rgba(255,255,255,.03)", borderRadius: 16, border: "1px solid rgba(52,211,153,.12)", minWidth: 160 }}>
              {[{l:"Tone",v:"Confident",c:"#34D399"},{l:"Stress",v:"Low",c:"#22D3EE"},{l:"Energy",v:"High",c:"#a78bfa"},{l:"Clarity",v:"Clear",c:"#fb923c"}].map(r => (
                <div key={r.l} style={{ display: "flex", justifyContent: "space-between", gap: 16, padding: "6px 0", borderBottom: "1px solid rgba(255,255,255,.04)" }}>
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,.3)" }}>{r.l}</span>
                  <span style={{ fontSize: 11, fontWeight: 600, color: r.c }}>{r.v}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Small — EEG */}
          <div className="bento" style={{ padding: "28px 26px" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(167,139,250,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <BarChart2 size={21} color="#a78bfa" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>EEG Neurofeedback</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>Muse 2 / Muse S brainwave monitoring, real-time focus training, and meditation depth scoring.</p>
          </div>

          {/* Small — Dreams */}
          <div className="bento" style={{ padding: "28px 26px" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(167,139,250,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <Moon size={21} color="#a78bfa" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Dream Decoding</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>Jot down fragments — even jumbled words. AI reads the emotional narrative beneath the symbols.</p>
          </div>

          {/* Large — Sleep */}
          <div className="bento" style={{ gridColumn: "span 2", padding: "32px 30px", display: "flex", gap: 28, alignItems: "center", background: "linear-gradient(135deg,rgba(34,211,238,.04),rgba(167,139,250,.04))" }}>
            <div style={{ flex: 1 }}>
              <div style={{ width: 52, height: 52, borderRadius: 16, background: "rgba(34,211,238,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20 }}>
                <Activity size={24} color="#22D3EE" />
              </div>
              <h3 className="font-display" style={{ fontSize: 20, fontWeight: 800, marginBottom: 10, letterSpacing: "-.02em" }}>Sleep Staging</h3>
              <p style={{ fontSize: 14.5, color: "rgba(226,232,240,.5)", lineHeight: 1.65 }}>92.98% accurate sleep-stage detection — REM, deep, and light sleep tracked nightly. Wake up knowing exactly what your body did while you were gone.</p>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, minWidth: 140 }}>
              {[{s:"REM",v:"1h 42m",p:62,c:"#a78bfa"},{s:"Deep",v:"1h 18m",p:48,c:"#34D399"},{s:"Light",v:"3h 10m",p:82,c:"#22D3EE"}].map(r => (
                <div key={r.s}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 10, color: "rgba(255,255,255,.35)" }}>{r.s}</span>
                    <span style={{ fontSize: 10, fontWeight: 600, color: r.c }}>{r.v}</span>
                  </div>
                  <div style={{ height: 4, borderRadius: 99, background: "rgba(255,255,255,.06)" }}>
                    <div style={{ height: "100%", width: `${r.p}%`, borderRadius: 99, background: r.c }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Small — Nutrition */}
          <div className="bento" style={{ padding: "28px 26px" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(251,146,60,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <Apple size={21} color="#fb923c" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Nutrition & Food–Mood</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>6 eating states detected. See which foods lifted your mood, sharpened focus, or disrupted sleep.</p>
          </div>

          {/* Small — Breathwork */}
          <div className="bento" style={{ padding: "28px 26px" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(34,211,238,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <Wind size={21} color="#22D3EE" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Guided Breathwork</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>7 evidence-based exercises — box breathing, 4-7-8, coherence. HRV-guided and stress-aware.</p>
          </div>

          {/* Small — Peak Focus */}
          <div className="bento" style={{ padding: "28px 26px" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(52,211,153,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <Zap size={21} color="#34D399" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Peak Focus Windows</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>Predicts your optimal performance window daily — based on sleep, circadian rhythm, and mood baseline.</p>
          </div>

          {/* Small — Emotions */}
          <div className="bento" style={{ padding: "28px 26px" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(167,139,250,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <TrendingUp size={21} color="#a78bfa" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Emotion Tracking</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>Map your emotional landscape over weeks. Correlate mood with sleep, food, voice, and cycle phase.</p>
          </div>

        </div>
      </section>

      <hr className="divider" />

      {/* ── What the AI tells you ── */}
      <section style={{ padding: "80px 32px", maxWidth: 1140, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 52 }}>
          <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><Sparkles size={10} />Real AI insights</div>
          <h2 className="font-display" style={{ fontSize: "clamp(26px,4vw,46px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.1, marginBottom: 14 }}>
            What GetZen actually <span className="g">tells you</span>
          </h2>
          <p style={{ fontSize: 16, color: "rgba(226,232,240,.4)", maxWidth: 460, margin: "0 auto" }}>Real examples of insights generated by our AI models — not marketing copy.</p>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(300px,1fr))", gap: 16 }}>
          <AIQuote screen="Brain Report" color="#34D399" quote="Your voice stress markers dropped 23% this week. Based on your sleep pattern and yesterday's nutrition, your peak focus window is 9–11am today." />
          <AIQuote screen="Dream Journal" color="#a78bfa" quote="Three of your last five dreams involve water and movement. This pattern correlates with periods of change-seeking in your waking emotional data." />
          <AIQuote screen="Nutrition" color="#fb923c" quote="You logged salmon twice this week. On both following days, your focus score was 18–22% above your personal average. Omega-3 correlation: strong." />
          <AIQuote screen="Sleep Staging" color="#22D3EE" quote="You entered deep sleep 14 minutes faster than your baseline tonight. The 4-7-8 breathing session 30 minutes before bed likely contributed." />
          <AIQuote screen="Emotion Tracking" color="#34D399" quote="Your mood has been 31% more stable this month compared to last. The strongest correlating factor: consistent sleep schedule, 5+ days per week." />
          <AIQuote screen="Peak Focus" color="#a78bfa" quote="Tomorrow's readiness score projection: 78. Your circadian phase and today's recovery suggest afternoon energy will outperform your morning." />
        </div>
      </section>

      <hr className="divider" />

      {/* ── Science / How it works ── */}
      <section id="science" style={{ padding: "80px 32px", background: "rgba(52,211,153,.015)" }}>
        <div style={{ maxWidth: 1140, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 60 }}>
            <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><TrendingUp size={10} />The morning ritual</div>
            <h2 className="font-display" style={{ fontSize: "clamp(26px,4vw,48px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.1, marginBottom: 14 }}>
              Five minutes.<br /><span className="g">Clarity all day.</span>
            </h2>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(280px,1fr))", gap: 2 }}>
            {[
              { n:"01", icon:<Mic size={22} color="#34D399" />, title:"Speak for 30 seconds", body:"Voice analysis captures stress hormones, emotional state, and energy from acoustic features in your natural speech." },
              { n:"02", icon:<Moon size={22} color="#a78bfa" />, title:"Log your night", body:"Dream fragments. Sleep notes. How rested you feel. Under two minutes — and it compounds into something extraordinary." },
              { n:"03", icon:<Apple size={22} color="#fb923c" />, title:"Note what you ate", body:"A quick meal log triggers food–emotion correlation. GetZen connects what you eat to how you feel, automatically." },
              { n:"04", icon:<Brain size={22} color="#22D3EE" />, title:"Read your report", body:"Readiness score, peak focus window, mood prediction, and the one most impactful thing you can do today." },
            ].map((s, i) => (
              <div key={s.n} className="bento" style={{ padding: "32px 28px", borderRadius: i === 0 ? "20px 4px 4px 20px" : i === 3 ? "4px 20px 20px 4px" : "4px", position: "relative" }}>
                <p className="font-display" style={{ position: "absolute", top: 18, right: 22, fontSize: 52, fontWeight: 900, color: "rgba(255,255,255,.025)", lineHeight: 1 }}>{s.n}</p>
                <div style={{ marginBottom: 18 }}>{s.icon}</div>
                <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>{s.title}</h3>
                <p style={{ fontSize: 13.5, color: "rgba(226,232,240,.45)", lineHeight: 1.65 }}>{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <hr className="divider" />

      {/* ── Built for real life ── */}
      <section style={{ padding: "80px 32px" }}>
        <div style={{ maxWidth: 1140, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 48 }}>
            <h2 className="font-display" style={{ fontSize: "clamp(24px,3.5vw,44px)", fontWeight: 800, letterSpacing: "-.02em", marginBottom: 12 }}>Built for <span className="g">real life</span></h2>
            <p style={{ fontSize: 15, color: "rgba(226,232,240,.4)", maxWidth: 400, margin: "0 auto" }}>Your phone, your voice, your morning. No wearable. No lab.</p>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr 1fr", gap: 14, marginBottom: 14 }}>
            <LifeCard id="photo-1573496359142-b8d87734a5a2" alt="Talking and voice analysis" h={340} icon={<Mic size={14} color="#34D399" />} label="Voice" caption="Talk for 30 seconds. We read your emotion." accent="#34D399" />
            <LifeCard id="photo-1476480862126-209bfaa8edc8" alt="Morning run" h={340} icon={<Activity size={14} color="#22D3EE" />} label="Move" caption="Log your workout. See how it shifts your mood." accent="#22D3EE" />
            <LifeCard id="photo-1547592180-85f173990554" alt="Mindful eating" h={340} icon={<Apple size={14} color="#fb923c" />} label="Eat" caption="Every meal tells a story about how you feel." accent="#fb923c" />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 14 }}>
            <LifeCard id="photo-1544367567-0f2fcb009e0b" alt="Breathing exercise" h={300} icon={<Wind size={14} color="#a78bfa" />} label="Breathe" caption="Stressed? 7 guided sessions calm you in minutes." accent="#a78bfa" />
            <LifeCard id="photo-1508672019048-805c876b67e2" alt="Being mindful" h={300} icon={<Sparkles size={14} color="#34D399" />} label="Be Mindful" caption="GetZen keeps you aware of your inner world — every day." accent="#34D399" />
          </div>
        </div>
      </section>

      <hr className="divider" />

      {/* ── Compare ── */}
      <section id="compare" style={{ padding: "80px 32px", background: "rgba(255,255,255,.01)" }}>
        <div style={{ maxWidth: 720, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 52 }}>
            <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><Eye size={10} />How we compare</div>
            <h2 className="font-display" style={{ fontSize: "clamp(26px,4vw,44px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.1, marginBottom: 12 }}>
              Nothing else does <span className="g">all of this</span>
            </h2>
          </div>
          <div className="glass-bright" style={{ padding: "28px 32px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 16, marginBottom: 10, paddingBottom: 10, borderBottom: "1px solid rgba(255,255,255,.06)" }}>
              <span style={{ fontSize: 11, color: "rgba(255,255,255,.2)", letterSpacing: ".1em", textTransform: "uppercase" }}>Capability</span>
              <div style={{ width: 80, textAlign: "center" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src="/logo.png" alt="GetZen" style={{ width: 28, height: 28, objectFit: "cover", borderRadius: "50%", border: "1.5px solid rgba(52,211,153,.3)" }} />
              </div>
              <div style={{ width: 80, textAlign: "center" }}>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,.2)", fontWeight: 600 }}>Others*</span>
              </div>
            </div>
            {[
              ["Voice emotion analysis",true,false],["Dream decoding with AI",true,false],["Food–mood correlation",true,false],
              ["Sleep staging (92.98% accuracy)",true,false],["Daily unified brain report",true,false],
              ["Peak focus forecasting",true,false],["EEG neurofeedback (Muse)",true,false],
              ["Cycle-aware insights",true,false],["Breathwork with HRV guidance",true,true],
              ["Mood & emotion tracking",true,true],["Sleep duration tracking",true,true],
            ].map(([l,u,t]) => <CRow key={l as string} label={l as string} us={u as boolean} them={t as boolean} />)}
          </div>
          <p style={{ fontSize: 11.5, color: "rgba(255,255,255,.18)", textAlign: "center", marginTop: 16 }}>*Calm, Headspace, Oura, and Muse — none offers all of the above in one app</p>
        </div>
      </section>

      <hr className="divider" />

      {/* ── Waitlist ── */}
      <section id="waitlist" style={{ padding: "110px 32px 90px", position: "relative", overflow: "hidden" }}>
        <div className="a-mesh1" style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", width: 800, height: 800, borderRadius: "50%", background: "radial-gradient(circle,rgba(52,211,153,.06) 0%,transparent 65%)", pointerEvents: "none" }} />
        <div style={{ maxWidth: 600, margin: "0 auto", textAlign: "center", position: "relative" }}>
          <div className="tag" style={{ marginBottom: 24, display: "inline-flex" }}><Sparkles size={10} />Early access</div>
          <h2 className="font-display" style={{ fontSize: "clamp(30px,5vw,60px)", fontWeight: 900, letterSpacing: "-.03em", lineHeight: 1.0, marginBottom: 18 }}>
            Be first to <span className="shimmer">know<br />your mind</span>
          </h2>
          <p style={{ fontSize: 16, color: "rgba(226,232,240,.45)", lineHeight: 1.65, maxWidth: 460, margin: "0 auto 40px" }}>
            GetZen launches on iOS and Android. Join the waitlist and get early access — plus a free 3-month premium subscription.
          </p>
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 32 }}>
            <WaitlistForm />
          </div>
          {/* Platform badges */}
          <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
            <a href="#waitlist" className="platform-badge">
              <Smartphone size={18} color="rgba(255,255,255,.5)" />
              <div style={{ textAlign: "left" }}>
                <p style={{ fontSize: 9, color: "rgba(255,255,255,.35)", letterSpacing: ".08em" }}>COMING TO</p>
                <p style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,.8)" }}>App Store</p>
              </div>
            </a>
            <a href="#waitlist" className="platform-badge">
              <Play size={18} color="rgba(255,255,255,.5)" />
              <div style={{ textAlign: "left" }}>
                <p style={{ fontSize: 9, color: "rgba(255,255,255,.35)", letterSpacing: ".08em" }}>COMING TO</p>
                <p style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,.8)" }}>Google Play</p>
              </div>
            </a>
          </div>
          <p style={{ fontSize: 11.5, color: "rgba(255,255,255,.15)", marginTop: 24 }}>No spam. Unsubscribe anytime. Your data stays yours.</p>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer style={{ padding: "40px 32px", borderTop: "1px solid rgba(255,255,255,.05)", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 16 }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/logo.png" alt="GetZen" style={{ width: 52, height: 52, objectFit: "cover", borderRadius: "50%", border: "2px solid rgba(52,211,153,.25)" }} />
        <p style={{ fontSize: 12, color: "rgba(255,255,255,.15)" }}>© 2025 Know your mind.</p>
        <div style={{ display: "flex", gap: 24 }}>
          {["Privacy","Terms","Contact"].map(l => (
            <a key={l} href="#" style={{ fontSize: 12, color: "rgba(255,255,255,.22)", textDecoration: "none", transition: "color .2s" }}
              onMouseEnter={e => (e.currentTarget.style.color = "#e2e8f0")}
              onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,.22)")}
            >{l}</a>
          ))}
        </div>
      </footer>

    </div>
  );
}
