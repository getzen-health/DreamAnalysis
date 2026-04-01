"use client";
import { useState, useEffect, useRef } from "react";
import {
  ArrowRight, Mic, Brain, Moon, Wind, Activity, Zap,
  TrendingUp, Sparkles, ChevronRight, Apple,
  Eye, BarChart2, Check, X, Smartphone, Play, ChevronLeft
} from "lucide-react";

/* ─── Scroll progress bar ─── */
function ScrollProgress() {
  const [pct, setPct] = useState(0);
  useEffect(() => {
    const update = () => {
      const d = document.documentElement;
      setPct((d.scrollTop / (d.scrollHeight - d.clientHeight)) * 100);
    };
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);
  return (
    <div style={{ position: "fixed", top: 0, left: 0, height: 2, width: `${pct}%`, background: "linear-gradient(90deg,#34D399,#22D3EE,#a78bfa)", zIndex: 9999, transition: "width .08s linear", pointerEvents: "none" }} />
  );
}

/* ─── Word rotator for hero ─── */
function WordRotator({ words }: { words: string[] }) {
  const [idx, setIdx] = useState(0);
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const t = setInterval(() => {
      setVisible(false);
      setTimeout(() => { setIdx(i => (i + 1) % words.length); setVisible(true); }, 300);
    }, 2400);
    return () => clearInterval(t);
  }, [words.length]);
  return (
    <span style={{ display: "inline-block", opacity: visible ? 1 : 0, transform: visible ? "translateY(0)" : "translateY(10px)", transition: "opacity .3s ease, transform .3s ease" }}>
      {words[idx]}
    </span>
  );
}

/* ─── Animated voice waveform ─── */
function AnimatedWaveform({ color = "#34D399", bars = 28 }: { color?: string; bars?: number }) {
  const heights = [4,8,14,22,32,38,32,26,18,12,20,30,38,44,38,28,18,10,18,28,36,28,18,10,8,14,22,12];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 3, height: 52 }}>
      {heights.slice(0, bars).map((h, i) => (
        <div key={i} className="a-wave" style={{
          width: 3, height: h, borderRadius: 99, background: color, flexShrink: 0,
          "--dur": `${0.7 + (i % 6) * 0.12}s`,
          "--del": `${(i * 0.045).toFixed(2)}s`,
        } as React.CSSProperties} />
      ))}
    </div>
  );
}

/* ─── Animated SVG sleep chart ─── */
function SleepChart() {
  const [drawn, setDrawn] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setDrawn(true); obs.disconnect(); }
    }, { threshold: 0.3 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  const pts = "0,54 22,50 44,28 66,8 88,22 110,40 132,12 154,4 176,18 198,42 220,32 242,8 264,24 280,48";
  return (
    <div ref={ref} style={{ marginTop: 16 }}>
      <svg width="100%" height="60" viewBox="0 0 280 60" preserveAspectRatio="none" style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id="sg" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#34D399" />
            <stop offset="55%" stopColor="#22D3EE" />
            <stop offset="100%" stopColor="#a78bfa" />
          </linearGradient>
          <linearGradient id="sfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#34D399" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#34D399" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polyline points={pts + " 280,60 0,60"} fill="url(#sfill)" stroke="none" />
        <polyline points={pts} fill="none" stroke="url(#sg)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
          style={{ strokeDasharray: 800, strokeDashoffset: drawn ? 0 : 800, transition: drawn ? "stroke-dashoffset 2.2s cubic-bezier(.4,0,.2,1)" : "none" }}
        />
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
        {["10pm","12am","2am","4am","6am"].map(t => (
          <span key={t} style={{ fontSize: 8, color: "rgba(255,255,255,.2)", letterSpacing: ".04em" }}>{t}</span>
        ))}
      </div>
    </div>
  );
}

/* ─── Testimonial card ─── */
function TestimonialCard({ name, role, quote, metric, color, avatar }: { name: string; role: string; quote: string; metric: string; color: string; avatar: string }) {
  return (
    <div className="tcard" style={{ padding: "28px 26px", borderColor: `${color}14`, display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", gap: 4 }}>
        {[1,2,3,4,5].map(i => (
          <svg key={i} width="13" height="13" viewBox="0 0 14 14" fill="none">
            <path d="M7 1l1.545 3.13L12 4.635l-2.5 2.435.59 3.44L7 8.885l-3.09 1.625.59-3.44L2 4.635l3.455-.505L7 1z" fill={color} />
          </svg>
        ))}
      </div>
      <p style={{ fontSize: 14.5, color: "rgba(226,232,240,.75)", lineHeight: 1.65, fontStyle: "italic", flex: 1 }}>&ldquo;{quote}&rdquo;</p>
      <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "5px 12px", borderRadius: 999, background: `${color}12`, border: `1px solid ${color}28` }}>
        <TrendingUp size={11} color={color} />
        <span style={{ fontSize: 11, fontWeight: 700, color, letterSpacing: ".04em" }}>{metric}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={`https://images.unsplash.com/${avatar}?w=80&h=80&q=80&fit=crop&auto=format`} alt={name}
          style={{ width: 42, height: 42, borderRadius: "50%", objectFit: "cover", border: `2px solid ${color}30` }} />
        <div>
          <p style={{ fontSize: 13.5, fontWeight: 700, color: "#e2e8f0" }}>{name}</p>
          <p style={{ fontSize: 11, color: "rgba(255,255,255,.3)", marginTop: 2 }}>{role}</p>
        </div>
      </div>
    </div>
  );
}

/* ─── Scroll reveal wrapper ─── */
function Reveal({ children, delay = 0, y = 28 }: { children: React.ReactNode; delay?: number; y?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setVisible(true); obs.disconnect(); }
    }, { threshold: 0.1 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return (
    <div ref={ref} style={{ opacity: visible ? 1 : 0, transform: visible ? "none" : `translateY(${y}px)`, transition: `opacity .65s ease ${delay}ms, transform .65s ease ${delay}ms` }}>
      {children}
    </div>
  );
}

/* ─── Cursor flashlight hook ─── */
function useCursorGlow(ref: React.RefObject<HTMLElement | null>) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const move = (e: MouseEvent) => {
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${e.clientX - r.left}px`);
      el.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    el.addEventListener("mousemove", move);
    return () => el.removeEventListener("mousemove", move);
  }, [ref]);
}

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
      <div className="scanline-wrap" style={{ borderRadius: 36, overflow: "hidden", background: "#070811", minHeight: 490 }}>
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

/* ─── Insight image card ─── */
function InsightCard({ id, h, color, screen, quote }: { id: string; h: number; color: string; screen: string; quote: string }) {
  return (
    <div style={{ position: "relative", height: h, borderRadius: 20, overflow: "hidden" }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`https://images.unsplash.com/${id}?w=900&q=80&fit=crop&auto=format`} alt={screen}
        style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", transition: "transform .5s ease" }}
        onMouseEnter={e => (e.currentTarget.style.transform = "scale(1.04)")}
        onMouseLeave={e => (e.currentTarget.style.transform = "scale(1)")}
      />
      <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to top, rgba(7,8,15,.95) 0%, rgba(7,8,15,.4) 50%, rgba(7,8,15,.15) 100%)" }} />
      <div style={{ position: "absolute", bottom: 20, left: 20, right: 20 }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 10px", borderRadius: 999, background: `${color}20`, border: `1px solid ${color}40`, marginBottom: 10 }}>
          <Sparkles size={9} color={color} />
          <span style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: ".1em", textTransform: "uppercase", color }}>{screen}</span>
        </div>
        <p style={{ fontSize: 14, color: "rgba(255,255,255,.85)", lineHeight: 1.5, fontStyle: "italic" }}>&ldquo;{quote}&rdquo;</p>
      </div>
    </div>
  );
}

/* ─── Step image card ─── */
function StepCard({ id, h, n, color, title, caption }: { id: string; h: number; n: string; color: string; title: string; caption: string }) {
  return (
    <div style={{ position: "relative", height: h, borderRadius: 20, overflow: "hidden" }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`https://images.unsplash.com/${id}?w=700&q=80&fit=crop&auto=format`} alt={title}
        style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", transition: "transform .5s ease" }}
        onMouseEnter={e => (e.currentTarget.style.transform = "scale(1.04)")}
        onMouseLeave={e => (e.currentTarget.style.transform = "scale(1)")}
      />
      <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to top, rgba(7,8,15,.95) 0%, rgba(7,8,15,.2) 60%, transparent 100%)" }} />
      {/* Step number top left */}
      <div style={{ position: "absolute", top: 16, left: 16 }}>
        <span className="font-display" style={{ fontSize: 13, fontWeight: 900, color, letterSpacing: ".05em", background: `${color}18`, border: `1px solid ${color}35`, padding: "4px 10px", borderRadius: 999 }}>{n}</span>
      </div>
      <div style={{ position: "absolute", bottom: 20, left: 18, right: 18 }}>
        <h3 style={{ fontSize: 16, fontWeight: 800, marginBottom: 7, color: "#fff" }}>{title}</h3>
        <p style={{ fontSize: 12.5, color: "rgba(255,255,255,.65)", lineHeight: 1.5 }}>{caption}</p>
      </div>
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

/* ─── Bento grid with cursor flashlight ─── */
function BentoGrid({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useCursorGlow(ref as React.RefObject<HTMLElement>);
  return (
    <div ref={ref} className="cursor-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
      {children}
    </div>
  );
}

/* ─── Star rating ─── */
function Stars({ rating = 4.8, count = "2.1k" }: { rating?: number; count?: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ display: "flex", gap: 2 }}>
        {[1,2,3,4,5].map(i => (
          <svg key={i} width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M7 1l1.545 3.13L12 4.635l-2.5 2.435.59 3.44L7 8.885l-3.09 1.625.59-3.44L2 4.635l3.455-.505L7 1z"
              fill={i <= Math.floor(rating) ? "#34D399" : "rgba(52,211,153,.25)"} />
          </svg>
        ))}
      </div>
      <span style={{ fontSize: 13, fontWeight: 600, color: "#34D399" }}>{rating}</span>
      <span style={{ fontSize: 12, color: "rgba(255,255,255,.25)" }}>· {count} beta users</span>
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

/* ─── Drag-scroll carousel ─── */
function DragScroll({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const isDown = useRef(false);
  const startX = useRef(0);
  const scrollLeft = useRef(0);
  const onDown = (e: React.MouseEvent) => {
    isDown.current = true;
    startX.current = e.pageX - (ref.current?.offsetLeft ?? 0);
    scrollLeft.current = ref.current?.scrollLeft ?? 0;
    if (ref.current) ref.current.style.cursor = "grabbing";
  };
  const onUp = () => { isDown.current = false; if (ref.current) ref.current.style.cursor = "grab"; };
  const onMove = (e: React.MouseEvent) => {
    if (!isDown.current) return;
    e.preventDefault();
    const x = e.pageX - (ref.current?.offsetLeft ?? 0);
    const walk = (x - startX.current) * 1.4;
    if (ref.current) ref.current.scrollLeft = scrollLeft.current - walk;
  };
  const scroll = (dir: number) => { if (ref.current) ref.current.scrollBy({ left: dir * 380, behavior: "smooth" }); };
  return (
    <div style={{ position: "relative" }}>
      <button onClick={() => scroll(-1)} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", zIndex: 10, width: 42, height: 42, borderRadius: "50%", background: "rgba(7,8,15,.9)", border: "1px solid rgba(255,255,255,.12)", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}>
        <ChevronLeft size={18} color="rgba(255,255,255,.7)" />
      </button>
      <button onClick={() => scroll(1)} style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", zIndex: 10, width: 42, height: 42, borderRadius: "50%", background: "rgba(7,8,15,.9)", border: "1px solid rgba(255,255,255,.12)", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}>
        <ChevronRight size={18} color="rgba(255,255,255,.7)" />
      </button>
      <div ref={ref} onMouseDown={onDown} onMouseLeave={onUp} onMouseUp={onUp} onMouseMove={onMove}
        style={{ overflowX: "auto", cursor: "grab", scrollbarWidth: "none", msOverflowStyle: "none" } as React.CSSProperties}>
        {children}
      </div>
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
      <ScrollProgress />

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
        {/* Ambient particles */}
        {[
          { top:"12%",left:"18%", s:5, c:"#34D399", dur:14, dx:"20px", dy:"-24px" },
          { top:"25%",left:"82%", s:3, c:"#22D3EE", dur:18, dx:"-16px", dy:"20px" },
          { top:"65%",left:"12%", s:4, c:"#a78bfa", dur:12, dx:"24px", dy:"-18px" },
          { top:"72%",left:"88%", s:6, c:"#34D399", dur:20, dx:"-22px", dy:"-28px" },
          { top:"45%",left:"6%",  s:3, c:"#22D3EE", dur:16, dx:"18px", dy:"14px" },
          { top:"30%",left:"92%", s:4, c:"#a78bfa", dur:22, dx:"-14px", dy:"22px" },
          { top:"80%",left:"50%", s:3, c:"#34D399", dur:13, dx:"12px", dy:"-16px" },
          { top:"15%",left:"55%", s:5, c:"#22D3EE", dur:17, dx:"-18px", dy:"20px" },
        ].map((p, i) => (
          <div key={i} className="a-particle" style={{
            position: "absolute", top: p.top, left: p.left,
            width: p.s, height: p.s, borderRadius: "50%", background: p.c,
            "--pdur": `${p.dur}s`, "--dx": p.dx, "--dy": p.dy,
            boxShadow: `0 0 ${p.s * 3}px ${p.c}`,
            pointerEvents: "none",
          } as React.CSSProperties} />
        ))}

        {/* Logo with glow ring */}
        <div className="hero-seq-1 a-float" style={{ position: "relative", marginBottom: 28 }}>
          <div className="a-spin" style={{ position: "absolute", inset: -8, borderRadius: "50%", background: "conic-gradient(from 0deg,#34D399,#22D3EE,#a78bfa,transparent,#34D399)", opacity: .35, pointerEvents: "none" }} />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="GetZen Health" style={{ width: 280, height: 280, objectFit: "cover", borderRadius: "50%", position: "relative", zIndex: 1 }} />
        </div>

        <div className="hero-seq-2 tag" style={{ marginBottom: 24 }}>
          <Sparkles size={10} />
          Unified mind–body intelligence
        </div>

        <h1 className="hero-seq-3 font-display" style={{ fontSize: "clamp(40px,6vw,78px)", fontWeight: 900, lineHeight: 1.0, letterSpacing: "-.03em", marginBottom: 22, maxWidth: 820 }}>
          Know Your <span className="shimmer">Mind</span><br />Every Morning
        </h1>

        <p className="hero-seq-4" style={{ fontSize: 18, color: "rgba(226,232,240,.5)", lineHeight: 1.7, maxWidth: 480, marginBottom: 16 }}>
          Your <span style={{ color: "rgba(226,232,240,.85)", fontWeight: 500 }}><WordRotator words={["voice","dreams","nutrition","sleep","emotions","brainwaves"]} /></span> — decoded by AI into one daily picture of your inner world.
        </p>
        <p className="hero-seq-4" style={{ fontSize: 14, color: "rgba(226,232,240,.3)", maxWidth: 460, marginBottom: 36, lineHeight: 1.6 }}>
          16 ML models · No wearable required · 30 seconds a day
        </p>

        <div className="hero-seq-5" style={{ display: "flex", gap: 14, justifyContent: "center", marginBottom: 20 }}>
          <a href="#waitlist" className="btn btn-glow">Join the waitlist <ArrowRight size={16} /></a>
          <a href="#features" className="btn-outline">See how it works</a>
        </div>

        {/* Stars + counter below buttons */}
        <div className="hero-seq-5" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, marginBottom: 48 }}>
          <Stars />
          <span style={{ fontSize: 12, color: "rgba(255,255,255,.2)" }}>Join <strong style={{ color: "rgba(255,255,255,.45)" }}>2,847</strong> people already on the waitlist</span>
        </div>

        {/* Animated stats */}
        <div style={{ display: "flex", gap: 0, marginBottom: 72, background: "rgba(255,255,255,.03)", border: "1px solid rgba(255,255,255,.07)", borderRadius: 18, overflow: "hidden" }}>
          {[{v:16,s:"",l:"AI Models"},{v:92,s:".98%",l:"Sleep Accuracy"},{v:7,s:"",l:"Biometric Layers"}].map((s, i) => (
            <div key={s.l} style={{ padding: "20px 40px", borderRight: i < 2 ? "1px solid rgba(255,255,255,.06)" : "none", textAlign: "center" }}>
              <p className="font-display a-numglow" style={{ fontSize: 30, fontWeight: 800, background: "linear-gradient(135deg,#34D399,#22D3EE)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", lineHeight: 1 }}>
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
      <section id="features" data-science="true" style={{ padding: "100px 32px", maxWidth: 1140, margin: "0 auto" }}><span id="science" style={{ position: "absolute", marginTop: -80 }} />
        <Reveal>
          <div style={{ textAlign: "center", marginBottom: 64 }}>
            <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><Activity size={10} />What GetZen tracks</div>
            <h2 className="font-display" style={{ fontSize: "clamp(28px,4vw,50px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.05, marginBottom: 16 }}>
              Seven layers of <span className="gc">self-knowledge</span>
            </h2>
            <p style={{ fontSize: 16, color: "rgba(226,232,240,.4)", maxWidth: 480, margin: "0 auto" }}>
              Most apps track one signal. GetZen layers seven — then connects the dots.
            </p>
          </div>
        </Reveal>

        {/* Bento grid with cursor flashlight */}
        <BentoGrid>

          {/* Large — Voice */}
          <div className="bento" style={{ gridColumn: "span 2", padding: "36px 32px", display: "flex", gap: 32, alignItems: "center", background: "linear-gradient(135deg,rgba(52,211,153,.06),rgba(34,211,238,.04))" }}>
            <div style={{ flex: 1 }}>
              <div style={{ width: 52, height: 52, borderRadius: 16, background: "rgba(52,211,153,.12)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20 }}>
                <Mic size={24} color="#34D399" />
              </div>
              <h3 className="font-display" style={{ fontSize: 20, fontWeight: 800, marginBottom: 10, letterSpacing: "-.02em" }}>Voice Emotion Analysis</h3>
              <p style={{ fontSize: 14.5, color: "rgba(226,232,240,.5)", lineHeight: 1.65 }}>Speak 30 seconds each morning. GetZen extracts stress markers, emotional tone, and energy — no wearable required. Your voice knows things your mind hasn&apos;t admitted yet.</p>
            </div>
            <div style={{ padding: "18px 20px", background: "rgba(255,255,255,.03)", borderRadius: 16, border: "1px solid rgba(52,211,153,.12)", minWidth: 180 }}>
              <p style={{ fontSize: 9, color: "rgba(255,255,255,.3)", letterSpacing: ".1em", textTransform: "uppercase", marginBottom: 12 }}>Live voice pattern</p>
              <AnimatedWaveform color="#34D399" bars={28} />
              <div style={{ marginTop: 12, display: "flex", justifyContent: "space-between" }}>
                {[{l:"Tone",v:"Confident",c:"#34D399"},{l:"Stress",v:"Low",c:"#22D3EE"},{l:"Energy",v:"High",c:"#a78bfa"}].map(r => (
                  <div key={r.l} style={{ textAlign: "center" }}>
                    <p style={{ fontSize: 10, fontWeight: 700, color: r.c }}>{r.v}</p>
                    <p style={{ fontSize: 8, color: "rgba(255,255,255,.25)", marginTop: 2 }}>{r.l}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Small — EEG */}
          <Reveal delay={80} y={20}>
          <div className="bento" style={{ padding: "28px 26px", height: "100%" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(167,139,250,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <BarChart2 size={21} color="#a78bfa" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>EEG Neurofeedback</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>Muse 2 / Muse S brainwave monitoring, real-time focus training, and meditation depth scoring.</p>
          </div>
          </Reveal>

          {/* Small — Dreams */}
          <Reveal delay={160} y={20}>
          <div className="bento" style={{ padding: "28px 26px", height: "100%" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(167,139,250,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <Moon size={21} color="#a78bfa" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Dream Decoding</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>Jot down fragments — even jumbled words. AI reads the emotional narrative beneath the symbols.</p>
          </div>
          </Reveal>

          {/* Large — Sleep */}
          <div className="bento" style={{ gridColumn: "span 2", padding: "32px 30px", display: "flex", gap: 28, alignItems: "center", background: "linear-gradient(135deg,rgba(34,211,238,.04),rgba(167,139,250,.04))" }}>
            <div style={{ flex: 1 }}>
              <div style={{ width: 52, height: 52, borderRadius: 16, background: "rgba(34,211,238,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20 }}>
                <Activity size={24} color="#22D3EE" />
              </div>
              <h3 className="font-display" style={{ fontSize: 20, fontWeight: 800, marginBottom: 10, letterSpacing: "-.02em" }}>Sleep Staging</h3>
              <p style={{ fontSize: 14.5, color: "rgba(226,232,240,.5)", lineHeight: 1.65 }}>92.98% accurate sleep-stage detection — REM, deep, and light sleep tracked nightly. Wake up knowing exactly what your body did while you were gone.</p>
            </div>
            <div style={{ minWidth: 200, flex: "0 0 200px" }}>
              <div style={{ display: "flex", gap: 14, marginBottom: 10 }}>
                {[{s:"REM",v:"1h 42m",c:"#a78bfa"},{s:"Deep",v:"1h 18m",c:"#34D399"},{s:"Light",v:"3h 10m",c:"#22D3EE"}].map(r => (
                  <div key={r.s} style={{ textAlign: "center" }}>
                    <p style={{ fontSize: 11, fontWeight: 700, color: r.c }}>{r.v}</p>
                    <p style={{ fontSize: 8, color: "rgba(255,255,255,.28)", marginTop: 2 }}>{r.s}</p>
                  </div>
                ))}
              </div>
              <SleepChart />
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
          <Reveal delay={80} y={20}>
          <div className="bento" style={{ padding: "28px 26px", height: "100%" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(34,211,238,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <Wind size={21} color="#22D3EE" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Guided Breathwork</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>7 evidence-based exercises — box breathing, 4-7-8, coherence. HRV-guided and stress-aware.</p>
          </div>
          </Reveal>

          {/* Small — Peak Focus */}
          <Reveal delay={160} y={20}>
          <div className="bento" style={{ padding: "28px 26px", height: "100%" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(52,211,153,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <Zap size={21} color="#34D399" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Peak Focus Windows</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>Predicts your optimal performance window daily — based on sleep, circadian rhythm, and mood baseline.</p>
          </div>
          </Reveal>

          {/* Small — Emotions */}
          <Reveal delay={240} y={20}>
          <div className="bento" style={{ padding: "28px 26px", height: "100%" }}>
            <div style={{ width: 46, height: 46, borderRadius: 14, background: "rgba(167,139,250,.1)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
              <TrendingUp size={21} color="#a78bfa" />
            </div>
            <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Emotion Tracking</h3>
            <p style={{ fontSize: 13, color: "rgba(226,232,240,.45)", lineHeight: 1.6 }}>Map your emotional landscape over weeks. Correlate mood with sleep, food, voice, and cycle phase.</p>
          </div>
          </Reveal>

        </BentoGrid>
      </section>

      <hr className="divider" />

      {/* ── What the AI tells you — insight stream ── */}
      <section style={{ padding: "90px 32px", maxWidth: 1140, margin: "0 auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: 72, alignItems: "start" }}>
          {/* Left — sticky heading */}
          <Reveal>
            <div style={{ position: "sticky", top: 96 }}>
              <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><Sparkles size={10} />Real AI insights</div>
              <h2 className="font-display" style={{ fontSize: "clamp(28px,3.5vw,50px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.05, marginBottom: 18 }}>
                What GetZen<br />actually <span className="g">tells you</span>
              </h2>
              <p style={{ fontSize: 15, color: "rgba(226,232,240,.38)", lineHeight: 1.7, maxWidth: 320, marginBottom: 36 }}>
                Every morning, five AI models run in parallel. This is what they report back — not generic advice, but your specific patterns.
              </p>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#34D399", boxShadow: "0 0 10px #34D399", animation: "pulse 2s ease-in-out infinite" }} />
                <span style={{ fontSize: 10.5, color: "rgba(255,255,255,.28)", letterSpacing: ".1em", textTransform: "uppercase" }}>Live AI · Updated daily</span>
              </div>
            </div>
          </Reveal>
          {/* Right — insight cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {([
              { icon: <Mic size={17} color="#34D399" />, label: "Voice AI", time: "8:14 am today", color: "#34D399", quote: "Stress markers dropped 23% this week. Vocal clarity suggests high confidence — your peak focus window opens at 9am.", badge: "Focus ↑ 23%", sub: "Stress · tone · energy · clarity" },
              { icon: <Moon size={17} color="#a78bfa" />, label: "Dream AI", time: "decoded last night", color: "#a78bfa", quote: "Water and movement appeared in 3 of your last 5 dreams. This pattern correlates strongly with active change-seeking.", badge: "Pattern detected", sub: "Symbols · recurring themes · emotion" },
              { icon: <Apple size={17} color="#fb923c" />, label: "Nutrition AI", time: "this week", color: "#fb923c", quote: "Salmon twice this week — your focus score ran 21% above average on both days. Omega-3 link confirmed.", badge: "Food–mood link", sub: "6 eating states · meal impact" },
              { icon: <Activity size={17} color="#22D3EE" />, label: "Sleep AI", time: "this morning", color: "#22D3EE", quote: "Deep sleep arrived 14 minutes faster than your baseline. Your 4-7-8 breathwork session before bed contributed.", badge: "Deep sleep ↑", sub: "REM · deep · light · 92.98% accurate" },
              { icon: <TrendingUp size={17} color="#a78bfa" />, label: "Emotion AI", time: "this month", color: "#a78bfa", quote: "Mood stability is up 31% from last month. Cross-signal analysis points to improved sleep consistency as the key driver.", badge: "Stability ↑ 31%", sub: "Daily mood · longitudinal patterns" },
            ] as const).map((ins, i) => (
              <Reveal key={i} delay={i * 65} y={16}>
                <div className="bento" style={{ padding: "18px 22px", borderLeft: `3px solid ${ins.color}45`, borderTopLeftRadius: 4, borderBottomLeftRadius: 4, transition: "border-left-color .25s, transform .25s" }}
                  onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.borderLeftColor = ins.color; (e.currentTarget as HTMLDivElement).style.transform = "translateX(5px)"; }}
                  onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.borderLeftColor = `${ins.color}45`; (e.currentTarget as HTMLDivElement).style.transform = "translateX(0)"; }}
                >
                  <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 10 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ width: 36, height: 36, borderRadius: 10, background: `${ins.color}14`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>{ins.icon}</div>
                      <div>
                        <p style={{ fontSize: 12, fontWeight: 700, color: ins.color }}>{ins.label}</p>
                        <p style={{ fontSize: 9, color: "rgba(255,255,255,.22)", marginTop: 1, letterSpacing: ".04em" }}>{ins.sub}</p>
                      </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                      <span style={{ fontSize: 9.5, color: "rgba(255,255,255,.2)" }}>{ins.time}</span>
                      <span style={{ padding: "3px 10px", borderRadius: 999, background: `${ins.color}14`, border: `1px solid ${ins.color}28`, fontSize: 10.5, fontWeight: 700, color: ins.color, whiteSpace: "nowrap" }}>{ins.badge}</span>
                    </div>
                  </div>
                  <p style={{ fontSize: 13.5, color: "rgba(226,232,240,.68)", lineHeight: 1.62, fontStyle: "italic" }}>&ldquo;{ins.quote}&rdquo;</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <hr className="divider" />

      {/* ── Morning ritual — cinematic alternating rows ── */}
      <section style={{ padding: "0" }}>
        <div style={{ maxWidth: 1140, margin: "0 auto", padding: "80px 32px 40px" }}>
          <Reveal>
            <div style={{ textAlign: "center", marginBottom: 64 }}>
              <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><TrendingUp size={10} />The morning ritual</div>
              <h2 className="font-display" style={{ fontSize: "clamp(28px,4vw,52px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.05, marginBottom: 14 }}>
                Five minutes.<br /><span className="g">Clarity all day.</span>
              </h2>
            </div>
          </Reveal>
        </div>
        {([
          { n: "01", color: "#34D399", title: "Speak", sub: "30 seconds. Your voice knows.", desc: "Each morning, speak freely for half a minute. GetZen's voice AI extracts stress markers, emotional tone, and energy from acoustic patterns — no wearable, no typing, no effort.", img: "photo-1516321497487-e288fb19713f", tags: ["Stress level", "Emotional tone", "Energy index", "Voice clarity"] },
          { n: "02", color: "#a78bfa", title: "Dream Log", sub: "Fragments. The AI reads the story.", desc: "Jot down anything — even a single word. GetZen decodes the emotional narrative beneath your symbols, recurring themes, and subconscious signals collected overnight.", img: "photo-1517842645767-c639042777db", tags: ["Symbol analysis", "Emotion themes", "Pattern history", "REM correlation"] },
          { n: "03", color: "#fb923c", title: "Eat & Note", sub: "Log it. Watch your mood shift.", desc: "Track your meals and GetZen reveals the food–mood correlations no nutrition app has ever shown you. Six eating states. Meal-by-meal mood impact. Across weeks and months.", img: "photo-1484723091739-30990688934a", tags: ["6 eating states", "Focus correlation", "Mood history", "Nutrient links"] },
          { n: "04", color: "#22D3EE", title: "Read Your Report", sub: "Everything unified. Every morning.", desc: "Your daily Brain Report synthesizes all 16 ML models: readiness score, peak focus window, sleep quality, mood trajectory — one clear picture of your inner world.", img: "photo-1551288049-bebda4e38f71", tags: ["Readiness score", "Peak focus time", "Cross-signal AI", "7 biometric layers"] },
        ] as const).map((step, i) => (
          <Reveal key={step.n} y={40}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", minHeight: 480, marginBottom: 3, overflow: "hidden" }}>
              {/* Photo side */}
              <div style={{ order: i % 2 === 0 ? 1 : 2, position: "relative", overflow: "hidden" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`https://images.unsplash.com/${step.img}?w=900&q=85&fit=crop&auto=format`} alt={step.title}
                  style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", transition: "transform .7s ease" }}
                  onMouseEnter={e => (e.currentTarget.style.transform = "scale(1.05)")}
                  onMouseLeave={e => (e.currentTarget.style.transform = "scale(1)")}
                />
                <div style={{ position: "absolute", inset: 0, background: i % 2 === 0 ? "linear-gradient(to right,transparent 45%,rgba(7,8,15,.95) 100%)" : "linear-gradient(to left,transparent 45%,rgba(7,8,15,.95) 100%)" }} />
                <div style={{ position: "absolute", [i % 2 === 0 ? "bottom" : "bottom"]: -10, [i % 2 === 0 ? "right" : "left"]: -8, lineHeight: 1, userSelect: "none", pointerEvents: "none" }}>
                  <span className="font-display" style={{ fontSize: "clamp(100px,14vw,180px)", fontWeight: 900, color: `${step.color}18`, letterSpacing: "-.06em", display: "block" }}>{step.n}</span>
                </div>
              </div>
              {/* Text side */}
              <div style={{ order: i % 2 === 0 ? 2 : 1, padding: "64px 60px", display: "flex", flexDirection: "column", justifyContent: "center", background: i % 2 === 0 ? "rgba(10,14,28,.92)" : "rgba(13,17,34,.92)" }}>
                <span className="font-display" style={{ fontSize: 11, fontWeight: 900, color: step.color, letterSpacing: ".16em", textTransform: "uppercase", marginBottom: 16 }}>Step {step.n}</span>
                <h3 className="font-display" style={{ fontSize: "clamp(26px,3.5vw,48px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.05, marginBottom: 10 }}>{step.title}</h3>
                <p style={{ fontSize: 14, fontWeight: 600, color: step.color, marginBottom: 18, opacity: .85 }}>{step.sub}</p>
                <p style={{ fontSize: 15, color: "rgba(226,232,240,.45)", lineHeight: 1.8, maxWidth: 420, marginBottom: 28 }}>{step.desc}</p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {step.tags.map(t => (
                    <span key={t} style={{ padding: "5px 14px", borderRadius: 999, background: `${step.color}10`, border: `1px solid ${step.color}22`, fontSize: 11, fontWeight: 600, color: step.color, letterSpacing: ".04em" }}>{t}</span>
                  ))}
                </div>
              </div>
            </div>
          </Reveal>
        ))}
      </section>

      <hr className="divider" />

      {/* ── Built for real life — horizontal carousel ── */}
      <section style={{ padding: "80px 0" }}>
        <Reveal>
          <div style={{ textAlign: "center", marginBottom: 48, padding: "0 32px" }}>
            <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><Zap size={10} />Built for real life</div>
            <h2 className="font-display" style={{ fontSize: "clamp(26px,4vw,50px)", fontWeight: 800, letterSpacing: "-.025em", marginBottom: 12 }}>Your phone.<br />Your <span className="g">morning ritual.</span></h2>
            <p style={{ fontSize: 15, color: "rgba(226,232,240,.35)", maxWidth: 380, margin: "0 auto" }}>No wearable. No lab. No friction. Just open GetZen.</p>
          </div>
        </Reveal>
        <DragScroll>
          <div style={{ display: "flex", gap: 14, padding: "0 40px 24px", width: "max-content" }}>
            {([
              { id: "photo-1573496359142-b8d87734a5a2", icon: <Mic size={16} color="#34D399" />, label: "Voice Check-in", caption: "Speak for 30 seconds. We read stress, energy, and emotion from your voice alone.", accent: "#34D399", sub: "Every morning · 30 sec" },
              { id: "photo-1476480862126-209bfaa8edc8", icon: <Activity size={16} color="#22D3EE" />, label: "Movement", caption: "Log a workout and instantly see how physical activity shifts your mood and recovery.", accent: "#22D3EE", sub: "Post-workout mood shift" },
              { id: "photo-1547592180-85f173990554", icon: <Apple size={16} color="#fb923c" />, label: "Mindful Eating", caption: "Every meal is tracked for its mood impact. GetZen shows the food-emotion link in real time.", accent: "#fb923c", sub: "6 eating states detected" },
              { id: "photo-1544367567-0f2fcb009e0b", icon: <Wind size={16} color="#a78bfa" />, label: "Breathwork", caption: "Stressed? 7 guided breathing exercises — box, 4-7-8, coherence — calm you in minutes.", accent: "#a78bfa", sub: "HRV-guided · 7 exercises" },
              { id: "photo-1508672019048-805c876b67e2", icon: <Sparkles size={16} color="#34D399" />, label: "Daily Awareness", caption: "GetZen keeps a live picture of your inner world — mood, patterns, and long-term trends.", accent: "#34D399", sub: "Longitudinal · always on" },
            ] as const).map((card, i) => (
              <div key={i} style={{ position: "relative", width: 320, height: 480, borderRadius: 24, overflow: "hidden", flexShrink: 0 }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`https://images.unsplash.com/${card.id}?w=700&q=80&fit=crop&auto=format`} alt={card.label}
                  style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", transition: "transform .6s ease" }}
                  onMouseEnter={e => (e.currentTarget.style.transform = "scale(1.06)")}
                  onMouseLeave={e => (e.currentTarget.style.transform = "scale(1)")}
                />
                <div style={{ position: "absolute", inset: 0, background: `linear-gradient(to top, rgba(7,8,15,.96) 0%, rgba(7,8,15,.4) 55%, rgba(7,8,15,.1) 100%)` }} />
                {/* Index number */}
                <div style={{ position: "absolute", top: 20, right: 20 }}>
                  <span className="font-display" style={{ fontSize: 11, fontWeight: 900, color: card.accent, letterSpacing: ".12em" }}>0{i + 1}</span>
                </div>
                {/* Content */}
                <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "24px 22px" }}>
                  <div style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "5px 12px", borderRadius: 999, background: `${card.accent}18`, border: `1px solid ${card.accent}35`, marginBottom: 10 }}>
                    {card.icon}
                    <span style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase", color: card.accent }}>{card.label}</span>
                  </div>
                  <p style={{ fontSize: 13.5, color: "rgba(255,255,255,.82)", lineHeight: 1.55, marginBottom: 10 }}>{card.caption}</p>
                  <p style={{ fontSize: 10, color: "rgba(255,255,255,.28)", letterSpacing: ".06em" }}>{card.sub}</p>
                </div>
              </div>
            ))}
          </div>
        </DragScroll>
        <div style={{ textAlign: "center", marginTop: 10 }}>
          <span style={{ fontSize: 11, color: "rgba(255,255,255,.18)", letterSpacing: ".08em" }}>← drag to explore →</span>
        </div>
      </section>

      <hr className="divider" />

      {/* ── Compare — two-panel showdown ── */}
      <section id="compare" style={{ padding: "90px 32px", background: "rgba(255,255,255,.01)" }}>
        <Reveal>
          <div style={{ textAlign: "center", marginBottom: 56 }}>
            <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><Eye size={10} />How we compare</div>
            <h2 className="font-display" style={{ fontSize: "clamp(26px,4vw,48px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.05, marginBottom: 14 }}>
              Nothing else does <span className="g">all of this</span>
            </h2>
            <p style={{ fontSize: 15, color: "rgba(226,232,240,.35)", maxWidth: 420, margin: "0 auto" }}>
              GetZen is 8 apps in one. Built for people who want the full picture — not one slice of it.
            </p>
          </div>
        </Reveal>
        <div style={{ maxWidth: 900, margin: "0 auto", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          {/* GetZen panel */}
          <Reveal delay={0} y={24}>
            <div style={{ background: "linear-gradient(135deg,rgba(52,211,153,.09),rgba(34,211,238,.05))", border: "1px solid rgba(52,211,153,.3)", borderRadius: 24, padding: "36px 32px", boxShadow: "0 0 60px rgba(52,211,153,.08)", height: "100%" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 28 }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src="/logo.png" alt="GetZen" style={{ width: 38, height: 38, borderRadius: "50%", objectFit: "cover", border: "2px solid rgba(52,211,153,.4)" }} />
                <div>
                  <p className="font-display" style={{ fontSize: 15, fontWeight: 800, color: "#34D399" }}>GetZen</p>
                  <p style={{ fontSize: 10, color: "rgba(52,211,153,.55)", letterSpacing: ".06em" }}>16 ML models · all-in-one</p>
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  "Voice emotion analysis","Dream decoding with AI","Food–mood correlation",
                  "Sleep staging · 92.98% acc.","Daily brain report","Peak focus forecasting",
                  "EEG neurofeedback (Muse)","Cycle-aware insights",
                  "Breathwork with HRV","Emotion tracking","Sleep duration",
                ].map((label) => (
                  <div key={label} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 14px", borderRadius: 12, background: "rgba(52,211,153,.07)", border: "1px solid rgba(52,211,153,.14)" }}>
                    <div style={{ width: 26, height: 26, borderRadius: 8, background: "rgba(52,211,153,.15)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <Check size={13} color="#34D399" strokeWidth={2.5} />
                    </div>
                    <span style={{ fontSize: 13, color: "rgba(226,232,240,.82)", flex: 1 }}>{label}</span>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#34D399", boxShadow: "0 0 6px #34D399" }} />
                  </div>
                ))}
              </div>
            </div>
          </Reveal>
          {/* Others panel */}
          <Reveal delay={120} y={24}>
            <div style={{ background: "rgba(255,255,255,.02)", border: "1px solid rgba(255,255,255,.07)", borderRadius: 24, padding: "36px 32px", height: "100%" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 28 }}>
                <div style={{ width: 38, height: 38, borderRadius: "50%", background: "rgba(255,255,255,.06)", display: "flex", alignItems: "center", justifyContent: "center", border: "2px solid rgba(255,255,255,.1)" }}>
                  <span style={{ fontSize: 14, color: "rgba(255,255,255,.3)" }}>★</span>
                </div>
                <div>
                  <p className="font-display" style={{ fontSize: 15, fontWeight: 800, color: "rgba(255,255,255,.4)" }}>The Rest</p>
                  <p style={{ fontSize: 10, color: "rgba(255,255,255,.2)", letterSpacing: ".06em" }}>Calm · Headspace · Oura · Muse</p>
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {[
                  ["Voice emotion analysis", false],["Dream decoding with AI", false],["Food–mood correlation", false],
                  ["Sleep staging", false],["Daily brain report", false],["Peak focus forecasting", false],
                  ["EEG neurofeedback", false],["Cycle-aware insights", false],
                  ["Breathwork", true],["Emotion tracking", true],["Sleep duration", true],
                ].map(([label, has]) => (
                  <div key={label as string} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 14px", borderRadius: 12, background: has ? "rgba(255,255,255,.03)" : "transparent", border: `1px solid ${has ? "rgba(255,255,255,.07)" : "transparent"}`, opacity: has ? 1 : 0.35 }}>
                    <div style={{ width: 26, height: 26, borderRadius: 8, background: "rgba(255,255,255,.06)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      {has ? <Check size={13} color="rgba(255,255,255,.4)" strokeWidth={2.5} /> : <X size={12} color="rgba(255,255,255,.2)" strokeWidth={2} />}
                    </div>
                    <span style={{ fontSize: 13, color: has ? "rgba(255,255,255,.5)" : "rgba(255,255,255,.25)", flex: 1, textDecoration: has ? "none" : "line-through", textDecorationColor: "rgba(255,255,255,.15)" }}>{label as string}</span>
                  </div>
                ))}
              </div>
            </div>
          </Reveal>
        </div>
        <Reveal>
          <p style={{ fontSize: 11.5, color: "rgba(255,255,255,.18)", textAlign: "center", marginTop: 20 }}>*No single competitor offers all 11 capabilities. GetZen is the only all-in-one.</p>
        </Reveal>
      </section>

      <hr className="divider" />

      {/* ── Testimonials ── */}
      <section style={{ padding: "90px 32px", maxWidth: 1140, margin: "0 auto" }}>
        <Reveal>
          <div style={{ textAlign: "center", marginBottom: 52 }}>
            <div className="tag" style={{ marginBottom: 20, display: "inline-flex" }}><Sparkles size={10} />Real people · Real results</div>
            <h2 className="font-display" style={{ fontSize: "clamp(26px,4vw,46px)", fontWeight: 800, letterSpacing: "-.025em", lineHeight: 1.1, marginBottom: 14 }}>
              What early users are <span className="g">discovering</span>
            </h2>
            <p style={{ fontSize: 15, color: "rgba(226,232,240,.38)", maxWidth: 440, margin: "0 auto" }}>
              Beta testers who let GetZen decode their mornings every day.
            </p>
          </div>
        </Reveal>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
          <Reveal delay={0}>
            <TestimonialCard
              name="Priya M." role="Yoga instructor, Bangalore"
              quote="GetZen showed me my afternoon crash was tied to my lunch — I changed one meal and felt the difference next day. Nothing else ever gave me that link."
              metric="+31% mood stability" color="#34D399"
              avatar="photo-1494790108377-be9c29b29330"
            />
          </Reveal>
          <Reveal delay={120}>
            <TestimonialCard
              name="James K." role="Startup founder, London"
              quote="The daily brain report is the first thing I check each morning. It predicted a high-stress day from just my voice — and it was exactly right."
              metric="Focus peak predicted ±18 min" color="#22D3EE"
              avatar="photo-1507003211169-0a1dd7228f2d"
            />
          </Reveal>
          <Reveal delay={240}>
            <TestimonialCard
              name="Sara L." role="Therapist & meditator, NYC"
              quote="Dream decoding changed my entire journaling practice. I never connected my recurring water dreams to my anxiety — GetZen decoded it in week one."
              metric="3 hidden patterns decoded" color="#a78bfa"
              avatar="photo-1438761681033-6461ffad8d80"
            />
          </Reveal>
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
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
            <WaitlistForm />
          </div>
          {/* Stars + count adjacent to CTA */}
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 28 }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <Stars rating={4.8} count="2.1k" />
              <span style={{ fontSize: 11.5, color: "rgba(255,255,255,.2)" }}><strong style={{ color: "rgba(255,255,255,.35)" }}>2,847</strong> people on the list · early access only</span>
            </div>
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
