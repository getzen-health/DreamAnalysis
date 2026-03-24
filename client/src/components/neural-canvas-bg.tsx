/**
 * NeuralCanvasBg — living, breathing background that responds to
 * the user's emotional state. Renders an organic particle/noise field
 * using Canvas 2D (GPU-lite, works everywhere including Android WebView).
 *
 * Emotional states drive:
 *   - Color palette (warm/cool shift)
 *   - Particle speed & density
 *   - Flow direction (upward = positive, downward = negative)
 *   - Turbulence (stress = chaotic, calm = smooth)
 */

import { useEffect, useRef, useCallback, useState } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  life: number;
  maxLife: number;
  hue: number;
  sat: number;
  light: number;
}

// Emotion-driven color palettes (HSL hue ranges)
const EMOTION_PALETTES: Record<string, { hue: number; sat: number; light: number }> = {
  happy:    { hue: 150, sat: 60, light: 50 },
  excited:  { hue: 45,  sat: 70, light: 55 },
  calm:     { hue: 175, sat: 50, light: 45 },
  relaxed:  { hue: 170, sat: 45, light: 42 },
  content:  { hue: 155, sat: 50, light: 48 },
  serene:   { hue: 180, sat: 45, light: 40 },
  focused:  { hue: 220, sat: 55, light: 50 },
  sad:      { hue: 235, sat: 40, light: 35 },
  angry:    { hue: 0,   sat: 55, light: 40 },
  anxious:  { hue: 280, sat: 45, light: 40 },
  stressed: { hue: 25,  sat: 50, light: 42 },
  fear:     { hue: 270, sat: 50, light: 38 },
  neutral:  { hue: 260, sat: 30, light: 35 },
};

const DEFAULT_PALETTE = { hue: 260, sat: 25, light: 30 };
const PARTICLE_COUNT = 35;

/** Read emotion from localStorage directly to avoid hook dependency issues */
function readEmotion(): { emotion: string; arousal: number; valence: number } | null {
  try {
    const raw = localStorage.getItem("ndw_last_emotion");
    if (!raw) return null;
    const data = JSON.parse(raw);
    const r = data?.result ?? data;
    if (!r?.emotion) return null;
    return { emotion: r.emotion, arousal: r.arousal ?? 0.3, valence: r.valence ?? 0 };
  } catch { return null; }
}

export function NeuralCanvasBg() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animRef = useRef<number>(0);
  const [emotion, setEmotion] = useState(readEmotion);

  // Listen for emotion updates
  useEffect(() => {
    const handler = () => setEmotion(readEmotion());
    window.addEventListener("ndw-emotion-update", handler);
    window.addEventListener("ndw-voice-updated", handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener("ndw-emotion-update", handler);
      window.removeEventListener("ndw-voice-updated", handler);
      window.removeEventListener("storage", handler);
    };
  }, []);

  // Get target palette from current emotion
  const targetPalette = emotion
    ? EMOTION_PALETTES[emotion.emotion] ?? DEFAULT_PALETTE
    : DEFAULT_PALETTE;

  // Smooth interpolation state
  const currentPalette = useRef({ ...DEFAULT_PALETTE });
  const currentArousal = useRef(0.3);
  const currentValence = useRef(0);

  const initParticles = useCallback((w: number, h: number) => {
    const particles: Particle[] = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: 1 + Math.random() * 2.5,
        life: Math.random(),
        maxLife: 0.6 + Math.random() * 0.4,
        hue: currentPalette.current.hue + (Math.random() - 0.5) * 30,
        sat: currentPalette.current.sat,
        light: currentPalette.current.light,
      });
    }
    particlesRef.current = particles;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      ctx.scale(dpr, dpr);
      if (particlesRef.current.length === 0) {
        initParticles(window.innerWidth, window.innerHeight);
      }
    };

    resize();
    window.addEventListener("resize", resize);

    let time = 0;

    const animate = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      time += 0.008;

      // Smoothly interpolate palette toward target
      const lerp = 0.02;
      currentPalette.current.hue += (targetPalette.hue - currentPalette.current.hue) * lerp;
      currentPalette.current.sat += (targetPalette.sat - currentPalette.current.sat) * lerp;
      currentPalette.current.light += (targetPalette.light - currentPalette.current.light) * lerp;

      // Smooth arousal/valence
      const targetArousal = emotion?.arousal ?? 0.3;
      const targetValence = emotion?.valence ?? 0;
      currentArousal.current += (targetArousal - currentArousal.current) * 0.01;
      currentValence.current += (targetValence - currentValence.current) * 0.01;

      const arousal = currentArousal.current;
      const valence = currentValence.current;

      // Clear with very subtle fade (creates trails)
      ctx.fillStyle = `hsla(${currentPalette.current.hue}, ${currentPalette.current.sat * 0.3}%, ${Math.max(3, currentPalette.current.light * 0.15)}%, 0.12)`;
      ctx.fillRect(0, 0, w, h);

      // Update & draw particles
      const speed = 0.15 + arousal * 0.6;
      const flowY = valence > 0 ? -0.15 : 0.1; // positive = upward drift

      for (const p of particlesRef.current) {
        // Noise-based flow
        const nx = Math.sin(p.x * 0.003 + time) * 0.5 + Math.cos(p.y * 0.004 + time * 0.7) * 0.3;
        const ny = Math.cos(p.y * 0.003 + time * 0.8) * 0.5 + Math.sin(p.x * 0.004 + time * 0.6) * 0.3;

        p.vx += nx * speed * 0.02;
        p.vy += (ny + flowY) * speed * 0.02;

        // Damping
        p.vx *= 0.985;
        p.vy *= 0.985;

        p.x += p.vx;
        p.y += p.vy;

        // Wrap
        if (p.x < -20) p.x = w + 20;
        if (p.x > w + 20) p.x = -20;
        if (p.y < -20) p.y = h + 20;
        if (p.y > h + 20) p.y = -20;

        // Life cycle
        p.life += 0.003;
        if (p.life > p.maxLife) {
          p.life = 0;
          p.hue = currentPalette.current.hue + (Math.random() - 0.5) * 40;
          p.sat = currentPalette.current.sat + (Math.random() - 0.5) * 10;
          p.light = currentPalette.current.light + Math.random() * 15;
        }

        // Opacity based on life
        const lifeRatio = p.life / p.maxLife;
        const opacity = lifeRatio < 0.2 ? lifeRatio * 5 : lifeRatio > 0.8 ? (1 - lifeRatio) * 5 : 1;

        // Draw
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, ${p.sat}%, ${p.light}%, ${opacity * 0.35})`;
        ctx.fill();

        // Glow for larger particles
        if (p.size > 2) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 3, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${p.hue}, ${p.sat}%, ${p.light}%, ${opacity * 0.06})`;
          ctx.fill();
        }
      }

      // Draw subtle connection lines between nearby particles
      for (let i = 0; i < particlesRef.current.length; i++) {
        for (let j = i + 1; j < particlesRef.current.length; j++) {
          const a = particlesRef.current[i];
          const b = particlesRef.current[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            const lineOpacity = (1 - dist / 120) * 0.06;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `hsla(${currentPalette.current.hue}, ${currentPalette.current.sat}%, ${currentPalette.current.light + 10}%, ${lineOpacity})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [targetPalette, initParticles, emotion]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0, opacity: 0.7 }}
      aria-hidden="true"
    />
  );
}
