/**
 * ParticleTouch — canvas overlay that spawns emotion-colored particles
 * on every touch/click. Makes the entire app feel alive and responsive.
 *
 * Particle behavior adapts to current emotional state:
 *   - Calm: few gentle particles, slow drift
 *   - Excited: burst of bright particles, fast scatter
 *   - Stressed: tight amber spirals
 */

import { useEffect, useRef } from "react";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";

interface TouchParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  size: number;
  color: string;
}

const EMOTION_COLORS: Record<string, string> = {
  happy: "160, 230, 120",
  excited: "250, 200, 60",
  calm: "80, 210, 200",
  relaxed: "80, 200, 190",
  content: "100, 210, 150",
  serene: "80, 200, 180",
  focused: "100, 170, 250",
  sad: "130, 140, 250",
  angry: "250, 120, 130",
  anxious: "190, 130, 250",
  stressed: "250, 170, 80",
  fear: "180, 120, 240",
  neutral: "160, 160, 180",
};

export function ParticleTouch() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<TouchParticle[]>([]);
  const animRef = useRef<number>(0);
  const { emotion } = useCurrentEmotion();

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
    };
    resize();
    window.addEventListener("resize", resize);

    const spawnParticles = (x: number, y: number) => {
      const arousal = emotion?.arousal ?? 0.4;
      const count = Math.floor(4 + arousal * 10);
      const speed = 1 + arousal * 3;
      const colorKey = emotion?.emotion ?? "neutral";
      const rgb = EMOTION_COLORS[colorKey] ?? EMOTION_COLORS.neutral;

      for (let i = 0; i < count; i++) {
        const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.8;
        const v = speed * (0.5 + Math.random() * 0.5);
        particlesRef.current.push({
          x,
          y,
          vx: Math.cos(angle) * v,
          vy: Math.sin(angle) * v,
          life: 1,
          size: 1.5 + Math.random() * 2.5,
          color: rgb,
        });
      }
    };

    const handlePointer = (e: PointerEvent) => {
      // Only spawn on actual clicks/taps, not drags
      if (e.type === "pointerdown") {
        spawnParticles(e.clientX, e.clientY);
      }
    };

    document.addEventListener("pointerdown", handlePointer, { passive: true });

    const animate = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;

      ctx.clearRect(0, 0, w, h);

      for (let i = particlesRef.current.length - 1; i >= 0; i--) {
        const p = particlesRef.current[i];
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.03; // slight gravity
        p.vx *= 0.98; // friction
        p.vy *= 0.98;
        p.life -= 0.025;

        if (p.life <= 0) {
          particlesRef.current.splice(i, 1);
          continue;
        }

        const alpha = p.life * 0.6;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${p.color}, ${alpha})`;
        ctx.fill();

        // Soft glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * p.life * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${p.color}, ${alpha * 0.15})`;
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animRef.current);
      document.removeEventListener("pointerdown", handlePointer);
      window.removeEventListener("resize", resize);
    };
  }, [emotion]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 9999 }}
      aria-hidden="true"
    />
  );
}
