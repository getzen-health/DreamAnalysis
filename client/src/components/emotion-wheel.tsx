/**
 * EmotionWheel — premium radial emotion visualization with gradients,
 * glow effects, and smooth animated transitions.
 *
 * Inspired by: How We Feel's 2D emotion space, Oura's score rings,
 * Apple Health's clean data visualization.
 */

import { motion } from "framer-motion";
import { springs } from "@/lib/animations";

interface EmotionWheelProps {
  probabilities: Record<string, number>;
  dominantEmotion: string;
  confidence: number;
}

const EMOTION_CONFIG: Record<string, {
  gradient: [string, string];
  glow: string;
  emoji: string;
  angle: number;
}> = {
  happy:   { gradient: ["#4ADE80", "#22C55E"], glow: "rgba(74, 222, 128, 0.4)",  emoji: "\u{1F60A}", angle: 0 },
  sad:     { gradient: ["#818CF8", "#6366F1"], glow: "rgba(129, 140, 248, 0.4)", emoji: "\u{1F622}", angle: 60 },
  angry:   { gradient: ["#FB7185", "#EF4444"], glow: "rgba(251, 113, 133, 0.4)", emoji: "\u{1F620}", angle: 120 },
  fearful: { gradient: ["#C084FC", "#A855F7"], glow: "rgba(192, 132, 252, 0.4)", emoji: "\u{1F628}", angle: 180 },
  relaxed: { gradient: ["#2DD4BF", "#14B8A6"], glow: "rgba(45, 212, 191, 0.4)",  emoji: "\u{1F60C}", angle: 240 },
  focused: { gradient: ["#60A5FA", "#3B82F6"], glow: "rgba(96, 165, 250, 0.4)",  emoji: "\u{1F3AF}", angle: 300 },
};

export function EmotionWheel({ probabilities, dominantEmotion, confidence }: EmotionWheelProps) {
  const radius = 85;
  const center = 120;

  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        {/* Ambient glow behind wheel */}
        <div
          className="absolute inset-0 rounded-full blur-3xl opacity-20"
          style={{
            background: `radial-gradient(circle, ${EMOTION_CONFIG[dominantEmotion]?.glow ?? "rgba(124, 58, 237, 0.3)"}, transparent 70%)`,
          }}
        />

        <svg width={240} height={240} viewBox="0 0 240 240" className="relative z-10">
          <defs>
            {/* Gradient definitions for each emotion */}
            {Object.entries(EMOTION_CONFIG).map(([emotion, config]) => (
              <linearGradient key={emotion} id={`grad-${emotion}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor={config.gradient[0]} />
                <stop offset="100%" stopColor={config.gradient[1]} />
              </linearGradient>
            ))}
            {/* Glow filter */}
            <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            {/* Soft outer glow filter for dominant */}
            <filter id="dominant-glow" x="-100%" y="-100%" width="300%" height="300%">
              <feGaussianBlur stdDeviation="8" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background rings — subtle depth */}
          <circle cx={center} cy={center} r={radius + 12} fill="none"
            stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
          <circle cx={center} cy={center} r={radius * 0.65} fill="none"
            stroke="rgba(255,255,255,0.03)" strokeWidth={1} />
          <circle cx={center} cy={center} r={radius * 0.35} fill="none"
            stroke="rgba(255,255,255,0.02)" strokeWidth={1} />

          {/* Emotion nodes and connections */}
          {Object.entries(EMOTION_CONFIG).map(([emotion, config], i) => {
            const prob = probabilities[emotion] || 0;
            const r = radius * Math.max(0.35, prob * 2.2);
            const angleRad = (config.angle - 90) * (Math.PI / 180);
            const x = center + r * Math.cos(angleRad);
            const y = center + r * Math.sin(angleRad);
            const isDominant = emotion === dominantEmotion;
            const labelR = radius + 22;
            const lx = center + labelR * Math.cos(angleRad);
            const ly = center + labelR * Math.sin(angleRad);

            return (
              <motion.g
                key={emotion}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.08, ...springs.bouncy }}
              >
                {/* Connection line — gradient opacity */}
                <motion.line
                  x1={center} y1={center} x2={x} y2={y}
                  stroke={`url(#grad-${emotion})`}
                  strokeWidth={isDominant ? 2 : 1}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: isDominant ? 0.6 : 0.15 }}
                  transition={{ delay: i * 0.08 + 0.2, duration: 0.4 }}
                />

                {/* Node — gradient fill with glow */}
                <motion.circle
                  cx={x} cy={y}
                  r={isDominant ? 18 : 12}
                  fill={`url(#grad-${emotion})`}
                  filter={isDominant ? "url(#dominant-glow)" : "url(#node-glow)"}
                  initial={{ r: 0 }}
                  animate={{
                    r: isDominant ? 18 : 12,
                    opacity: isDominant ? 1 : 0.5,
                  }}
                  transition={{ delay: i * 0.08, ...springs.bouncy }}
                />

                {/* Emoji inside node */}
                <text
                  x={x} y={y}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className={isDominant ? "text-sm" : "text-[10px]"}
                  style={{ pointerEvents: "none" }}
                >
                  {config.emoji}
                </text>

                {/* Probability label */}
                <text
                  x={lx} y={ly}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-[11px] font-medium"
                  fill={isDominant ? config.gradient[0] : "rgba(255,255,255,0.35)"}
                >
                  {Math.round(prob * 100)}%
                </text>
              </motion.g>
            );
          })}

          {/* Center — dominant emotion display */}
          <motion.g
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.5, ...springs.bouncy }}
          >
            {/* Center glow */}
            <circle
              cx={center} cy={center} r={28}
              fill={EMOTION_CONFIG[dominantEmotion]?.glow ?? "rgba(124, 58, 237, 0.2)"}
              opacity={0.3}
            />
            <circle
              cx={center} cy={center} r={22}
              fill="rgba(255,255,255,0.05)"
              stroke="rgba(255,255,255,0.08)"
              strokeWidth={1}
            />
            <text
              x={center} y={center - 2}
              textAnchor="middle"
              dominantBaseline="central"
              className="text-xl"
            >
              {EMOTION_CONFIG[dominantEmotion]?.emoji || "\u{1F9E0}"}
            </text>
          </motion.g>
        </svg>
      </div>

      {/* Confidence bar below */}
      <motion.div
        className="flex flex-col items-center gap-1.5 mt-3 w-full max-w-[180px]"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7, duration: 0.3 }}
      >
        <div className="flex justify-between w-full">
          <span className="text-xs font-medium capitalize" style={{
            color: EMOTION_CONFIG[dominantEmotion]?.gradient[0] ?? "rgba(255,255,255,0.6)",
          }}>
            {dominantEmotion}
          </span>
          <span className="text-xs font-mono font-bold" style={{
            color: EMOTION_CONFIG[dominantEmotion]?.gradient[0] ?? "rgba(255,255,255,0.6)",
          }}>
            {Math.round(confidence * 100)}%
          </span>
        </div>
        <div className="w-full h-1.5 rounded-full bg-foreground/[0.06] overflow-hidden">
          <motion.div
            className="h-full rounded-full"
            style={{
              background: `linear-gradient(90deg, ${EMOTION_CONFIG[dominantEmotion]?.gradient[0] ?? "#818CF8"}, ${EMOTION_CONFIG[dominantEmotion]?.gradient[1] ?? "#6366F1"})`,
            }}
            initial={{ width: "0%" }}
            animate={{ width: `${Math.round(confidence * 100)}%` }}
            transition={{ duration: 1, ease: [0.22, 1, 0.36, 1], delay: 0.8 }}
          />
        </div>
      </motion.div>
    </div>
  );
}
