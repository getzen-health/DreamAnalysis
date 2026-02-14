interface EmotionWheelProps {
  probabilities: Record<string, number>;
  dominantEmotion: string;
  confidence: number;
}

const EMOTION_CONFIG: Record<string, { color: string; emoji: string; angle: number }> = {
  happy: { color: "hsl(120, 100%, 55%)", emoji: "😊", angle: 0 },
  sad: { color: "hsl(220, 70%, 50%)", emoji: "😢", angle: 60 },
  angry: { color: "hsl(0, 80%, 50%)", emoji: "😠", angle: 120 },
  fearful: { color: "hsl(270, 70%, 65%)", emoji: "😨", angle: 180 },
  relaxed: { color: "hsl(195, 100%, 50%)", emoji: "😌", angle: 240 },
  focused: { color: "hsl(45, 100%, 50%)", emoji: "🎯", angle: 300 },
};

export function EmotionWheel({ probabilities, dominantEmotion, confidence }: EmotionWheelProps) {
  const radius = 90;
  const center = 120;

  return (
    <div className="flex flex-col items-center">
      <svg width={240} height={240} viewBox="0 0 240 240">
        {/* Background circle */}
        <circle cx={center} cy={center} r={radius + 10} fill="none" stroke="hsl(var(--border))" strokeWidth={1} opacity={0.3} />
        <circle cx={center} cy={center} r={radius / 2} fill="none" stroke="hsl(var(--border))" strokeWidth={1} opacity={0.2} />

        {/* Emotion segments */}
        {Object.entries(EMOTION_CONFIG).map(([emotion, config]) => {
          const prob = probabilities[emotion] || 0;
          const r = radius * Math.max(0.3, prob * 2.5);
          const angleRad = (config.angle - 90) * (Math.PI / 180);
          const x = center + r * Math.cos(angleRad);
          const y = center + r * Math.sin(angleRad);
          const isDominant = emotion === dominantEmotion;

          return (
            <g key={emotion}>
              {/* Connection line */}
              <line
                x1={center}
                y1={center}
                x2={x}
                y2={y}
                stroke={config.color}
                strokeWidth={isDominant ? 2.5 : 1.5}
                opacity={isDominant ? 0.8 : 0.3}
              />
              {/* Emotion node */}
              <circle
                cx={x}
                cy={y}
                r={isDominant ? 18 : 14}
                fill={config.color}
                opacity={isDominant ? 0.9 : 0.4}
                className={isDominant ? "animate-pulse" : ""}
              />
              {/* Label */}
              <text
                x={center + (radius + 25) * Math.cos(angleRad)}
                y={center + (radius + 25) * Math.sin(angleRad)}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-foreground text-[10px]"
                opacity={0.7}
              >
                {config.emoji} {Math.round(prob * 100)}%
              </text>
            </g>
          );
        })}

        {/* Center text */}
        <text x={center} y={center - 8} textAnchor="middle" className="fill-foreground text-lg font-bold">
          {EMOTION_CONFIG[dominantEmotion]?.emoji || "🧠"}
        </text>
        <text x={center} y={center + 10} textAnchor="middle" className="fill-foreground text-[10px] capitalize font-semibold">
          {dominantEmotion}
        </text>
      </svg>

      <div className="text-center mt-2">
        <span className="text-sm text-foreground/60">Confidence: </span>
        <span className="text-sm font-mono font-bold text-primary">{Math.round(confidence * 100)}%</span>
      </div>
    </div>
  );
}
