import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown } from "lucide-react";
import { useTheme } from "@/hooks/use-theme";

interface ScoreCircleProps {
  value: number;
  label: string;
  sublabel?: string;
  gradientId: string;
  colorFrom: string;
  colorTo: string;
  size?: "lg" | "md" | "sm";
  trend?: number;
}

const SIZES = {
  lg: { svg: 180, r: 68, stroke: 8, fontSize: 38, labelSize: 11, trendSize: 10 },
  md: { svg: 140, r: 52, stroke: 7, fontSize: 28, labelSize: 10, trendSize: 9 },
  sm: { svg: 100, r: 36, stroke: 5, fontSize: 20, labelSize: 9, trendSize: 8 },
};

export function ScoreCircle({
  value,
  label,
  sublabel,
  gradientId,
  colorFrom,
  colorTo,
  size = "lg",
  trend,
}: ScoreCircleProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [animatedValue, setAnimatedValue] = useState(0);
  const s = SIZES[size];
  const cx = s.svg / 2;
  const cy = s.svg / 2;
  const circumference = 2 * Math.PI * s.r;
  const arcLength = circumference * 0.75; // 270 degrees
  const gapLength = circumference * 0.25; // 90 degrees
  const dashOffset = arcLength * (1 - animatedValue / 100);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedValue(value), 100);
    return () => clearTimeout(timer);
  }, [value]);

  return (
    <div className="flex flex-col items-center">
      <svg
        width={s.svg}
        height={s.svg}
        viewBox={`0 0 ${s.svg} ${s.svg}`}
        className="drop-shadow-lg"
      >
        <defs>
          <linearGradient
            id={gradientId}
            gradientUnits="userSpaceOnUse"
            x1={s.svg * 0.15}
            y1={s.svg * 0.15}
            x2={s.svg * 0.85}
            y2={s.svg * 0.85}
          >
            <stop offset="0%" stopColor={colorFrom} />
            <stop offset="100%" stopColor={colorTo} />
          </linearGradient>
        </defs>

        {/* Background track */}
        <circle
          cx={cx}
          cy={cy}
          r={s.r}
          fill="none"
          stroke={isDark ? "hsl(220, 18%, 15%)" : "hsl(220, 14%, 88%)"}
          strokeWidth={s.stroke}
          strokeDasharray={`${arcLength} ${gapLength}`}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
          opacity={0.5}
        />

        {/* Animated score arc */}
        <circle
          cx={cx}
          cy={cy}
          r={s.r}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={s.stroke}
          strokeDasharray={`${arcLength} ${gapLength}`}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
          style={{
            transition: "stroke-dashoffset 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)",
          }}
        />

        {/* Score number */}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          dominantBaseline="central"
          fill={isDark ? "hsl(38, 20%, 92%)" : "hsl(222, 25%, 10%)"}
          fontSize={s.fontSize}
          fontWeight="600"
          fontFamily="Inter, system-ui, sans-serif"
        >
          {Math.round(animatedValue)}
        </text>

        {/* Label */}
        <text
          x={cx}
          y={cy + s.fontSize * 0.55}
          textAnchor="middle"
          dominantBaseline="central"
          fill={isDark ? "hsl(220, 12%, 52%)" : "hsl(220, 12%, 42%)"}
          fontSize={s.labelSize}
          fontFamily="Inter, system-ui, sans-serif"
        >
          {label}
        </text>
      </svg>

      {/* Trend badge */}
      {trend !== undefined && trend !== 0 && (
        <div className="flex items-center gap-1 mt-1">
          {trend > 0 ? (
            <TrendingUp
              className="text-success"
              style={{ width: s.trendSize + 2, height: s.trendSize + 2 }}
            />
          ) : (
            <TrendingDown
              className="text-destructive"
              style={{ width: s.trendSize + 2, height: s.trendSize + 2 }}
            />
          )}
          <span
            className={`font-mono ${trend > 0 ? "text-success" : "text-destructive"}`}
            style={{ fontSize: s.trendSize }}
          >
            {trend > 0 ? "+" : ""}
            {trend} {sublabel || "vs yesterday"}
          </span>
        </div>
      )}
    </div>
  );
}
