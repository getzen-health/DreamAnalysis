import { AreaChart, Area, XAxis, Tooltip, ResponsiveContainer } from "recharts";

export interface EmotionDataPoint {
  time: string;
  valence: number;
  arousal: number;
  stress: number;
  label?: string;
}

export interface EmotionFlowProps {
  data: EmotionDataPoint[];
  height?: number;
}

function getEmotionState(valence: number, arousal: number): string {
  if (valence > 0.3 && arousal > 0.6) return "Energized";
  if (valence > 0.3 && arousal <= 0.6) return "Calm";
  if (valence <= -0.3 && arousal > 0.6) return "Stressed";
  if (valence <= -0.3 && arousal <= 0.6) return "Low energy";
  if (valence > 0 && arousal > 0.5) return "Alert";
  if (valence > 0) return "Relaxed";
  if (arousal > 0.6) return "Tense";
  return "Neutral";
}

interface TooltipPayloadEntry {
  name: string;
  value: number;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  const valenceEntry = payload.find((p) => p.name === "valence");
  const arousalEntry = payload.find((p) => p.name === "arousal");
  const stressEntry = payload.find((p) => p.name === "stress");

  const valence = valenceEntry?.value ?? 0;
  const arousal = arousalEntry?.value ?? 0;
  const stress = stressEntry?.value ?? 0;
  const state = getEmotionState(valence, arousal);

  return (
    <div className="rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm shadow-lg">
      <p className="text-gray-400 mb-1">{label}</p>
      <p className="text-gray-100 font-medium">{state}</p>
      <p className="text-gray-400 text-xs mt-1">
        Stress: {(stress * 100).toFixed(0)}%
      </p>
    </div>
  );
}

export default function EmotionFlow({ data, height = 200 }: EmotionFlowProps) {
  return (
    <div className="bg-gray-900 rounded-xl p-4">
      <h3 className="text-gray-100 text-sm font-semibold mb-3 tracking-wide uppercase">
        Daily Emotion Flow
      </h3>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart
          data={data}
          margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="positiveGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ea580c" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#c2410c" stopOpacity={0.2} />
            </linearGradient>
            <linearGradient id="negativeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#64748b" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.7} />
            </linearGradient>
            <linearGradient id="arousalGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#a855f7" stopOpacity={0.5} />
              <stop offset="95%" stopColor="#a855f7" stopOpacity={0.05} />
            </linearGradient>
          </defs>

          <XAxis
            dataKey="time"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
          />

          {/* Positive valence — warm colors (amber/orange) */}
          <Area
            type="monotone"
            dataKey="valence"
            name="valence"
            stroke="#ea580c"
            strokeWidth={2}
            fill="url(#positiveGradient)"
            baseValue={0}
          />

          {/* Arousal — purple overlay */}
          <Area
            type="monotone"
            dataKey="arousal"
            name="arousal"
            stroke="#a855f7"
            strokeWidth={1}
            strokeDasharray="4 2"
            fill="url(#arousalGradient)"
            baseValue={0}
          />

          {/* Stress — hidden from chart rendering but included in tooltip */}
          <Area
            type="monotone"
            dataKey="stress"
            name="stress"
            stroke="transparent"
            fill="transparent"
            strokeWidth={0}
          />

          <Tooltip content={<CustomTooltip />} />
        </AreaChart>
      </ResponsiveContainer>

      <div className="flex gap-4 mt-2">
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-1.5 rounded bg-amber-400" />
          <span className="text-gray-400 text-xs">Valence</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-1.5 rounded bg-purple-500" />
          <span className="text-gray-400 text-xs">Arousal</span>
        </div>
      </div>
    </div>
  );
}
