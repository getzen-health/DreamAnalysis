import { useState } from "react";

export interface HeatmapCell {
  day: string;
  hour: number;
  stress: number;
}

export interface EmotionLandscapeProps {
  data: HeatmapCell[];
  title?: string;
}

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const VISIBLE_HOURS = [6, 9, 12, 15, 18, 21];
const HOUR_LABELS: Record<number, string> = {
  6: "6am",
  9: "9am",
  12: "12pm",
  15: "3pm",
  18: "6pm",
  21: "9pm",
};
// Render hours 6 through 23 (18 rows)
const HOUR_RANGE = Array.from({ length: 18 }, (_, i) => i + 6);

function stressToColor(stress: number): string {
  if (Number.isNaN(stress)) return "#374151"; // no data — gray-700
  if (stress < 0.33) return "#06b6d4";         // low — cyan-500
  if (stress < 0.66) return "#d4a017";         // medium — golden honey
  return "#e879a8";                            // high — warm coral
}

function formatHour(hour: number): string {
  if (hour === 0) return "12am";
  if (hour === 12) return "12pm";
  return hour < 12 ? `${hour}am` : `${hour - 12}pm`;
}

export default function EmotionLandscape({
  data,
  title = "Weekly Stress Landscape",
}: EmotionLandscapeProps) {
  const [tooltip, setTooltip] = useState<{
    day: string;
    hour: number;
    stress: number;
    x: number;
    y: number;
  } | null>(null);

  // Build fast lookup: "Mon-9" → stress value
  const lookup = new Map<string, number>();
  for (const cell of data) {
    lookup.set(`${cell.day}-${cell.hour}`, cell.stress);
  }

  return (
    <div className="bg-gray-900 rounded-xl p-4 relative">
      {title && (
        <h3 className="text-gray-100 text-sm font-semibold mb-4 tracking-wide uppercase">
          {title}
        </h3>
      )}

      {/* Day column headers */}
      <div className="flex ml-10 mb-1">
        {DAYS.map((day) => (
          <div
            key={day}
            className="flex-1 text-center text-gray-400 text-xs font-medium"
          >
            {day}
          </div>
        ))}
      </div>

      {/* Grid rows — one per hour */}
      <div className="flex flex-col gap-px">
        {HOUR_RANGE.map((hour) => (
          <div key={hour} className="flex items-center gap-px">
            {/* Row label — only show on visible hours */}
            <div className="w-10 flex-shrink-0 text-right pr-2">
              {VISIBLE_HOURS.includes(hour) ? (
                <span className="text-gray-500 text-xs">
                  {HOUR_LABELS[hour]}
                </span>
              ) : null}
            </div>

            {/* 7 day cells */}
            {DAYS.map((day) => {
              const stress = lookup.get(`${day}-${hour}`) ?? NaN;
              const color = stressToColor(stress);
              return (
                <div
                  key={day}
                  className="flex-1 h-4 rounded-sm cursor-pointer transition-opacity hover:opacity-80"
                  style={{ backgroundColor: color }}
                  onMouseEnter={(e) => {
                    const rect = (e.target as HTMLElement).getBoundingClientRect();
                    setTooltip({
                      day,
                      hour,
                      stress,
                      x: rect.left + rect.width / 2,
                      y: rect.top,
                    });
                  }}
                  onMouseLeave={() => setTooltip(null)}
                />
              );
            })}
          </div>
        ))}
      </div>

      {/* Color scale legend */}
      <div className="flex items-center gap-3 mt-3">
        <span className="text-gray-500 text-xs">Stress:</span>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm bg-cyan-600" />
          <span className="text-gray-400 text-xs">Low</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm bg-yellow-500" />
          <span className="text-gray-400 text-xs">Medium</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm bg-rose-500" />
          <span className="text-gray-400 text-xs">High</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm bg-gray-700" />
          <span className="text-gray-400 text-xs">No data</span>
        </div>
      </div>

      {/* Floating tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm shadow-lg -translate-x-1/2 -translate-y-full"
          style={{ left: tooltip.x, top: tooltip.y - 8 }}
        >
          <p className="text-gray-100 font-medium">
            {tooltip.day} {formatHour(tooltip.hour)}
          </p>
          <p className="text-gray-400 text-xs">
            {Number.isNaN(tooltip.stress)
              ? "No data"
              : `Stress: ${(tooltip.stress * 100).toFixed(0)}%`}
          </p>
        </div>
      )}
    </div>
  );
}
