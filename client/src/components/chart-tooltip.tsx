/**
 * Shared compact tooltip for all Recharts charts in the app.
 * Props are passed directly from Recharts — typed loosely to avoid
 * the Payload<ValueType, NameType> generics mismatch.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function ChartTooltip({ active, payload, label, unit = "%", formatter }: any) {
  if (!active || !payload?.length) return null;

  return (
    <div
      className="rounded-xl border border-border/60 shadow-xl text-xs bg-card/95"
      style={{
        padding: "8px 10px",
        minWidth: 90,
        pointerEvents: "none",
      }}
    >
      {label && (
        <p className="text-muted-foreground font-mono mb-1.5 text-[10px]">{label}</p>
      )}
      <div className="space-y-1">
        {payload.map((entry: any) => {
          const color = entry.stroke ?? entry.color ?? entry.fill ?? "hsl(152,60%,48%)";
          const displayValue = formatter
            ? formatter(entry.value, entry.name)
            : `${entry.value}${unit}`;
          return (
            <div key={entry.dataKey ?? entry.name} className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
              <span className="text-muted-foreground">{entry.name}</span>
              <span className="font-mono ml-auto" style={{ color }}>{displayValue}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
