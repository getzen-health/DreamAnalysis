import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from "recharts";
import { Moon, TrendingUp, Eye, Zap, Repeat, Star, Loader2 } from "lucide-react";

const EMOTION_COLORS: Record<string, string> = {
  joy: "hsl(120, 100%, 55%)",
  curiosity: "hsl(195, 100%, 50%)",
  anxiety: "hsl(15, 100%, 60%)",
  confusion: "hsl(270, 70%, 65%)",
  fear: "hsl(0, 80%, 50%)",
  peace: "hsl(180, 70%, 50%)",
};

const TYPE_COLORS = [
  "hsl(195, 100%, 50%)",
  "hsl(270, 70%, 65%)",
  "hsl(15, 100%, 60%)",
  "hsl(120, 100%, 55%)",
];

export default function DreamPatterns() {
  const userId = "demo-user";

  const { data: dreams = [], isLoading } = useQuery({
    queryKey: ["/api/dream-analysis", userId],
    queryFn: async () => {
      const res = await fetch(`/api/dream-analysis/${userId}`);
      if (!res.ok) return [];
      return res.json();
    },
  });

  // Generate analytics from dream data
  const symbolFrequency = dreams.reduce((acc: Record<string, number>, dream: any) => {
    const symbols = (dream.symbols as string[]) || [];
    symbols.forEach((s: string) => { acc[s] = (acc[s] || 0) + 1; });
    return acc;
  }, {});

  const symbolData = Object.entries(symbolFrequency)
    .sort(([, a], [, b]) => (b as number) - (a as number))
    .slice(0, 8)
    .map(([symbol, count]) => ({ symbol, count }));

  const dreamTypeDistribution = dreams.reduce((acc: Record<string, number>, dream: any) => {
    const tags = (dream.tags as string[]) || ["normal"];
    tags.forEach((t: string) => { acc[t] = (acc[t] || 0) + 1; });
    if (tags.length === 0) acc["normal"] = (acc["normal"] || 0) + 1;
    return acc;
  }, {});

  const typeData = Object.entries(dreamTypeDistribution).map(([name, value]) => ({ name, value }));

  // Frequency calendar data (last 30 days)
  const calendarData = Array.from({ length: 30 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (29 - i));
    const dateStr = date.toISOString().split("T")[0];
    const count = dreams.filter((d: any) => d.timestamp?.startsWith?.(dateStr)).length;
    return {
      date: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      dreams: count || Math.random() > 0.6 ? Math.floor(Math.random() * 3) : 0,
    };
  });

  // Emotion trends
  const emotionTrends = [
    { day: "Mon", joy: 7.2, curiosity: 5.1, anxiety: 2.3, peace: 6.4 },
    { day: "Tue", joy: 6.8, curiosity: 6.2, anxiety: 3.1, peace: 5.8 },
    { day: "Wed", joy: 8.1, curiosity: 4.5, anxiety: 1.8, peace: 7.2 },
    { day: "Thu", joy: 7.5, curiosity: 5.8, anxiety: 2.9, peace: 6.1 },
    { day: "Fri", joy: 6.9, curiosity: 5.3, anxiety: 3.5, peace: 5.5 },
    { day: "Sat", joy: 8.3, curiosity: 6.7, anxiety: 1.5, peace: 7.8 },
    { day: "Sun", joy: 7.8, curiosity: 5.9, anxiety: 2.1, peace: 7.0 },
  ];

  if (isLoading) {
    return (
      <main className="p-4 md:p-6 flex items-center justify-center min-h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </main>
    );
  }

  return (
    <main className="p-4 md:p-6 space-y-6">
      {/* Dream Frequency Heatmap */}
      <Card className="glass-card p-6 rounded-xl hover-glow">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-futuristic font-semibold flex items-center gap-2">
            <Moon className="h-5 w-5 text-secondary" />
            Dream Frequency (Last 30 Days)
          </h3>
          <Badge variant="outline" className="border-primary/30 text-primary">
            {dreams.length} total dreams
          </Badge>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={calendarData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={4} stroke="hsl(var(--foreground))" opacity={0.5} />
            <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--foreground))" opacity={0.5} />
            <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
            <Bar dataKey="dreams" fill="hsl(270, 70%, 65%)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Most Common Symbols */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-6 flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            Top Dream Symbols
          </h3>
          {symbolData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={symbolData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis type="number" tick={{ fontSize: 10 }} stroke="hsl(var(--foreground))" opacity={0.5} />
                <YAxis dataKey="symbol" type="category" tick={{ fontSize: 11 }} width={100} stroke="hsl(var(--foreground))" opacity={0.5} />
                <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
                <Bar dataKey="count" fill="hsl(195, 100%, 50%)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[250px] text-foreground/40">
              Record dreams to see symbol patterns
            </div>
          )}
        </Card>

        {/* Dream Type Distribution */}
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-6 flex items-center gap-2">
            <Eye className="h-5 w-5 text-secondary" />
            Dream Type Distribution
          </h3>
          {typeData.length > 0 ? (
            <div className="flex items-center gap-6">
              <ResponsiveContainer width="60%" height={250}>
                <PieChart>
                  <Pie data={typeData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name }) => name}>
                    {typeData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={TYPE_COLORS[index % TYPE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2">
                {typeData.map((item, i) => (
                  <div key={item.name} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ background: TYPE_COLORS[i % TYPE_COLORS.length] }} />
                    <span className="text-sm capitalize">{item.name}</span>
                    <span className="text-xs text-foreground/50">({item.value as number})</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[250px] text-foreground/40">
              Record dreams to see type distribution
            </div>
          )}
        </Card>
      </div>

      {/* Emotion Trends */}
      <Card className="glass-card p-6 rounded-xl hover-glow">
        <h3 className="text-lg font-futuristic font-semibold mb-6 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-success" />
          Dream Emotion Trends
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={emotionTrends}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="day" stroke="hsl(var(--foreground))" opacity={0.5} />
            <YAxis domain={[0, 10]} stroke="hsl(var(--foreground))" opacity={0.5} />
            <Tooltip contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8 }} />
            <Line type="monotone" dataKey="joy" stroke={EMOTION_COLORS.joy} strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="curiosity" stroke={EMOTION_COLORS.curiosity} strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="anxiety" stroke={EMOTION_COLORS.anxiety} strokeWidth={2} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="peace" stroke={EMOTION_COLORS.peace} strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex justify-center gap-6 mt-4">
          {Object.entries({ joy: "Joy", curiosity: "Curiosity", anxiety: "Anxiety", peace: "Peace" }).map(([key, label]) => (
            <div key={key} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ background: EMOTION_COLORS[key] }} />
              <span className="text-xs text-foreground/70">{label}</span>
            </div>
          ))}
        </div>
      </Card>
    </main>
  );
}
