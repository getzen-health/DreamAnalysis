import { useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import {
  logSupplement,
  getSupplementLog,
  getSupplementReport,
  getActiveSupplements,
  type SupplementLogEntry,
  type ActiveSupplement,
  type SupplementVerdict,
} from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { Pill, TrendingUp, Clock, Plus, Activity } from "lucide-react";

const SUPPLEMENT_TYPES = ["vitamin", "supplement", "medication", "food_supplement"] as const;
const COMMON_UNITS = ["mg", "IU", "mcg", "g", "ml", "capsules", "tablets"] as const;

const VERDICT_STYLE: Record<string, string> = {
  positive: "border-cyan-500/40 text-cyan-400 bg-cyan-600/10",
  negative: "border-rose-500/40 text-rose-400 bg-rose-500/10",
  neutral: "border-zinc-500/40 text-zinc-400 bg-zinc-500/10",
  insufficient_data: "border-amber-500/40 text-amber-400 bg-amber-500/10",
};

const VERDICT_LABEL: Record<string, string> = {
  positive: "Helps",
  negative: "Hurts",
  neutral: "Neutral",
  insufficient_data: "Need more data",
};

function ShiftBar({ value, label }: { value: number; label: string }) {
  const pct = Math.min(Math.abs(value) * 100, 100);
  const positive = value >= 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 text-zinc-400 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${positive ? "bg-cyan-600" : "bg-rose-500"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`w-12 text-right tabular-nums ${positive ? "text-cyan-400" : "text-rose-400"}`}>
        {positive ? "+" : ""}{(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function VerdictCard({ s }: { s: SupplementVerdict }) {
  return (
    <Card className="bg-zinc-900/60 border-zinc-800">
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-medium text-white">{s.name}</p>
            <p className="text-xs text-zinc-500">{s.n_exposures} session{s.n_exposures !== 1 ? "s" : ""} tracked</p>
          </div>
          <Badge className={VERDICT_STYLE[s.verdict]}>{VERDICT_LABEL[s.verdict]}</Badge>
        </div>
        {s.n_exposures > 0 && (
          <div className="space-y-1.5">
            <ShiftBar value={s.valence_shift} label="Mood" />
            <ShiftBar value={-s.stress_shift} label="Stress ↓" />
            <ShiftBar value={s.focus_shift} label="Focus" />
            <ShiftBar value={s.alpha_beta_shift} label="Calm α/β" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ActiveCard({ s }: { s: ActiveSupplement }) {
  const hrs = s.hours_ago;
  return (
    <div className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0">
      <div>
        <p className="text-sm font-medium text-white">{s.name}</p>
        <p className="text-xs text-zinc-500">{s.dosage} {s.unit} · {s.type.replace("_", " ")}</p>
      </div>
      <span className="text-xs text-zinc-400">
        {hrs < 1 ? `${Math.round(hrs * 60)}m ago` : `${hrs.toFixed(1)}h ago`}
      </span>
    </div>
  );
}

function LogEntry({ e }: { e: SupplementLogEntry }) {
  const dt = new Date(e.timestamp * 1000);
  return (
    <div className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0">
      <div>
        <p className="text-sm font-medium text-white">{e.name}</p>
        <p className="text-xs text-zinc-500">
          {e.dosage} {e.unit} · {e.type.replace("_", " ")}
          {e.notes ? ` · ${e.notes}` : ""}
        </p>
      </div>
      <span className="text-xs text-zinc-400 shrink-0">
        {dt.toLocaleDateString()} {dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
      </span>
    </div>
  );
}

export default function SupplementsPage() {
  const userId = getParticipantId();
  const qc = useQueryClient();
  const { toast } = useToast();

  const [name, setName] = useState("");
  const [type, setType] = useState<string>("supplement");
  const [dosage, setDosage] = useState("");
  const [unit, setUnit] = useState<string>("mg");
  const [notes, setNotes] = useState("");

  const logQ = useQuery({
    queryKey: ["supplement-log", userId],
    queryFn: () => getSupplementLog(userId, 50),
    refetchInterval: 30_000,
  });

  const reportQ = useQuery({
    queryKey: ["supplement-report", userId],
    queryFn: () => getSupplementReport(userId),
    refetchInterval: 60_000,
  });

  const activeQ = useQuery({
    queryKey: ["supplement-active", userId],
    queryFn: () => getActiveSupplements(userId, 24),
    refetchInterval: 30_000,
  });

  const logMut = useMutation({
    mutationFn: logSupplement,
    onSuccess: () => {
      toast({ title: "Logged", description: `${name} recorded.` });
      qc.invalidateQueries({ queryKey: ["supplement-log", userId] });
      qc.invalidateQueries({ queryKey: ["supplement-active", userId] });
      qc.invalidateQueries({ queryKey: ["supplement-report", userId] });
      setName("");
      setDosage("");
      setNotes("");
    },
    onError: (e: Error) => {
      toast({ title: "Error", description: e.message, variant: "destructive" });
    },
  });

  const handleLog = () => {
    if (!name.trim() || !dosage) return;
    logMut.mutate({
      user_id: userId,
      name: name.trim(),
      type,
      dosage: parseFloat(dosage),
      unit,
      notes: notes.trim() || undefined,
    });
  };

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-4">
      <div className="flex items-center gap-2 mb-2">
        <Pill className="h-5 w-5 text-purple-400" />
        <h1 className="text-xl font-semibold text-white">Supplement Tracker</h1>
      </div>
      <p className="text-sm text-zinc-400">
        Log vitamins, supplements &amp; medications to see how they affect your brain states.
      </p>

      <Tabs defaultValue="log">
        <TabsList className="w-full bg-zinc-900 border border-zinc-800">
          <TabsTrigger value="log" className="flex-1 gap-1.5">
            <Plus className="h-3.5 w-3.5" /> Log
          </TabsTrigger>
          <TabsTrigger value="active" className="flex-1 gap-1.5">
            <Clock className="h-3.5 w-3.5" /> Active
          </TabsTrigger>
          <TabsTrigger value="insights" className="flex-1 gap-1.5">
            <TrendingUp className="h-3.5 w-3.5" /> Insights
          </TabsTrigger>
        </TabsList>

        {/* ── Log Tab ── */}
        <TabsContent value="log" className="space-y-4 mt-4">
          <Card className="bg-zinc-900/60 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">Quick Add</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                placeholder="Name (e.g. Vitamin D, Omega-3)"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="bg-zinc-800 border-zinc-700 text-white"
              />
              <div className="grid grid-cols-2 gap-2">
                <Select value={type} onValueChange={setType}>
                  <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-zinc-900 border-zinc-700">
                    {SUPPLEMENT_TYPES.map((t) => (
                      <SelectItem key={t} value={t} className="text-zinc-200">
                        {t.replace("_", " ")}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="flex gap-1.5">
                  <Input
                    placeholder="Dosage"
                    type="number"
                    value={dosage}
                    onChange={(e) => setDosage(e.target.value)}
                    className="bg-zinc-800 border-zinc-700 text-white flex-1"
                  />
                  <Select value={unit} onValueChange={setUnit}>
                    <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white w-20">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-zinc-900 border-zinc-700">
                      {COMMON_UNITS.map((u) => (
                        <SelectItem key={u} value={u} className="text-zinc-200">{u}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <Input
                placeholder="Notes (optional)"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="bg-zinc-800 border-zinc-700 text-white"
              />
              <Button
                onClick={handleLog}
                disabled={!name.trim() || !dosage || logMut.isPending}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white"
              >
                {logMut.isPending ? "Logging…" : "Log Intake"}
              </Button>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900/60 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300 flex items-center gap-2">
                <Activity className="h-4 w-4" /> Recent Log
              </CardTitle>
            </CardHeader>
            <CardContent>
              {logQ.isLoading && (
                <div className="space-y-2">
                  {[1,2,3].map(i => <Skeleton key={i} className="h-9 w-full rounded-lg" />)}
                </div>
              )}
              {logQ.data?.entries.length === 0 && (
                <p className="text-xs text-zinc-500">No supplements logged yet.</p>
              )}
              {logQ.data?.entries.map((e) => <LogEntry key={e.id} e={e} />)}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Active Tab ── */}
        <TabsContent value="active" className="mt-4">
          <Card className="bg-zinc-900/60 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">
                Currently Active (last 24h)
              </CardTitle>
            </CardHeader>
            <CardContent>
              {activeQ.isLoading && (
                <div className="space-y-2">
                  {[1,2].map(i => <Skeleton key={i} className="h-9 w-full rounded-lg" />)}
                </div>
              )}
              {activeQ.data?.supplements.length === 0 && (
                <p className="text-xs text-zinc-500">Nothing active in the last 24 hours.</p>
              )}
              {activeQ.data?.supplements.map((s, i) => (
                <ActiveCard key={i} s={s} />
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Insights Tab ── */}
        <TabsContent value="insights" className="mt-4 space-y-3">
          {reportQ.isLoading && (
            <div className="space-y-2">
              {[1,2].map(i => <Skeleton key={i} className="h-20 w-full rounded-xl" />)}
            </div>
          )}
          {reportQ.data?.supplements.length === 0 && (
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardContent className="p-4 text-center text-sm text-zinc-500">
                Log supplements for 3+ sessions to see correlation insights.
              </CardContent>
            </Card>
          )}
          {reportQ.data?.supplements.map((s) => (
            <VerdictCard key={s.name} s={s} />
          ))}
        </TabsContent>
      </Tabs>
    </div>
  );
}
