import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Moon, Check, Sparkles } from "lucide-react";
import {
  saveIntention,
  getTodayIntention,
  getIntentionHistory,
  computeIntentionStats,
  alignmentLabel,
  alignmentDescription,
  ALIGNMENT_COLOR,
  ALIGNMENT_BG,
  type IntentionEntry,
} from "@/lib/sleep-intention";

const PLACEHOLDER_PROMPTS = [
  "I want to dream about resolving a difficult situation…",
  "I intend to become aware that I am dreaming…",
  "I want to explore a creative challenge in my dream…",
  "I hope to reconnect with someone important to me…",
  "I will notice recurring symbols and remember them…",
];

function randomPlaceholder() {
  return PLACEHOLDER_PROMPTS[Math.floor(Math.random() * PLACEHOLDER_PROMPTS.length)];
}

interface HistoryRowProps {
  entry: IntentionEntry;
}

function HistoryRow({ entry }: HistoryRowProps) {
  const d = new Date(entry.date);
  const dateStr = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const label = entry.alignmentScore !== null ? alignmentLabel(entry.alignmentScore) : null;

  return (
    <div className="flex items-start gap-2.5 py-1.5 border-b border-white/5 last:border-0">
      <span className="text-[10px] text-muted-foreground/50 shrink-0 w-12 pt-0.5">{dateStr}</span>
      <p className="text-xs text-muted-foreground flex-1 leading-relaxed">{entry.text}</p>
      {label ? (
        <span className={`text-[10px] px-1.5 py-0.5 rounded-full shrink-0 ${ALIGNMENT_BG[label]} ${ALIGNMENT_COLOR[label]}`}>
          {label}
        </span>
      ) : (
        <span className="text-[10px] text-muted-foreground/30 shrink-0">no dream</span>
      )}
    </div>
  );
}

export function PresleepIntentionCard() {
  const [text, setText] = useState("");
  const [saved, setSaved] = useState(false);
  const [todayEntry, setTodayEntry] = useState<IntentionEntry | null>(null);
  const [history, setHistory] = useState<IntentionEntry[]>([]);
  const [placeholder] = useState(randomPlaceholder);

  useEffect(() => {
    const today = getTodayIntention();
    setTodayEntry(today);
    if (today) setText(today.text);
    setHistory(getIntentionHistory());
  }, []);

  function handleSave() {
    if (!text.trim()) return;
    const entry = saveIntention(text);
    setTodayEntry(entry);
    setSaved(true);
    setHistory(getIntentionHistory());
    setTimeout(() => setSaved(false), 2000);
  }

  const stats = computeIntentionStats(history.filter((e) => e.alignmentScore !== null));
  const scoredHistory = history.filter((e) => e.alignmentScore !== null);
  const showStats = scoredHistory.length >= 2;

  return (
    <Card className="glass-card p-5 hover-glow border-secondary/20">
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Moon className="h-4 w-4 text-secondary" />
          Presleep Intention
          {showStats && stats.avgAlignment !== null && (
            <span className="ml-auto text-[10px] text-muted-foreground">
              avg alignment: {Math.round(stats.avgAlignment * 100)}%
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-3">
        {/* Intention input */}
        <div className="space-y-2">
          <p className="text-[11px] text-muted-foreground/70 leading-relaxed">
            Set an intention before sleep. After recording your dream in the morning,
            the app will score how closely the dream reflected it.
          </p>
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={placeholder}
            rows={3}
            className="text-xs bg-white/5 border-white/10 focus:border-secondary/40 resize-none"
          />
          <Button
            size="sm"
            variant="outline"
            className="w-full border-secondary/30 text-secondary hover:bg-secondary/10 text-xs"
            onClick={handleSave}
            disabled={!text.trim()}
          >
            {saved ? (
              <><Check className="h-3.5 w-3.5 mr-1.5" /> Saved</>
            ) : todayEntry ? (
              "Update tonight's intention"
            ) : (
              <><Sparkles className="h-3.5 w-3.5 mr-1.5" /> Set intention for tonight</>
            )}
          </Button>
        </div>

        {/* Today's alignment (if scored) */}
        {todayEntry?.alignmentScore !== null && todayEntry?.alignmentScore !== undefined && (() => {
          const lbl = alignmentLabel(todayEntry.alignmentScore!);
          return (
            <div className={`rounded-lg px-3 py-2 ${ALIGNMENT_BG[lbl]} border border-white/10`}>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-medium ${ALIGNMENT_COLOR[lbl]}`}>
                  {lbl.charAt(0).toUpperCase() + lbl.slice(1)} alignment
                </span>
                <span className={`ml-auto text-[10px] ${ALIGNMENT_COLOR[lbl]}`}>
                  {Math.round(todayEntry.alignmentScore! * 100)}%
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground/70 mt-0.5">
                {alignmentDescription(lbl)}
              </p>
            </div>
          );
        })()}

        {/* History */}
        {history.length > 1 && (
          <div>
            <p className="text-[10px] text-muted-foreground/50 mb-1.5">Past intentions</p>
            <div className="max-h-[200px] overflow-y-auto">
              {history.slice(1, 8).map((e) => (
                <HistoryRow key={e.date} entry={e} />
              ))}
            </div>
          </div>
        )}

        {/* Stats bar */}
        {showStats && (
          <div className="grid grid-cols-3 gap-2 pt-1 text-center border-t border-white/5">
            <div>
              <p className="text-xs text-secondary">{stats.totalScored}</p>
              <p className="text-[10px] text-muted-foreground/50">scored</p>
            </div>
            <div>
              <p className="text-xs text-emerald-400">
                {Math.round(stats.strongRate * 100)}%
              </p>
              <p className="text-[10px] text-muted-foreground/50">strong match</p>
            </div>
            <div>
              <p className="text-xs text-secondary">
                {stats.avgAlignment !== null ? Math.round(stats.avgAlignment * 100) + "%" : "—"}
              </p>
              <p className="text-[10px] text-muted-foreground/50">avg score</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
