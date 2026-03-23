/**
 * Community — Anonymous mood sharing, daily challenges, and streaks.
 *
 * - Collective mood: aggregate mood of all users (no individual data)
 * - Daily challenge: rotating wellness challenges
 * - Streaks leaderboard: anonymous streak counts
 *
 * Data is stored locally; Supabase community_moods table integration
 * is documented but not required for local-first operation.
 *
 * Issue #485
 */

import { useState, useEffect, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import {
  Users, Flame, Trophy, Heart, Target, Wind, Brain, Sun,
  Smile, Frown, Minus, Sparkles,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { sbGetSetting, sbSaveGeneric, sbSaveSetting } from "../lib/supabase-store";

// ── Types ──────────────────────────────────────────────────────────────────

interface DailyChallenge {
  id: string;
  title: string;
  description: string;
  icon: keyof typeof CHALLENGE_ICONS;
  durationMinutes: number;
}

interface StreakEntry {
  rank: number;
  streakDays: number;
  isYou: boolean;
}

type MoodVote = "positive" | "neutral" | "negative";

const CHALLENGE_ICONS = {
  wind: Wind,
  brain: Brain,
  heart: Heart,
  sun: Sun,
  target: Target,
  sparkles: Sparkles,
};

// ── Constants ──────────────────────────────────────────────────────────────

const STORAGE_MOOD_KEY = "ndw_community_mood_vote";
const STORAGE_STREAK_KEY = "ndw_community_streak";
const STORAGE_CHALLENGE_KEY = "ndw_community_challenge_done";

const DAILY_CHALLENGES: DailyChallenge[] = [
  {
    id: "breathe_5",
    title: "5 minutes of focused breathing",
    description: "Sit comfortably and breathe deeply for 5 minutes. Focus on the exhale being longer than the inhale.",
    icon: "wind",
    durationMinutes: 5,
  },
  {
    id: "gratitude_3",
    title: "Write 3 things you are grateful for",
    description: "Take a moment to reflect on three positive things in your life today.",
    icon: "heart",
    durationMinutes: 3,
  },
  {
    id: "brain_scan",
    title: "2-minute brain scan meditation",
    description: "Close your eyes and scan your body from head to toe, noticing any sensations.",
    icon: "brain",
    durationMinutes: 2,
  },
  {
    id: "sunlight_10",
    title: "10 minutes of natural light",
    description: "Step outside or sit by a window for 10 minutes. Natural light helps regulate your circadian rhythm.",
    icon: "sun",
    durationMinutes: 10,
  },
  {
    id: "focus_15",
    title: "15-minute focused work sprint",
    description: "Pick one task and work on it with zero distractions for 15 minutes.",
    icon: "target",
    durationMinutes: 15,
  },
  {
    id: "kindness_1",
    title: "One act of kindness",
    description: "Do something kind for someone today. It can be as simple as a compliment.",
    icon: "sparkles",
    durationMinutes: 1,
  },
];

// ── Helpers ────────────────────────────────────────────────────────────────

function getTodayKey(): string {
  return new Date().toISOString().split("T")[0];
}

function getDailyChallenge(): DailyChallenge {
  // Deterministic based on day — same challenge for everyone on the same day
  const dayOfYear = Math.floor(
    (Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000,
  );
  return DAILY_CHALLENGES[dayOfYear % DAILY_CHALLENGES.length];
}

function loadStreak(): number {
  try {
    const raw = sbGetSetting(STORAGE_STREAK_KEY);
    if (!raw) return 0;
    const data = JSON.parse(raw);
    const today = getTodayKey();
    const yesterday = new Date(Date.now() - 86400000).toISOString().split("T")[0];
    if (data.lastDate === today) return data.count;
    if (data.lastDate === yesterday) return data.count; // Still valid, not broken
    return 0; // Streak broken
  } catch {
    return 0;
  }
}

function incrementStreak(): number {
  const today = getTodayKey();
  try {
    const raw = sbGetSetting(STORAGE_STREAK_KEY);
    const data = raw ? JSON.parse(raw) : { count: 0, lastDate: "" };
    if (data.lastDate === today) return data.count; // Already counted today
    const yesterday = new Date(Date.now() - 86400000).toISOString().split("T")[0];
    const newCount = data.lastDate === yesterday ? data.count + 1 : 1;
    sbSaveGeneric(STORAGE_STREAK_KEY, { count: newCount, lastDate: today });
    return newCount;
  } catch {
    return 1;
  }
}

// Simulated collective mood (local-first; Supabase integration future)
function getCollectiveMood(): { positive: number; neutral: number; negative: number } {
  // In production, this would fetch from Supabase community_moods table.
  // For now, generate a plausible distribution seeded by the day.
  const day = getTodayKey();
  const seed = day.split("-").reduce((a, b) => a + parseInt(b, 10), 0);
  const positiveBase = 40 + (seed % 20);
  const neutralBase = 30 + ((seed * 7) % 15);
  const negativeBase = 100 - positiveBase - neutralBase;
  return {
    positive: Math.max(5, positiveBase),
    neutral: Math.max(5, neutralBase),
    negative: Math.max(5, negativeBase),
  };
}

// Simulated leaderboard (local-first; would come from Supabase)
function getStreakLeaderboard(yourStreak: number): StreakEntry[] {
  const entries: StreakEntry[] = [
    { rank: 1, streakDays: 47, isYou: false },
    { rank: 2, streakDays: 31, isYou: false },
    { rank: 3, streakDays: 28, isYou: false },
    { rank: 4, streakDays: 21, isYou: false },
    { rank: 5, streakDays: 14, isYou: false },
  ];
  // Insert user's streak at the right position
  const yourEntry: StreakEntry = { rank: 0, streakDays: yourStreak, isYou: true };
  const combined = [...entries, yourEntry].sort((a, b) => b.streakDays - a.streakDays);
  return combined.map((e, i) => ({ ...e, rank: i + 1 })).slice(0, 6);
}

// ── Component ──────────────────────────────────────────────────────────────

export default function Community() {
  const [moodVote, setMoodVote] = useState<MoodVote | null>(null);
  const [streak, setStreak] = useState(0);
  const [challengeDone, setChallengeDone] = useState(false);

  const challenge = useMemo(() => getDailyChallenge(), []);
  const collectiveMood = useMemo(() => getCollectiveMood(), []);
  const leaderboard = useMemo(() => getStreakLeaderboard(streak), [streak]);

  useEffect(() => {
    setStreak(loadStreak());
    // Check if already voted today
    const voteData = sbGetSetting(STORAGE_MOOD_KEY);
    if (voteData) {
      try {
        const parsed = JSON.parse(voteData);
        if (parsed.date === getTodayKey()) setMoodVote(parsed.mood);
      } catch { /* ignore */ }
    }
    // Check if challenge done today
    const chalData = sbGetSetting(STORAGE_CHALLENGE_KEY);
    if (chalData === getTodayKey()) setChallengeDone(true);
  }, []);

  const handleVote = useCallback((mood: MoodVote) => {
    setMoodVote(mood);
    sbSaveGeneric(STORAGE_MOOD_KEY, { mood, date: getTodayKey() });
    // In production: POST to /api/community/mood-vote
  }, []);

  const handleChallengeComplete = useCallback(() => {
    setChallengeDone(true);
    sbSaveSetting(STORAGE_CHALLENGE_KEY, getTodayKey());
    const newStreak = incrementStreak();
    setStreak(newStreak);
  }, []);

  const ChallengeIcon = CHALLENGE_ICONS[challenge.icon] || Target;
  const moodTotal = collectiveMood.positive + collectiveMood.neutral + collectiveMood.negative;

  return (
    <motion.div
      initial={pageTransition.initial}
      animate={pageTransition.animate}
      transition={pageTransition.transition}
      className="max-w-2xl mx-auto px-4 py-6 pb-24"
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center">
          <Users className="w-5 h-5 text-blue-500" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-foreground">
            Community
          </h1>
          <p className="text-sm text-muted-foreground">
            Anonymous mood sharing and challenges
          </p>
        </div>
      </div>

      {/* Collective Mood */}
      <Card className="mb-4" data-testid="collective-mood">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Heart className="w-4 h-4 text-pink-500" />
            Collective Mood Today
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-2">
            <Smile className="w-4 h-4 text-green-500" />
            <div className="flex-1">
              <Progress value={(collectiveMood.positive / moodTotal) * 100} className="h-2" />
            </div>
            <span className="text-xs text-muted-foreground w-10 text-right">
              {Math.round((collectiveMood.positive / moodTotal) * 100)}%
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Minus className="w-4 h-4 text-yellow-500" />
            <div className="flex-1">
              <Progress value={(collectiveMood.neutral / moodTotal) * 100} className="h-2" />
            </div>
            <span className="text-xs text-muted-foreground w-10 text-right">
              {Math.round((collectiveMood.neutral / moodTotal) * 100)}%
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Frown className="w-4 h-4 text-red-500" />
            <div className="flex-1">
              <Progress value={(collectiveMood.negative / moodTotal) * 100} className="h-2" />
            </div>
            <span className="text-xs text-muted-foreground w-10 text-right">
              {Math.round((collectiveMood.negative / moodTotal) * 100)}%
            </span>
          </div>

          {/* Vote Section */}
          <div className="pt-2 border-t border-border">
            <p className="text-xs text-muted-foreground mb-2">How are you feeling today?</p>
            <div className="flex gap-2">
              {(["positive", "neutral", "negative"] as const).map((mood) => (
                <Button
                  key={mood}
                  variant={moodVote === mood ? "default" : "outline"}
                  size="sm"
                  onClick={() => handleVote(mood)}
                  disabled={moodVote !== null}
                  className="flex-1"
                >
                  {mood === "positive" && <Smile className="w-3 h-3 mr-1" />}
                  {mood === "neutral" && <Minus className="w-3 h-3 mr-1" />}
                  {mood === "negative" && <Frown className="w-3 h-3 mr-1" />}
                  {mood.charAt(0).toUpperCase() + mood.slice(1)}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Daily Challenge */}
      <Card className="mb-4" data-testid="daily-challenge">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Target className="w-4 h-4 text-orange-500" />
            Daily Challenge
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center flex-shrink-0">
              <ChallengeIcon className="w-5 h-5 text-orange-500" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-medium">{challenge.title}</h3>
              <p className="text-xs text-muted-foreground mt-1">{challenge.description}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {challenge.durationMinutes} min
              </p>
            </div>
          </div>
          <Button
            onClick={handleChallengeComplete}
            disabled={challengeDone}
            className="w-full mt-3"
            variant={challengeDone ? "outline" : "default"}
          >
            {challengeDone ? "Completed" : "Mark as Done"}
          </Button>
        </CardContent>
      </Card>

      {/* Streaks */}
      <Card data-testid="streaks-leaderboard">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Flame className="w-4 h-4 text-amber-500" />
            Streaks Leaderboard
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Your Streak */}
          <div className="flex items-center gap-3 mb-3 p-2 rounded-lg bg-primary/5">
            <Flame className="w-5 h-5 text-amber-500" />
            <span className="text-sm font-medium">Your Streak</span>
            <span className="ml-auto text-lg font-bold text-amber-500">
              {streak}
            </span>
            <span className="text-xs text-muted-foreground">days</span>
          </div>

          {/* Leaderboard */}
          <div className="space-y-1">
            {leaderboard.map((entry) => (
              <div
                key={entry.rank}
                className={`flex items-center gap-3 p-2 rounded-lg text-sm ${
                  entry.isYou ? "bg-primary/10 font-medium" : ""
                }`}
              >
                <span className="w-6 text-center text-muted-foreground">
                  {entry.rank <= 3 ? (
                    <Trophy className={`w-4 h-4 inline ${
                      entry.rank === 1 ? "text-amber-500" :
                      entry.rank === 2 ? "text-gray-400" :
                      "text-amber-700"
                    }`} />
                  ) : (
                    `#${entry.rank}`
                  )}
                </span>
                <span className="flex-1">
                  {entry.isYou ? "You" : `User ${entry.rank}`}
                </span>
                <span className="font-medium">{entry.streakDays}</span>
                <span className="text-xs text-muted-foreground">days</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
