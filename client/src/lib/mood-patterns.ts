/**
 * Mood Pattern Detection — analyzes emotion history to find actionable patterns.
 *
 * Runs client-side on the emotion reading data from /api/brain/history.
 * Generates human-readable insights about emotional patterns.
 */

export interface EmotionReading {
  dominantEmotion: string;
  stress?: number;
  focus?: number;
  valence?: number;
  arousal?: number;
  timestamp: string;
}

export interface MoodInsight {
  /** Icon identifier — mapped to a lucide-react icon in the rendering component */
  icon: string;
  title: string;
  description: string;
  type: "positive" | "warning" | "neutral";
}

/**
 * Analyze emotion readings and return up to 3 actionable insights.
 */
/**
 * Predict tomorrow's likely emotional state based on recent patterns.
 * Uses weighted average: recent days count more + same-day-last-week bonus.
 */
export function forecastMood(readings: EmotionReading[]): {
  emotion: string;
  confidence: number;
  reason: string;
} | null {
  if (readings.length < 3) return null;

  // Weight recent readings more (exponential decay)
  const emotionScores: Record<string, number> = {};
  const n = readings.length;
  for (let i = 0; i < n; i++) {
    const weight = Math.pow(0.85, n - i - 1); // most recent = weight 1.0
    const emo = readings[i].dominantEmotion;
    emotionScores[emo] = (emotionScores[emo] || 0) + weight;
  }

  // Bonus for same-day-last-week emotion
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const targetDay = tomorrow.getDay();
  const sameDayReadings = readings.filter(r => new Date(r.timestamp).getDay() === targetDay);
  for (const r of sameDayReadings) {
    emotionScores[r.dominantEmotion] = (emotionScores[r.dominantEmotion] || 0) + 0.5;
  }

  // Pick top emotion
  const sorted = Object.entries(emotionScores).sort((a, b) => b[1] - a[1]);
  if (sorted.length === 0) return null;

  const topEmo = sorted[0][0];
  const totalWeight = sorted.reduce((s, [, w]) => s + w, 0);
  const confidence = Math.min(0.9, sorted[0][1] / totalWeight);

  // Generate reason
  const recentCount = readings.slice(-3).filter(r => r.dominantEmotion === topEmo).length;
  let reason: string;
  if (recentCount >= 2) {
    reason = `You've been ${topEmo} recently — likely to continue`;
  } else if (sameDayReadings.length > 0 && sameDayReadings[sameDayReadings.length - 1].dominantEmotion === topEmo) {
    reason = `You usually feel this way on ${["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][targetDay]}s`;
  } else {
    reason = `Based on your overall emotional pattern`;
  }

  return { emotion: topEmo, confidence, reason };
}

/**
 * Analyze emotion readings and return up to 3 actionable insights.
 */
export function detectMoodPatterns(readings: EmotionReading[]): MoodInsight[] {
  if (readings.length < 3) return [];

  const insights: MoodInsight[] = [];

  // 1. Most frequent emotion
  const emotionCounts: Record<string, number> = {};
  for (const r of readings) {
    emotionCounts[r.dominantEmotion] = (emotionCounts[r.dominantEmotion] || 0) + 1;
  }
  const sorted = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]);
  const topEmotion = sorted[0]?.[0];
  const topPct = Math.round((sorted[0]?.[1] / readings.length) * 100);

  if (topEmotion && topPct > 40) {
    if (topEmotion === "happy") {
      insights.push({
        icon: "star", title: "Consistently positive",
        description: `You've been happy ${topPct}% of the time — that's great emotional resilience.`,
        type: "positive",
      });
    } else if (topEmotion === "sad" || topEmotion === "fear") {
      insights.push({
        icon: "heart", title: "Persistent low mood",
        description: `${topEmotion === "sad" ? "Sadness" : "Anxiety"} appeared in ${topPct}% of check-ins. Consider talking to someone you trust.`,
        type: "warning",
      });
    } else if (topEmotion === "angry") {
      insights.push({
        icon: "waves", title: "Frequent frustration",
        description: `Anger showed up ${topPct}% of the time. Physical activity or journaling may help channel this energy.`,
        type: "warning",
      });
    }
  }

  // 2. Stress trend
  const stressReadings = readings.filter(r => r.stress != null).map(r => r.stress!);
  if (stressReadings.length >= 5) {
    const firstHalf = stressReadings.slice(0, Math.floor(stressReadings.length / 2));
    const secondHalf = stressReadings.slice(Math.floor(stressReadings.length / 2));
    const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    if (avgSecond < avgFirst - 0.05) {
      insights.push({
        icon: "trending-down", title: "Stress is decreasing",
        description: `Your stress dropped from ${Math.round(avgFirst * 100)}% to ${Math.round(avgSecond * 100)}%. Whatever you're doing is working.`,
        type: "positive",
      });
    } else if (avgSecond > avgFirst + 0.05) {
      insights.push({
        icon: "trending-up", title: "Stress is climbing",
        description: `Stress went from ${Math.round(avgFirst * 100)}% to ${Math.round(avgSecond * 100)}%. Try adding a daily breathing session.`,
        type: "warning",
      });
    }
  }

  // 3. Time-of-day patterns
  const morningReadings = readings.filter(r => {
    const h = new Date(r.timestamp).getHours();
    return h >= 5 && h < 12;
  });
  const eveningReadings = readings.filter(r => {
    const h = new Date(r.timestamp).getHours();
    return h >= 17 && h < 23;
  });

  if (morningReadings.length >= 3 && eveningReadings.length >= 3) {
    const morningValence = morningReadings.reduce((s, r) => s + (r.valence ?? 0), 0) / morningReadings.length;
    const eveningValence = eveningReadings.reduce((s, r) => s + (r.valence ?? 0), 0) / eveningReadings.length;

    if (morningValence > eveningValence + 0.15) {
      insights.push({
        icon: "sunrise", title: "Morning person detected",
        description: "Your mood is consistently better in the morning. Schedule important tasks early.",
        type: "neutral",
      });
    } else if (eveningValence > morningValence + 0.15) {
      insights.push({
        icon: "cloud-moon", title: "Night owl energy",
        description: "You feel better in the evenings. Your creative window may be later in the day.",
        type: "neutral",
      });
    }
  }

  // 4. Emotion diversity
  const uniqueEmotions = Object.keys(emotionCounts).length;
  if (uniqueEmotions >= 4 && readings.length >= 7) {
    insights.push({
      icon: "palette", title: "Emotionally diverse",
      description: `You've experienced ${uniqueEmotions} different emotions this week. That's healthy emotional range.`,
      type: "positive",
    });
  } else if (uniqueEmotions <= 1 && readings.length >= 5) {
    insights.push({
      icon: "refresh", title: "Emotional flatness",
      description: "You've been in the same emotional state consistently. Try new activities to shift your state.",
      type: "neutral",
    });
  }

  return insights.slice(0, 3);
}
