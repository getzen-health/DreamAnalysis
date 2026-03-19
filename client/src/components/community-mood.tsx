/**
 * Community Mood — anonymous mood sharing + peer mood feed.
 * Opt-in: shows after voice check-in or on Discover page.
 * No personal data — just emotion label + timestamp.
 */
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { resolveUrl } from "@/lib/queryClient";
import { Users } from "lucide-react";

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊", sad: "😢", angry: "😠", fear: "😰", surprise: "😮", neutral: "😐",
};

interface MoodFeed {
  counts: Record<string, number>;
  total: number;
}

export function CommunityMood() {
  const [shared, setShared] = useState(false);
  const queryClient = useQueryClient();

  const { data: feed } = useQuery<MoodFeed>({
    queryKey: ["community-mood-feed"],
    queryFn: () => fetch(resolveUrl("/api/community/mood-feed")).then(r => r.json()),
    staleTime: 60 * 1000,
    retry: false,
  });

  const shareMutation = useMutation({
    mutationFn: (emotion: string) =>
      fetch(resolveUrl("/api/community/share-mood"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ emotion }),
      }).then(r => r.json()),
    onSuccess: () => {
      setShared(true);
      queryClient.invalidateQueries({ queryKey: ["community-mood-feed"] });
    },
  });

  // Check if user already shared today
  useEffect(() => {
    const today = new Date().toISOString().slice(0, 10);
    if (localStorage.getItem(`ndw_community_shared_${today}`)) {
      setShared(true);
    }
  }, []);

  function handleShare() {
    try {
      const raw = localStorage.getItem("ndw_last_emotion");
      if (!raw) return;
      const emotion = JSON.parse(raw)?.result?.emotion;
      if (!emotion) return;
      shareMutation.mutate(emotion);
      const today = new Date().toISOString().slice(0, 10);
      localStorage.setItem(`ndw_community_shared_${today}`, "true");
    } catch { /* ignore */ }
  }

  const topEmotion = feed?.counts
    ? Object.entries(feed.counts).sort((a, b) => b[1] - a[1])[0]
    : null;

  return (
    <div style={{
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: 14, padding: "12px 14px", marginBottom: 12,
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 6, marginBottom: 8,
      }}>
        <Users style={{ width: 14, height: 14, color: "hsl(200 70% 55%)" }} />
        <span style={{
          fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
          textTransform: "uppercase" as const, letterSpacing: "0.5px",
        }}>
          Community
        </span>
      </div>

      {/* Mood distribution */}
      {feed && feed.total > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" as const }}>
            {Object.entries(feed.counts)
              .sort((a, b) => b[1] - a[1])
              .map(([emotion, count]) => {
                const pct = Math.round((count / feed.total) * 100);
                return (
                  <div key={emotion} style={{
                    background: "var(--muted)", borderRadius: 8, padding: "4px 8px",
                    display: "flex", alignItems: "center", gap: 4,
                  }}>
                    <span style={{ fontSize: 14 }}>{EMOTION_EMOJI[emotion] || "😐"}</span>
                    <span style={{ fontSize: 10, fontWeight: 600, color: "var(--foreground)" }}>{pct}%</span>
                  </div>
                );
              })}
          </div>
          {topEmotion && (
            <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 6 }}>
              {feed.total} people checked in today — most are feeling {topEmotion[0]}
            </div>
          )}
        </div>
      )}

      {/* Share button */}
      {!shared ? (
        <button
          onClick={handleShare}
          style={{
            width: "100%", padding: "8px 12px", borderRadius: 8,
            background: "hsl(200 70% 55% / 0.12)", color: "hsl(200 70% 65%)",
            border: "1px solid hsl(200 70% 55% / 0.25)",
            fontSize: 11, fontWeight: 600, cursor: "pointer",
          }}
        >
          Share how you're feeling anonymously
        </button>
      ) : (
        <div style={{
          fontSize: 11, color: "var(--muted-foreground)", textAlign: "center" as const,
          padding: "6px 0",
        }}>
          You shared your mood today — you're not alone
        </div>
      )}
    </div>
  );
}
