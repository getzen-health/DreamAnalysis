import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  Moon, Mic, MicOff, Send, Star, Brain, Eye, Repeat, Zap, Clock, Loader2, ChevronDown, ImageIcon
} from "lucide-react";

const DREAM_TAGS = [
  { id: "lucid", label: "Lucid", icon: Eye },
  { id: "nightmare", label: "Nightmare", icon: Zap },
  { id: "recurring", label: "Recurring", icon: Repeat },
  { id: "vivid", label: "Vivid", icon: Star },
];

export default function DreamJournal() {
  const [dreamText, setDreamText] = useState("");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [sleepQuality, setSleepQuality] = useState(3);
  const [sleepDuration, setSleepDuration] = useState("7.5");
  const [isRecording, setIsRecording] = useState(false);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // Use auth context userId or fallback
  const userId = "demo-user";

  const { data: dreams = [], isLoading: dreamsLoading } = useQuery({
    queryKey: ["/api/dream-analysis", userId],
    queryFn: async () => {
      const res = await fetch(`/api/dream-analysis/${userId}`);
      if (!res.ok) return [];
      return res.json();
    },
  });

  const analyzeMutation = useMutation({
    mutationFn: async (data: { dreamText: string; tags: string[]; sleepQuality: number; sleepDuration: number }) => {
      const res = await apiRequest("POST", "/api/dream-analysis", {
        dreamText: data.dreamText,
        userId,
        tags: data.tags,
        sleepQuality: data.sleepQuality,
        sleepDuration: data.sleepDuration,
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/dream-analysis", userId] });
      setDreamText("");
      setSelectedTags([]);
      setSleepQuality(3);
      toast({ title: "Dream Recorded", description: "Your dream has been analyzed by AI." });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to analyze dream.", variant: "destructive" });
    },
  });

  const handleSubmit = () => {
    if (!dreamText.trim()) return;
    analyzeMutation.mutate({
      dreamText,
      tags: selectedTags,
      sleepQuality,
      sleepDuration: parseFloat(sleepDuration) || 7.5,
    });
  };

  const toggleTag = (tagId: string) => {
    setSelectedTags(prev =>
      prev.includes(tagId) ? prev.filter(t => t !== tagId) : [...prev, tagId]
    );
  };

  const handleVoiceInput = () => {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      toast({ title: "Not Supported", description: "Voice input is not supported in this browser.", variant: "destructive" });
      return;
    }
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;

    if (isRecording) {
      recognition.stop();
      setIsRecording(false);
      return;
    }

    recognition.onresult = (event: any) => {
      let transcript = "";
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setDreamText(prev => prev + " " + transcript);
    };
    recognition.onerror = () => setIsRecording(false);
    recognition.onend = () => setIsRecording(false);
    recognition.start();
    setIsRecording(true);
  };

  return (
    <main className="p-4 md:p-6 space-y-6">
      <Tabs defaultValue="record" className="w-full">
        <TabsList className="grid w-full grid-cols-2 bg-card/50">
          <TabsTrigger value="record">Record Dream</TabsTrigger>
          <TabsTrigger value="history">Dream History ({dreams.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="record" className="space-y-6 mt-6">
          {/* Dream Entry Form */}
          <Card className="glass-card p-6 rounded-xl hover-glow">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-futuristic font-semibold flex items-center gap-2">
                <Moon className="h-5 w-5 text-secondary" />
                Record Your Dream
              </h3>
              <Button
                variant="outline"
                size="sm"
                onClick={handleVoiceInput}
                className={`${isRecording ? "bg-destructive/20 border-destructive/30 text-destructive" : "bg-secondary/10 border-secondary/30 text-secondary"}`}
              >
                {isRecording ? <MicOff className="h-4 w-4 mr-2" /> : <Mic className="h-4 w-4 mr-2" />}
                {isRecording ? "Stop Recording" : "Voice Input"}
              </Button>
            </div>

            <Textarea
              value={dreamText}
              onChange={(e) => setDreamText(e.target.value)}
              placeholder="Describe your dream in as much detail as you can remember... What did you see? How did you feel? What happened?"
              className="min-h-[200px] bg-card/50 border-primary/20 mb-4"
            />

            {/* Dream Tags */}
            <div className="mb-4">
              <Label className="text-sm font-medium text-foreground/80 mb-2 block">Dream Type</Label>
              <div className="flex flex-wrap gap-2">
                {DREAM_TAGS.map(tag => {
                  const Icon = tag.icon;
                  const isSelected = selectedTags.includes(tag.id);
                  return (
                    <Button
                      key={tag.id}
                      variant="outline"
                      size="sm"
                      onClick={() => toggleTag(tag.id)}
                      className={`${isSelected ? "bg-primary/20 border-primary/50 text-primary" : "bg-card/50 border-primary/10"}`}
                    >
                      <Icon className="h-3 w-3 mr-1" />
                      {tag.label}
                    </Button>
                  );
                })}
              </div>
            </div>

            {/* Sleep Info */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div>
                <Label className="text-sm font-medium text-foreground/80 mb-2 block">Sleep Quality</Label>
                <div className="flex items-center gap-1">
                  {[1, 2, 3, 4, 5].map(star => (
                    <button
                      key={star}
                      onClick={() => setSleepQuality(star)}
                      className="p-1"
                    >
                      <Star
                        className={`h-6 w-6 transition-colors ${star <= sleepQuality ? "text-secondary fill-secondary" : "text-foreground/20"}`}
                      />
                    </button>
                  ))}
                  <span className="text-sm text-foreground/60 ml-2">{sleepQuality}/5</span>
                </div>
              </div>
              <div>
                <Label className="text-sm font-medium text-foreground/80 mb-2 block">Sleep Duration (hours)</Label>
                <Input
                  type="number"
                  step="0.5"
                  min="0"
                  max="24"
                  value={sleepDuration}
                  onChange={(e) => setSleepDuration(e.target.value)}
                  className="bg-card/50 border-primary/20"
                />
              </div>
            </div>

            <Button
              onClick={handleSubmit}
              disabled={!dreamText.trim() || analyzeMutation.isPending}
              className="w-full bg-gradient-to-r from-primary to-secondary text-primary-foreground hover-glow"
            >
              {analyzeMutation.isPending ? (
                <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Analyzing Dream...</>
              ) : (
                <><Brain className="h-4 w-4 mr-2" /> Analyze Dream with AI</>
              )}
            </Button>
          </Card>
        </TabsContent>

        <TabsContent value="history" className="space-y-4 mt-6">
          {dreamsLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : dreams.length === 0 ? (
            <Card className="glass-card p-12 rounded-xl text-center">
              <Moon className="h-12 w-12 mx-auto text-secondary/50 mb-4" />
              <h3 className="text-lg font-futuristic font-semibold mb-2">No Dreams Yet</h3>
              <p className="text-foreground/60">Record your first dream to get started with AI analysis.</p>
            </Card>
          ) : (
            dreams.map((dream: any) => (
              <Card key={dream.id} className="glass-card p-6 rounded-xl hover-glow">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Clock className="h-4 w-4 text-foreground/50" />
                      <span className="text-sm text-foreground/60">
                        {new Date(dream.timestamp).toLocaleDateString("en-US", {
                          weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
                        })}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {(dream.tags as string[] || []).map((tag: string) => (
                        <Badge key={tag} variant="outline" className="border-secondary/30 text-secondary text-xs">
                          {tag}
                        </Badge>
                      ))}
                      {dream.sleepQuality && (
                        <Badge variant="outline" className="border-primary/30 text-primary text-xs">
                          {Array(dream.sleepQuality).fill("★").join("")}
                        </Badge>
                      )}
                    </div>
                  </div>
                  {dream.imageUrl && (
                    <div className="w-16 h-16 rounded-lg overflow-hidden bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
                      <ImageIcon className="h-6 w-6 text-primary/50" />
                    </div>
                  )}
                </div>

                <p className="text-sm text-foreground/80 mb-3 line-clamp-3">{dream.dreamText}</p>

                {dream.symbols && (dream.symbols as string[]).length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mb-3">
                    {(dream.symbols as string[]).map((symbol: string, i: number) => (
                      <Badge key={i} variant="outline" className="border-primary/20 text-primary text-xs">
                        {symbol}
                      </Badge>
                    ))}
                  </div>
                )}

                {dream.aiAnalysis && (
                  <div className="bg-card/30 rounded-lg p-3 border border-primary/10">
                    <div className="flex items-center gap-2 mb-1">
                      <Brain className="h-3 w-3 text-primary" />
                      <span className="text-xs font-semibold text-primary">AI Interpretation</span>
                    </div>
                    <p className="text-xs text-foreground/70 line-clamp-3">{dream.aiAnalysis}</p>
                  </div>
                )}
              </Card>
            ))
          )}
        </TabsContent>
      </Tabs>
    </main>
  );
}
