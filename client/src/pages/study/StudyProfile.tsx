import { useState } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, Loader2, Zap, Brain } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

type DietType = "omnivore" | "vegetarian" | "vegan" | "other";
type SessionType = "stress" | "food";

export default function StudyProfile() {
  const [, navigate] = useLocation();
  const search = useSearch();
  const { toast } = useToast();

  const params = new URLSearchParams(search);
  const participantCode = params.get("code") ?? "";

  const [age, setAge] = useState<string>("");
  const [dietType, setDietType] = useState<DietType>("omnivore");
  const [hasAppleWatch, setHasAppleWatch] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const ageNum = parseInt(age, 10);
  const isAgeValid = !isNaN(ageNum) && ageNum >= 18 && ageNum <= 99;
  const canSubmit = isAgeValid && dietType.length > 0;

  async function handleStartSession(sessionType: SessionType) {
    if (!canSubmit) return;
    setIsSubmitting(true);

    try {
      await apiRequest("POST", "/api/study/consent", {
        participant_code: participantCode,
        consent_text: `Profile updated on ${new Date().toISOString()}`,
        age: ageNum,
        diet_type: dietType,
        has_apple_watch: hasAppleWatch,
      });

      navigate(
        `/study/session?code=${encodeURIComponent(participantCode)}&block=${sessionType}`
      );
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Could not save profile — please try again";
      toast({
        title: "Profile save failed",
        description: msg,
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-lg mx-auto px-4 py-10 space-y-6">

        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-2xl font-bold">Your Profile</h1>
          {participantCode && (
            <Badge variant="outline" className="border-primary/40 text-primary font-mono">
              {participantCode}
            </Badge>
          )}
          <p className="text-sm text-muted-foreground">
            We collect basic demographic information once to help contextualize your EEG data.
            This is not linked to your identity.
          </p>
        </div>

        {/* Demographics form */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Demographics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">

            {/* Age */}
            <div className="space-y-1.5">
              <Label htmlFor="age" className="text-sm">Age</Label>
              <Input
                id="age"
                type="number"
                inputMode="numeric"
                min={18}
                max={99}
                placeholder="e.g. 25"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="w-28"
              />
              {age.length > 0 && !isAgeValid && (
                <div className="flex items-center gap-1.5">
                  <AlertCircle className="h-3.5 w-3.5 text-destructive" />
                  <p className="text-xs text-destructive">Must be between 18 and 99</p>
                </div>
              )}
            </div>

            {/* Diet type */}
            <div className="space-y-1.5">
              <Label htmlFor="diet-type" className="text-sm">Diet type</Label>
              <Select
                value={dietType}
                onValueChange={(val) => setDietType(val as DietType)}
              >
                <SelectTrigger id="diet-type" className="w-48">
                  <SelectValue placeholder="Select diet" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="omnivore">Omnivore</SelectItem>
                  <SelectItem value="vegetarian">Vegetarian</SelectItem>
                  <SelectItem value="vegan">Vegan</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Apple Watch toggle */}
            <div className="flex items-center justify-between py-1">
              <div className="space-y-0.5">
                <Label htmlFor="apple-watch" className="text-sm">Apple Watch</Label>
                <p className="text-xs text-muted-foreground">
                  Do you own and regularly wear an Apple Watch?
                </p>
              </div>
              <Switch
                id="apple-watch"
                checked={hasAppleWatch}
                onCheckedChange={setHasAppleWatch}
              />
            </div>
          </CardContent>
        </Card>

        {/* Session buttons */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Choose a session to start</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              You can complete the two sessions in any order, on different days if you prefer.
            </p>

            <Button
              className="w-full"
              size="lg"
              disabled={!canSubmit || isSubmitting}
              onClick={() => handleStartSession("stress")}
            >
              {isSubmitting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Zap className="mr-2 h-4 w-4" />
              )}
              Start Stress Session
              <span className="ml-auto text-xs opacity-70">25 min</span>
            </Button>

            <Button
              className="w-full"
              size="lg"
              variant="outline"
              disabled={!canSubmit || isSubmitting}
              onClick={() => handleStartSession("food")}
            >
              {isSubmitting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Brain className="mr-2 h-4 w-4" />
              )}
              Start Food Session
              <span className="ml-auto text-xs opacity-70">40 min</span>
            </Button>

            {!canSubmit && age.length > 0 && (
              <div className="flex items-start gap-2 pt-1">
                <AlertCircle className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
                <p className="text-xs text-muted-foreground">
                  Enter a valid age to enable session start.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
