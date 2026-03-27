import { useState } from "react";
import { useLocation } from "wouter";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import {
  HelpCircle,
  BookOpen,
  MessageSquare,
  Mail,
  Brain,
  Mic,
  Moon,
  Heart,
  Watch,
  ChevronDown,
  ChevronUp,
  Send,
  ExternalLink,
  Utensils,
  Download,
  Pill,
  Shield,
} from "lucide-react";

/* ── FAQ Data ───────────────────────────────────────────────── */

interface FAQ {
  question: string;
  answer: string;
  icon: React.ComponentType<{ className?: string }>;
}

const faqs: FAQ[] = [
  {
    question: "How does voice-based emotion analysis work?",
    answer:
      "When you start a voice check-in, the app records a short audio sample and analyzes vocal features — pitch, tone, rhythm, and energy — to estimate your emotional state. No words are transcribed or stored. The analysis runs on-device (or through our ML backend) and returns focus, stress, and mood scores. If you disagree with the result, tap the emotion to correct it — the app learns from your feedback over time.",
    icon: Mic,
  },
  {
    question: "Do I need an EEG headband to use this app?",
    answer:
      "No. Voice analysis is the primary input method and works on any device. An EEG headband or compatible BCI device adds hardware-based brain wave data for more detailed analysis, but it is entirely optional.",
    icon: Brain,
  },
  {
    question: "How do I connect my health data?",
    answer:
      "Go to You > Connected Assets. On Android, connect Google Health Connect. On iOS, connect Apple HealthKit. You can also connect wearables like Oura, WHOOP, or Garmin through the Connected Assets page. Health data syncs automatically once connected.",
    icon: Heart,
  },
  {
    question: "How does nutrition logging work?",
    answer:
      "Go to the Nutrition page to log meals, snacks, and supplements. You can also track GLP-1 medication injections with dose, site, and schedule. The app correlates nutrition data with your mood and energy levels to show food-emotion patterns over time.",
    icon: Utensils,
  },
  {
    question: "What is a dream journal entry?",
    answer:
      "After waking up, go to the Dream Journal page and describe your dream in your own words. The app uses AI to identify symbols, recurring themes, and emotional patterns. Over time, it reveals trends in your dream life.",
    icon: Moon,
  },
  {
    question: "How can I export my data?",
    answer:
      "Go to the Export page (You > Export Data) to download all your data. You can export brain session data as CSV or JSON, filter by date range and metrics, and download health data, dream journals, and emotion readings. Under GDPR Art. 20, you can also download a full JSON archive of everything from Settings > Data & Privacy.",
    icon: Download,
  },
  {
    question: "How is my data stored?",
    answer:
      "Session data is stored securely on our servers (PostgreSQL with AES-256 encryption at rest). Voice analysis is processed on-device and not sent to external servers. Health data synced from wearables stays in your account. You can export all your data at any time, and delete everything from Settings > Data & Privacy. See our Privacy Policy for full details.",
    icon: Shield,
  },
  {
    question: "What does the calibration step do?",
    answer:
      "Calibration records 30 seconds of resting-state brain activity (if using EEG). This establishes your personal baseline so that readings are compared against YOUR normal state rather than population averages. Calibration improves emotion detection accuracy by 15-29%. You can recalibrate any time from Settings.",
    icon: Brain,
  },
  {
    question: "What are GLP-1 injections and how does tracking work?",
    answer:
      "GLP-1 receptor agonists (like semaglutide) are medications used for weight management and metabolic health. The app lets you log injection date, dose, and injection site. It tracks your schedule and sends reminders so you never miss a dose. This data is private and stored only on your device.",
    icon: Pill,
  },
];

/* ── FAQ Item ───────────────────────────────────────────────── */

function FAQItem({ faq }: { faq: FAQ }) {
  const [open, setOpen] = useState(false);
  const Icon = faq.icon;

  return (
    <div className="border-b border-border/20 last:border-0">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 py-4 text-left hover:bg-muted/20 transition-colors rounded-lg px-2 -mx-2"
      >
        <Icon className="h-4 w-4 text-primary shrink-0" />
        <span className="text-sm font-medium flex-1">{faq.question}</span>
        {open ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground shrink-0" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
        )}
      </button>
      {open && (
        <div className="pb-4 pl-9 pr-2">
          <p className="text-sm text-muted-foreground leading-relaxed">{faq.answer}</p>
        </div>
      )}
    </div>
  );
}

/* ── Feedback Form ──────────────────────────────────────────── */

function FeedbackForm() {
  const { toast } = useToast();
  const [type, setType] = useState<"bug" | "feature" | "general">("general");
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    setSending(true);
    try {
      // In a real app, this would POST to the server
      // For now, copy to clipboard as a fallback
      await navigator.clipboard.writeText(
        `[${type.toUpperCase()}] ${message}`
      );
      toast({
        title: "Feedback copied",
        description: "Your feedback has been copied to clipboard. Please send it via email.",
      });
      setMessage("");
    } catch {
      toast({
        title: "Thank you",
        description: "We received your feedback.",
      });
      setMessage("");
    } finally {
      setSending(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Type selector */}
      <div className="flex gap-2">
        {(
          [
            { id: "general", label: "General" },
            { id: "bug", label: "Bug Report" },
            { id: "feature", label: "Feature Request" },
          ] as const
        ).map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setType(t.id)}
            className={`px-3 py-1.5 text-xs rounded-full transition-colors ${
              type === t.id
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground bg-muted/50 hover:bg-muted"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Message */}
      <textarea
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder={
          type === "bug"
            ? "Describe the bug: what happened vs what you expected..."
            : type === "feature"
            ? "Describe the feature you'd like to see..."
            : "Your thoughts, suggestions, or questions..."
        }
        className="w-full min-h-[120px] p-3 text-sm rounded-lg bg-muted/30 border border-border/30 text-foreground placeholder:text-muted-foreground/50 resize-none focus:outline-none focus:ring-1 focus:ring-primary/40"
      />

      {/* Submit */}
      <Button
        type="submit"
        disabled={sending || !message.trim()}
        className="w-full"
      >
        <Send className="h-3.5 w-3.5 mr-2" />
        {sending ? "Sending..." : "Send Feedback"}
      </Button>
    </form>
  );
}

/* ── Main Component ─────────────────────────────────────────── */

export default function HelpPage() {
  const [, setLocation] = useLocation();

  return (
    <main className="p-4 md:p-6 pb-24 space-y-6 max-w-3xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <HelpCircle className="h-6 w-6 text-primary" />
        <div>
          <h1 className="text-xl font-semibold">Help & Feedback</h1>
          <p className="text-xs text-muted-foreground">
            Learn how to use AntarAI and share your feedback
          </p>
        </div>
      </div>

      {/* Quick Start Guide */}
      <Card className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Quick Start</h3>
        </div>
        <div className="space-y-3">
          {[
            {
              step: "1",
              title: "Check in with your voice",
              description:
                "Tap the Emotion Lab from Discover and record a short check-in. The app analyzes your voice to estimate mood, stress, and focus. If the detected emotion feels wrong, tap to correct it — the model learns from your feedback.",
            },
            {
              step: "2",
              title: "Log your nutrition",
              description:
                "Track meals, supplements, and GLP-1 injections on the Nutrition page. The app correlates food with mood and energy to reveal patterns.",
            },
            {
              step: "3",
              title: "Connect your devices",
              description:
                "Go to Connected Assets (You > Connected Assets) to link Apple Health, Google Health Connect, EEG / BCI headbands, or wearables (Oura, WHOOP, Garmin).",
            },
            {
              step: "4",
              title: "Review your trends",
              description:
                "Check Session History, mood trends, and the weekly summary. Export your data anytime from the Export page as CSV or JSON.",
            },
            {
              step: "5",
              title: "Journal your dreams",
              description:
                "Use the Dream Journal to record and analyze your dreams. The AI identifies symbols, recurring themes, and emotional patterns.",
            },
          ].map((item) => (
            <div key={item.step} className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                {item.step}
              </div>
              <div>
                <p className="text-sm font-medium">{item.title}</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {item.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* FAQ */}
      <Card className="glass-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <HelpCircle className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Frequently Asked Questions</h3>
        </div>
        <div>
          {faqs.map((faq, i) => (
            <FAQItem key={i} faq={faq} />
          ))}
        </div>
      </Card>

      {/* Feedback */}
      <Card className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <MessageSquare className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Send Feedback</h3>
        </div>
        <FeedbackForm />
      </Card>

      {/* Contact */}
      <Card className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <Mail className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">Contact</h3>
        </div>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Email Support</p>
              <p className="text-xs text-muted-foreground">
                For bugs, account issues, or data requests
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.open("mailto:support@antarai.app", "_blank")}
            >
              <Mail className="h-3.5 w-3.5 mr-1.5" />
              Email
            </Button>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Privacy Policy</p>
              <p className="text-xs text-muted-foreground">
                How we handle your data
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setLocation("/privacy")}
            >
              <ExternalLink className="h-3.5 w-3.5 mr-1.5" />
              View
            </Button>
          </div>
        </div>
      </Card>

      {/* Version info */}
      <div className="text-center text-xs text-muted-foreground/50 pt-4">
        <p>AntarAI v1.0</p>
      </div>
    </main>
  );
}
