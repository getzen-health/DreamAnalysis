import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Mic, MicOff } from "lucide-react";

interface VoiceRecorderProps {
  onTranscript: (text: string) => void;
  onRecordingChange?: (isRecording: boolean) => void;
}

export function VoiceRecorder({ onTranscript, onRecordingChange }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const recognitionRef = useRef<any>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);

  const startRecording = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event: any) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      if (finalTranscript) {
        onTranscript(finalTranscript);
      }
    };

    recognition.onerror = () => stopRecording();
    recognition.onend = () => {
      if (isRecording) {
        try { recognition.start(); } catch {}
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
    onRecordingChange?.(true);

    // Audio level visualization
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateLevel = () => {
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setAudioLevel(avg / 255);
        animFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();
    }).catch(() => {});
  };

  const stopRecording = () => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    cancelAnimationFrame(animFrameRef.current);
    setIsRecording(false);
    setAudioLevel(0);
    onRecordingChange?.(false);
  };

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
      cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  return (
    <div className="flex items-center gap-2">
      <Button
        variant="outline"
        size="sm"
        onClick={isRecording ? stopRecording : startRecording}
        className={`relative ${isRecording ? "bg-destructive/20 border-destructive/30 text-destructive" : "bg-secondary/10 border-secondary/30 text-secondary"}`}
      >
        {isRecording ? <MicOff className="h-4 w-4 mr-1" /> : <Mic className="h-4 w-4 mr-1" />}
        {isRecording ? "Stop" : "Voice"}
      </Button>

      {isRecording && (
        <div className="flex items-center gap-0.5">
          {Array.from({ length: 5 }, (_, i) => (
            <div
              key={i}
              className="w-1 bg-destructive rounded-full transition-all duration-100"
              style={{
                height: `${Math.max(4, audioLevel * 24 * (1 + Math.sin(i * 1.5) * 0.5))}px`,
              }}
            />
          ))}
          <span className="text-xs text-destructive ml-1 animate-pulse">REC</span>
        </div>
      )}
    </div>
  );
}
