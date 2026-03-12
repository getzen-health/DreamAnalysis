import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { getParticipantId } from "@/lib/participant";
import { resolveUrl } from "@/lib/queryClient";

export interface HealthMetrics {
  heartRate: number;
  stressLevel: number;
  sleepQuality: number;
  neuralActivity: number;
  dailySteps?: number;
  sleepDuration?: number;
}

export interface EEGData {
  alphaWaves: number[];
  betaWaves: number[];
  timestamp: number;
}

export interface NeuralActivity {
  frontal: number;
  temporal: number;
  parietal: number;
  occipital: number;
  brainstem: number;
}

const USER_ID = getParticipantId();

export function useMetrics() {
  const [currentMetrics, setCurrentMetrics] = useState<HealthMetrics>({
    heartRate: 72,
    stressLevel: 34,
    sleepQuality: 87,
    neuralActivity: 92,
    dailySteps: 8247,
    sleepDuration: 7.38
  });

  const [eegData, setEegData] = useState<EEGData>({
    alphaWaves: Array.from({length: 50}, () => Math.sin(Math.random() * Math.PI * 2) * 50 + Math.random() * 20),
    betaWaves: Array.from({length: 50}, () => Math.sin(Math.random() * Math.PI * 4) * 30 + Math.random() * 15),
    timestamp: Date.now()
  });

  const [neuralActivity, setNeuralActivity] = useState<NeuralActivity>({
    frontal: 85,
    temporal: 72,
    parietal: 68,
    occipital: 91,
    brainstem: 77
  });

  // Fetch historical metrics
  const { data: historicalMetrics, isLoading } = useQuery({
    queryKey: ["/api/health-metrics", USER_ID],
    queryFn: async () => {
      const response = await fetch(resolveUrl(`/api/health-metrics/${USER_ID}`));
      if (!response.ok) throw new Error('Failed to fetch metrics');
      return response.json();
    }
  });

  // Update metrics simulation
  const updateMetrics = useCallback(() => {
    setCurrentMetrics(prev => ({
      heartRate: Math.max(60, Math.min(100, prev.heartRate + Math.floor(Math.random() * 6) - 3)),
      stressLevel: Math.max(0, Math.min(100, prev.stressLevel + Math.floor(Math.random() * 10) - 5)),
      sleepQuality: Math.max(0, Math.min(100, prev.sleepQuality + Math.floor(Math.random() * 6) - 3)),
      neuralActivity: Math.max(0, Math.min(100, prev.neuralActivity + Math.floor(Math.random() * 8) - 4)),
      dailySteps: prev.dailySteps ? prev.dailySteps + Math.floor(Math.random() * 50) : 8247,
      sleepDuration: prev.sleepDuration
    }));

    // Update EEG data
    setEegData(prev => ({
      alphaWaves: [
        ...prev.alphaWaves.slice(1),
        Math.sin(Date.now() * 0.01) * 50 + Math.random() * 20
      ],
      betaWaves: [
        ...prev.betaWaves.slice(1),
        Math.sin(Date.now() * 0.02) * 30 + Math.random() * 15
      ],
      timestamp: Date.now()
    }));

    // Update neural activity
    setNeuralActivity(prev => ({
      frontal: Math.max(20, Math.min(100, prev.frontal + Math.floor(Math.random() * 10) - 5)),
      temporal: Math.max(20, Math.min(100, prev.temporal + Math.floor(Math.random() * 8) - 4)),
      parietal: Math.max(20, Math.min(100, prev.parietal + Math.floor(Math.random() * 6) - 3)),
      occipital: Math.max(20, Math.min(100, prev.occipital + Math.floor(Math.random() * 12) - 6)),
      brainstem: Math.max(20, Math.min(100, prev.brainstem + Math.floor(Math.random() * 4) - 2))
    }));
  }, []);

  useEffect(() => {
    const interval = setInterval(updateMetrics, 2000);
    return () => clearInterval(interval);
  }, [updateMetrics]);

  // Generate mood data for charts
  const moodData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [{
      label: 'Mood Score',
      data: [7.2, 6.8, 8.1, 7.5, 6.9, 8.3, 7.8],
      borderColor: 'hsl(195, 100%, 50%)',
      backgroundColor: 'hsla(195, 100%, 50%, 0.1)',
      tension: 0.4,
      fill: true
    }]
  };

  const sleepData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Sleep Hours',
        data: [7.5, 6.8, 8.2, 7.1, 6.5, 8.5, 7.3],
        backgroundColor: 'hsla(270, 70%, 65%, 0.8)',
        borderRadius: 4
      },
      {
        label: 'Dream Activity',
        data: [3.2, 2.8, 4.1, 3.5, 2.9, 4.3, 3.8],
        backgroundColor: 'hsla(120, 100%, 55%, 0.8)',
        borderRadius: 4
      }
    ]
  };

  return {
    currentMetrics,
    eegData,
    neuralActivity,
    moodData,
    sleepData,
    historicalMetrics,
    isLoading,
    userId: USER_ID
  };
}
