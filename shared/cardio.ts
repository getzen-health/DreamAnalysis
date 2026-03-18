// Shared cardio computation utilities — used by both server routes and client UI.

// Heart rate zone boundaries (220-age formula)
export function getHrZones(age: number, restingHr: number = 60) {
  const maxHr = 220 - age;
  return {
    max: maxHr,
    resting: restingHr,
    zones: [
      { zone: 1, name: 'Active Recovery', min: Math.round(maxHr * 0.50), max: Math.round(maxHr * 0.59) },
      { zone: 2, name: 'Fat Burning', min: Math.round(maxHr * 0.60), max: Math.round(maxHr * 0.69) },
      { zone: 3, name: 'Cardiovascular', min: Math.round(maxHr * 0.70), max: Math.round(maxHr * 0.79) },
      { zone: 4, name: 'High Intensity', min: Math.round(maxHr * 0.80), max: Math.round(maxHr * 0.89) },
      { zone: 5, name: 'Maximum Effort', min: Math.round(maxHr * 0.90), max: maxHr },
    ]
  };
}

// TRIMP (Training Impulse) calculation
export function computeTrimp(durationMin: number, avgHr: number, restingHr: number, maxHr: number, gender: 'male' | 'female' = 'male'): number {
  const hrRatio = (avgHr - restingHr) / (maxHr - restingHr);
  const genderFactor = gender === 'male' ? 1.92 : 1.67;
  return durationMin * hrRatio * Math.exp(genderFactor * hrRatio);
}

// Strain score from TRIMP (logarithmic, k=14.3)
export function computeStrain(trimp: number): number {
  return 14.3 * Math.log(1 + trimp);
}

// Epley 1RM estimation
export function estimate1rm(weight: number, reps: number): number {
  if (reps <= 0 || weight <= 0) return 0;
  if (reps === 1) return weight;
  return Math.round(weight * (1 + reps / 30) * 10) / 10;
}

// Cardio Load: Acute (7-day) vs Chronic (42-day) Training Load
export function computeCardioLoad(dailyTrimpValues: { date: string; trimp: number }[]): {
  atl: number; // Acute Training Load (7-day)
  ctl: number; // Chronic Training Load (42-day)
  tsb: number; // Training Stress Balance (CTL - ATL)
  status: string; // 7 status categories
} {
  const sorted = [...dailyTrimpValues].sort((a, b) => a.date.localeCompare(b.date));

  let atl = 0; // 7-day exponentially weighted
  let ctl = 0; // 42-day exponentially weighted
  const alphaAtl = 2 / (7 + 1);
  const alphaCtl = 2 / (42 + 1);

  for (const entry of sorted) {
    atl = atl * (1 - alphaAtl) + entry.trimp * alphaAtl;
    ctl = ctl * (1 - alphaCtl) + entry.trimp * alphaCtl;
  }

  const tsb = ctl - atl;

  let status: string;
  if (sorted.length < 14) status = 'Calibrating';
  else if (tsb > 15) status = 'Detraining';
  else if (tsb > 5) status = 'Maintaining';
  else if (tsb > -5) status = 'Productive';
  else if (tsb > -15) status = 'Peaking';
  else if (tsb > -25) status = 'Fatigued';
  else status = 'Overtraining';

  return { atl: Math.round(atl * 10) / 10, ctl: Math.round(ctl * 10) / 10, tsb: Math.round(tsb * 10) / 10, status };
}

// Heart Rate Recovery (post-workout HR drop)
export function computeHrRecovery(peakHr: number, hrAt2Min: number): { value: number; rating: string } {
  const drop = peakHr - hrAt2Min;
  let rating: string;
  if (drop > 50) rating = 'Excellent';
  else if (drop > 40) rating = 'Good';
  else if (drop > 30) rating = 'Average';
  else rating = 'Below Average';
  return { value: drop, rating };
}
