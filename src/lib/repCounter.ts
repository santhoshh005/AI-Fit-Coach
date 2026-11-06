/**
 * Generic, frame-driven repetition counter.
 *
 * A rep is a full cycle: leave the start position, reach the "bottom" (deepest
 * point of the movement), then return to the start position. Hysteresis between
 * the thresholds plus a short stable-frame requirement stops landmark jitter from
 * producing phantom reps, and a minimum rep duration rejects twitches.
 *
 * For "down" movements (squat, pushup, curl, lunge) the signal is a joint angle
 * that decreases into the rep. For "up" movements (jumping jack) the signal is an
 * openness value that increases into the rep.
 */

export type RepPhase = "top" | "descending" | "bottom" | "ascending";

export interface RepCounterConfig {
  /** Signal crossing this value (moving into the rep) leaves the top position. */
  startThreshold: number;
  /** Signal reaching this value marks the bottom of the rep (full range). */
  bottomThreshold: number;
  /** Signal returning past this value completes the rep. */
  topThreshold: number;
  /** "down": signal decreases into the rep. "up": signal increases into the rep. */
  direction?: "down" | "up";
  /** Consecutive frames required to confirm a phase change. */
  stableFrames?: number;
  /** Reps faster than this are treated as noise. */
  minRepMs?: number;
  /** Cycles longer than this are abandoned (user paused mid-rep). */
  maxRepMs?: number;
}

export interface RepRecord {
  index: number;
  /** 0-100 form score for the rep. */
  score: number;
  /** Most extreme signal value reached during the rep (e.g. lowest knee angle). */
  depth: number;
  durationMs: number;
  completedAt: number;
}

export type RepEvent =
  | { type: "rep"; rep: RepRecord }
  | { type: "partial"; depth: number }
  | null;

export interface RepCounterUpdate {
  phase: RepPhase;
  event: RepEvent;
}

const DEFAULTS = {
  direction: "down" as const,
  stableFrames: 2,
  minRepMs: 500,
  maxRepMs: 10000,
};

export class RepCounter {
  private readonly cfg: Required<RepCounterConfig>;
  private phase: RepPhase = "top";
  private candidate: RepPhase | null = null;
  private candidateFrames = 0;
  private repStartMs: number | null = null;
  private reachedBottom = false;
  private extreme = 0;
  private qualitySum = 0;
  private qualityCount = 0;
  private qualityPeak = 0;
  private count = 0;

  constructor(config: RepCounterConfig) {
    this.cfg = { ...DEFAULTS, ...config };
  }

  get reps(): number {
    return this.count;
  }

  get currentPhase(): RepPhase {
    return this.phase;
  }

  reset(): void {
    this.count = 0;
    this.abortCycle();
  }

  /** Discard any in-progress rep without counting it (e.g. tracking was lost). */
  abortCycle(): void {
    this.phase = "top";
    this.candidate = null;
    this.candidateFrames = 0;
    this.clearCycle();
  }

  update(rawSignal: number, quality: number, nowMs: number): RepCounterUpdate {
    if (!Number.isFinite(rawSignal)) return { phase: this.phase, event: null };

    // Normalise so that "into the rep" is always a decreasing value.
    const flip = this.cfg.direction === "up" ? -1 : 1;
    const s = rawSignal * flip;
    const start = this.cfg.startThreshold * flip;
    const bottom = this.cfg.bottomThreshold * flip;
    const top = this.cfg.topThreshold * flip;

    if (this.repStartMs !== null) {
      this.extreme = Math.min(this.extreme, s);
      this.qualitySum += quality;
      this.qualityCount += 1;
      this.qualityPeak = Math.max(this.qualityPeak, quality);

      if (nowMs - this.repStartMs > this.cfg.maxRepMs) {
        this.abortCycle();
        return { phase: this.phase, event: null };
      }
    }

    let target: RepPhase | null = null;
    switch (this.phase) {
      case "top":
        if (s < start) target = "descending";
        break;
      case "descending":
        if (s <= bottom) target = "bottom";
        else if (s >= top) target = "top";
        break;
      case "bottom":
        if (s > bottom) target = "ascending";
        break;
      case "ascending":
        if (s >= top) target = "top";
        else if (s <= bottom) target = "bottom";
        break;
    }

    if (!target) {
      this.candidate = null;
      this.candidateFrames = 0;
      return { phase: this.phase, event: null };
    }

    if (this.candidate === target) {
      this.candidateFrames += 1;
    } else {
      this.candidate = target;
      this.candidateFrames = 1;
    }

    // Reaching the bottom is confirmed immediately so fast reps are not missed;
    // every other transition needs stable frames to filter jitter.
    const needed = target === "bottom" ? 1 : this.cfg.stableFrames;
    if (this.candidateFrames < needed) return { phase: this.phase, event: null };

    const previous = this.phase;
    this.phase = target;
    this.candidate = null;
    this.candidateFrames = 0;

    if (target === "descending" && previous === "top") {
      this.startCycle(s, quality, nowMs);
      return { phase: this.phase, event: null };
    }

    if (target === "bottom") {
      this.reachedBottom = true;
      return { phase: this.phase, event: null };
    }

    if (target === "top") {
      return { phase: this.phase, event: this.finishCycle(nowMs, flip) };
    }

    return { phase: this.phase, event: null };
  }

  private startCycle(s: number, quality: number, nowMs: number): void {
    this.repStartMs = nowMs;
    this.reachedBottom = false;
    this.extreme = s;
    this.qualitySum = quality;
    this.qualityCount = 1;
    this.qualityPeak = quality;
  }

  private finishCycle(nowMs: number, flip: number): RepEvent {
    const startedAt = this.repStartMs;
    const reachedBottom = this.reachedBottom;
    const depth = this.extreme * flip;
    const meanQuality = this.qualityCount > 0 ? this.qualitySum / this.qualityCount : 0;
    const peakQuality = this.qualityPeak;
    this.clearCycle();

    if (startedAt === null) return null;
    const durationMs = nowMs - startedAt;

    if (!reachedBottom) {
      return durationMs >= 250 ? { type: "partial", depth } : null;
    }

    if (durationMs < this.cfg.minRepMs) return null;

    this.count += 1;
    // Peak captures form at the hardest point; mean captures control through the rep.
    const score = Math.round(Math.max(0, Math.min(100, peakQuality * 0.5 + meanQuality * 0.5)));
    return {
      type: "rep",
      rep: { index: this.count, score, depth, durationMs, completedAt: nowMs },
    };
  }

  private clearCycle(): void {
    this.repStartMs = null;
    this.reachedBottom = false;
    this.extreme = 0;
    this.qualitySum = 0;
    this.qualityCount = 0;
    this.qualityPeak = 0;
  }
}

export interface RepSummary {
  avgScore: number;
  bestScore: number;
  avgTempoSeconds: number;
  /** 0-100: how even the tempo was across reps. */
  consistency: number;
}

/** Aggregate per-rep records into session-level metrics. */
export function summarizeReps(reps: RepRecord[]): RepSummary {
  if (reps.length === 0) {
    return { avgScore: 0, bestScore: 0, avgTempoSeconds: 0, consistency: 0 };
  }
  const scores = reps.map((r) => r.score);
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  const tempos = reps.map((r) => r.durationMs / 1000);
  const avgTempo = tempos.reduce((a, b) => a + b, 0) / tempos.length;
  const variance = tempos.reduce((acc, t) => acc + (t - avgTempo) ** 2, 0) / tempos.length;
  const cv = avgTempo > 0 ? Math.sqrt(variance) / avgTempo : 0;
  return {
    avgScore: Math.round(avg),
    bestScore: Math.max(...scores),
    avgTempoSeconds: Number(avgTempo.toFixed(2)),
    consistency: Math.round(Math.max(0, Math.min(100, 100 - cv * 100))),
  };
}

/** Approximate kcal per rep for a 70 kg person, scaled linearly by body weight. */
const KCAL_PER_REP_70KG: Record<string, number> = {
  Squat: 0.32,
  Pushup: 0.36,
  "Biceps Curl": 0.15,
  Lunge: 0.3,
  "Jumping Jack": 0.2,
};

/** Plank MET ~3.8; kcal/min = MET * 3.5 * kg / 200. */
const PLANK_MET = 3.8;

export function estimateCalories(
  exercise: string,
  reps: number,
  plankActiveSeconds: number,
  bodyWeightKg = 70
): number {
  const kg = Number.isFinite(bodyWeightKg) && bodyWeightKg >= 30 && bodyWeightKg <= 250 ? bodyWeightKg : 70;
  if (exercise === "Plank") {
    const kcal = ((PLANK_MET * 3.5 * kg) / 200 / 60) * plankActiveSeconds;
    return Number(kcal.toFixed(1));
  }
  const perRep = KCAL_PER_REP_70KG[exercise] ?? 0.3;
  return Number((reps * perRep * (kg / 70)).toFixed(1));
}
