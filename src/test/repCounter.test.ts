import { describe, it, expect } from "vitest";
import { RepCounter, summarizeReps, estimateCalories, type RepEvent } from "@/lib/repCounter";

const FRAME_MS = 1000 / 24;

/** Feed a sequence of signal values at ~24fps and collect emitted events. */
function run(counter: RepCounter, signals: number[], quality = 90, startMs = 0) {
  const events: Exclude<RepEvent, null>[] = [];
  signals.forEach((s, i) => {
    const { event } = counter.update(s, quality, startMs + i * FRAME_MS);
    if (event) events.push(event);
  });
  return events;
}

/** Linear ramp from a to b over n frames. */
const ramp = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

const squatConfig = { startThreshold: 150, bottomThreshold: 120, topThreshold: 155, minRepMs: 700 };

const fullSquat = () => [...ramp(170, 95, 15), ...ramp(95, 170, 15), ...Array(4).fill(170)];

describe("RepCounter", () => {
  it("counts one rep per full range-of-motion cycle", () => {
    const counter = new RepCounter(squatConfig);
    const events = run(counter, [...fullSquat(), ...fullSquat(), ...fullSquat()]);
    expect(events.filter((e) => e.type === "rep")).toHaveLength(3);
    expect(counter.reps).toBe(3);
  });

  it("does not count jitter around a single threshold", () => {
    const counter = new RepCounter(squatConfig);
    // Oscillate in the zone the old lenient gate counted (140-148) without reaching depth.
    const jitter = Array.from({ length: 120 }, (_, i) => (i % 2 === 0 ? 140 : 148));
    const events = run(counter, [170, 170, ...jitter, 170, 170, 170]);
    expect(events.filter((e) => e.type === "rep")).toHaveLength(0);
    expect(counter.reps).toBe(0);
  });

  it("reports a partial rep when depth is not reached", () => {
    const counter = new RepCounter(squatConfig);
    const half = [...ramp(170, 135, 12), ...ramp(135, 170, 12), ...Array(4).fill(170)];
    const events = run(counter, half);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("partial");
    expect(counter.reps).toBe(0);
  });

  it("rejects reps faster than the minimum duration", () => {
    const counter = new RepCounter(squatConfig);
    const twitch = [170, 170, 110, 110, 170, 170, 170];
    expect(run(counter, twitch).filter((e) => e.type === "rep")).toHaveLength(0);
  });

  it("supports upward movements (jumping jack openness)", () => {
    const counter = new RepCounter({
      direction: "up",
      startThreshold: 0.3,
      bottomThreshold: 0.7,
      topThreshold: 0.25,
      stableFrames: 1,
      minRepMs: 350,
    });
    const jack = [...ramp(0.05, 0.9, 7), ...ramp(0.9, 0.05, 7), 0.05, 0.05];
    const events = run(counter, [...jack, ...jack, ...jack, ...jack]);
    expect(events.filter((e) => e.type === "rep")).toHaveLength(4);
  });

  it("records depth, duration and a bounded score per rep", () => {
    const counter = new RepCounter(squatConfig);
    const rep = run(counter, fullSquat(), 80).find((e) => e.type === "rep");
    expect(rep?.type).toBe("rep");
    if (rep?.type !== "rep") return;
    expect(rep.rep.depth).toBeLessThanOrEqual(100);
    expect(rep.rep.durationMs).toBeGreaterThan(700);
    expect(rep.rep.score).toBe(80);
  });

  it("abortCycle discards an in-progress rep", () => {
    const counter = new RepCounter(squatConfig);
    run(counter, ramp(170, 95, 15));
    counter.abortCycle();
    const events = run(counter, [...ramp(95, 170, 15), 170, 170, 170], 90, 2000);
    expect(events.filter((e) => e.type === "rep")).toHaveLength(0);
  });
});

describe("summarizeReps", () => {
  it("returns zeros for an empty session", () => {
    expect(summarizeReps([])).toEqual({ avgScore: 0, bestScore: 0, avgTempoSeconds: 0, consistency: 0 });
  });

  it("aggregates score, tempo and consistency", () => {
    const summary = summarizeReps([
      { index: 1, score: 80, depth: 100, durationMs: 2000, completedAt: 0 },
      { index: 2, score: 90, depth: 95, durationMs: 2000, completedAt: 0 },
    ]);
    expect(summary).toEqual({ avgScore: 85, bestScore: 90, avgTempoSeconds: 2, consistency: 100 });
  });
});

describe("estimateCalories", () => {
  it("scales rep calories with body weight", () => {
    expect(estimateCalories("Squat", 10, 0, 70)).toBe(3.2);
    expect(estimateCalories("Squat", 10, 0, 140)).toBe(6.4);
  });

  it("uses only good-form hold time for plank and ignores invalid weights", () => {
    expect(estimateCalories("Plank", 0, 60, 70)).toBe(4.7);
    expect(estimateCalories("Plank", 0, 60, -5)).toBe(4.7);
    expect(estimateCalories("Plank", 0, 0, 70)).toBe(0);
  });
});
