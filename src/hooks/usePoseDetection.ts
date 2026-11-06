import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import {
  POSE_CONNECTIONS,
  type Pose,
  type NormalizedLandmark,
  type NormalizedLandmarkList,
  type Results,
} from "@mediapipe/pose";
import * as drawingUtilsModule from "@mediapipe/drawing_utils";
import {
  RepCounter,
  estimateCalories,
  summarizeReps,
  type RepCounterConfig,
  type RepEvent,
  type RepRecord,
  type RepSummary,
} from "@/lib/repCounter";

// Gate debug logging to development builds only
const IS_DEV = import.meta.env.DEV;

// Vite production build can resolve legacy UMD/CommonJS imports differently.
// We implement a robust fallback strategy to retrieve drawConnectors and drawLandmarks safely.
// When the MediaPipe drawing_utils library cannot be resolved, we use native Canvas2D drawing.

// Native Canvas2D fallback: draw connection lines between landmarks
const nativeDrawConnectors = (
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmarkList,
  connections: [number, number][],
  style: { color?: string; lineWidth?: number }
): void => {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.save();
  ctx.strokeStyle = style.color || "#22c55e";
  ctx.lineWidth = style.lineWidth || 3;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const [startIdx, endIdx] of connections) {
    const startLm = landmarks[startIdx];
    const endLm = landmarks[endIdx];
    if (!startLm || !endLm) continue;
    // Skip low-visibility landmarks
    if ((startLm.visibility ?? 0) < 0.3 || (endLm.visibility ?? 0) < 0.3) continue;

    ctx.beginPath();
    ctx.moveTo(startLm.x * w, startLm.y * h);
    ctx.lineTo(endLm.x * w, endLm.y * h);
    ctx.stroke();
  }
  ctx.restore();
};

// Native Canvas2D fallback: draw landmark dots
const nativeDrawLandmarks = (
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmarkList,
  style: { color?: string; lineWidth?: number; radius?: number }
): void => {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const r = style.radius || 3;
  ctx.save();
  ctx.fillStyle = style.color || "#14b8a6";
  ctx.strokeStyle = style.color || "#14b8a6";
  ctx.lineWidth = style.lineWidth || 2;

  for (const lm of landmarks) {
    if (!lm || (lm.visibility ?? 0) < 0.3) continue;
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
};

const getDrawingUtils = () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mod = drawingUtilsModule as any;
  if (mod && typeof mod.drawConnectors === "function") {
    return {
      drawConnectors: mod.drawConnectors as typeof drawingUtilsModule.drawConnectors,
      drawLandmarks: mod.drawLandmarks as typeof drawingUtilsModule.drawLandmarks,
    };
  }

  if (mod && mod.default && typeof mod.default.drawConnectors === "function") {
    return {
      drawConnectors: mod.default.drawConnectors as typeof drawingUtilsModule.drawConnectors,
      drawLandmarks: mod.default.drawLandmarks as typeof drawingUtilsModule.drawLandmarks,
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  if (w && typeof w.drawConnectors === "function") {
    return {
      drawConnectors: w.drawConnectors as typeof drawingUtilsModule.drawConnectors,
      drawLandmarks: w.drawLandmarks as typeof drawingUtilsModule.drawLandmarks,
    };
  }

  if (w && w.DrawingUtils) {
    if (typeof w.DrawingUtils.drawConnectors === "function") {
      return {
        drawConnectors: w.DrawingUtils.drawConnectors as typeof drawingUtilsModule.drawConnectors,
        drawLandmarks: w.DrawingUtils.drawLandmarks as typeof drawingUtilsModule.drawLandmarks,
      };
    }
  }

  return null;
};

const mpDrawingUtils = getDrawingUtils();

// Whether the MediaPipe drawing library resolved successfully
const hasMediaPipeDrawing = mpDrawingUtils !== null;

const FALLBACK_POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Arms & shoulders
  [11, 23], [12, 24], [23, 24],                    // Torso & hips
  [23, 25], [25, 27], [24, 26], [26, 28],          // Legs & ankles
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32], // Feet
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22], // Hands
  [0, 1], [1, 2], [2, 3], [3, 7],                  // Left face
  [0, 4], [4, 5], [5, 6], [6, 8],                  // Right face
  [9, 10]                                          // Mouth
];

const getPoseConnections = (): [number, number][] => {
  if (typeof POSE_CONNECTIONS !== "undefined" && Array.isArray(POSE_CONNECTIONS) && POSE_CONNECTIONS.length > 0) {
    return POSE_CONNECTIONS as [number, number][];
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  if (w && w.POSE_CONNECTIONS && Array.isArray(w.POSE_CONNECTIONS) && w.POSE_CONNECTIONS.length > 0) {
    return w.POSE_CONNECTIONS as [number, number][];
  }
  if (w && w.Pose && w.Pose.POSE_CONNECTIONS && Array.isArray(w.Pose.POSE_CONNECTIONS) && w.Pose.POSE_CONNECTIONS.length > 0) {
    return w.Pose.POSE_CONNECTIONS as [number, number][];
  }
  return FALLBACK_POSE_CONNECTIONS;
};

export type ExerciseType = "Biceps Curl" | "Squat" | "Pushup" | "Lunge" | "Jumping Jack" | "Plank";
export type DistanceStatus = "good" | "too-close" | "too-far" | "adjusting";

type FeedbackType = "success" | "warning";

export interface LiveFeedback {
  text: string;
  type: FeedbackType;
}

export interface LiveMetricCheck {
  label: string;
  current: string;
  threshold: string;
  pass: boolean;
}

interface UsePoseDetectionResult {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  cameraOn: boolean;
  errorMessage: string | null;
  reps: number;
  /** Per-rep records (score, depth, tempo) for the current session. */
  repHistory: RepRecord[];
  /** Movements that started but never reached full range. */
  partialReps: number;
  repSummary: RepSummary;
  /** 0-100 session form score: mean rep score, or mean plank posture while holding. */
  sessionFormScore: number;
  postureScore: number;
  calories: number;
  feedbackHistory: LiveFeedback[];
  elapsedSeconds: number;
  distanceStatus: DistanceStatus;
  distanceHint: string;
  currentPhase: string;
  liveChecks: LiveMetricCheck[];
  activePlankTime: number;
  perfectPlankTime: number;
  perfectPlankTimeAtTarget: number;
  totalElapsedTime: number;
  targetPlankTime: number;
  plankCompleted: boolean;
  setTargetPlankTime: (seconds: number) => void;
  startCamera: (overrideFacingMode?: "user" | "environment") => Promise<void>;
  stopCamera: () => void;
  resetSession: () => void;
  facingMode: "user" | "environment";
  toggleCameraFacing: () => Promise<void>;
}

const LANDMARKS = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
} as const;

const toDegrees = (radians: number): number => (radians * 180) / Math.PI;

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

const scoreFromRange = (value: number, target: number, spread: number): number => {
  const delta = Math.abs(value - target);
  return clamp(100 - (delta / spread) * 100, 0, 100);
};

const scoreInRange = (value: number, min: number, max: number, tolerance: number): number => {
  if (value >= min && value <= max) return 100;
  const delta = value < min ? min - value : value - max;
  return clamp(100 - (delta / tolerance) * 100, 0, 100);
};

const smoothValue = (prev: number, next: number, alpha = 0.5): number => prev * (1 - alpha) + next * alpha;

const estimateFrameBrightness = (ctx: CanvasRenderingContext2D, width: number, height: number): number => {
  const sampleW = Math.max(48, Math.floor(width / 8));
  const sampleH = Math.max(36, Math.floor(height / 8));
  const imageData = ctx.getImageData(0, 0, sampleW, sampleH).data;

  let sum = 0;
  for (let i = 0; i < imageData.length; i += 4) {
    const r = imageData[i];
    const g = imageData[i + 1];
    const b = imageData[i + 2];
    sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  const pixels = imageData.length / 4;
  return pixels > 0 ? sum / pixels : 255;
};

const visibilityOkay = (lms: NormalizedLandmarkList, indices: number[], min = 0.45): boolean => {
  return indices.every((index) => (lms[index]?.visibility ?? 0) >= min);
};

const preferVisiblePoint = (
  primary: { x: number; y: number; z?: number; visibility?: number },
  fallback: { x: number; y: number; z?: number; visibility?: number },
  min = 0.3
): { x: number; y: number; z?: number; visibility?: number } => {
  return (primary.visibility ?? 0) >= min ? primary : fallback;
};

const hasExerciseVisibility = (exercise: ExerciseType, lms: NormalizedLandmarkList, min = 0.35): boolean => {
  const torso = [
    LANDMARKS.leftShoulder,
    LANDMARKS.rightShoulder,
    LANDMARKS.leftHip,
    LANDMARKS.rightHip,
  ];

  const leftChain = [
    LANDMARKS.leftShoulder,
    LANDMARKS.leftElbow,
    LANDMARKS.leftWrist,
    LANDMARKS.leftHip,
    LANDMARKS.leftKnee,
    LANDMARKS.leftAnkle,
  ];

  const rightChain = [
    LANDMARKS.rightShoulder,
    LANDMARKS.rightElbow,
    LANDMARKS.rightWrist,
    LANDMARKS.rightHip,
    LANDMARKS.rightKnee,
    LANDMARKS.rightAnkle,
  ];

  const hasTorso = visibilityOkay(lms, torso, min);
  const hasLeft = visibilityOkay(lms, leftChain, min);
  const hasRight = visibilityOkay(lms, rightChain, min);
  const leftUpper = visibilityOkay(lms, [LANDMARKS.leftShoulder, LANDMARKS.leftElbow, LANDMARKS.leftWrist], min);
  const rightUpper = visibilityOkay(lms, [LANDMARKS.rightShoulder, LANDMARKS.rightElbow, LANDMARKS.rightWrist], min);
  const leftNoAnkle = visibilityOkay(
    lms,
    [LANDMARKS.leftShoulder, LANDMARKS.leftElbow, LANDMARKS.leftWrist, LANDMARKS.leftHip, LANDMARKS.leftKnee],
    min
  );
  const rightNoAnkle = visibilityOkay(
    lms,
    [LANDMARKS.rightShoulder, LANDMARKS.rightElbow, LANDMARKS.rightWrist, LANDMARKS.rightHip, LANDMARKS.rightKnee],
    min
  );
  const hasUpperBody = hasTorso && (leftUpper || rightUpper);

  if (exercise === "Biceps Curl") {
    // Curls are upper-body dominant; avoid blocking reps when legs are outside frame.
    return hasUpperBody;
  }

  if (exercise === "Pushup" || exercise === "Plank") {
    // Floor exercises can still be tracked if knees are visible while ankles drift out of frame.
    return hasTorso && (hasLeft || hasRight || leftNoAnkle || rightNoAnkle);
  }

  if (exercise === "Jumping Jack") {
    // Jumping jacks need both sides visible for open/close symmetry checks.
    return hasTorso && hasLeft && hasRight;
  }

  // Other exercises can be tracked from either dominant side.
  return hasTorso && (hasLeft || hasRight);
};

const getDistanceStatus = (
  exercise: ExerciseType,
  landmarks: NormalizedLandmarkList,
  aspectRatio = 1.777
): { status: DistanceStatus; hint: string } => {
  const tracked =
    exercise === "Biceps Curl"
      ? [
          landmarks[LANDMARKS.leftShoulder],
          landmarks[LANDMARKS.rightShoulder],
          landmarks[LANDMARKS.leftHip],
          landmarks[LANDMARKS.rightHip],
          landmarks[LANDMARKS.leftWrist],
          landmarks[LANDMARKS.rightWrist],
        ]
      : exercise === "Pushup" || exercise === "Plank"
        ? [
            landmarks[LANDMARKS.leftShoulder],
            landmarks[LANDMARKS.rightShoulder],
            landmarks[LANDMARKS.leftHip],
            landmarks[LANDMARKS.rightHip],
            landmarks[LANDMARKS.leftKnee],
            landmarks[LANDMARKS.rightKnee],
          ]
        : [
            landmarks[LANDMARKS.leftShoulder],
            landmarks[LANDMARKS.rightShoulder],
            landmarks[LANDMARKS.leftHip],
            landmarks[LANDMARKS.rightHip],
            landmarks[LANDMARKS.leftAnkle],
            landmarks[LANDMARKS.rightAnkle],
          ];

  if (tracked.some((lm) => !lm || (lm.visibility ?? 0) < 0.3)) {
    return {
      status: "adjusting",
      hint:
        exercise === "Biceps Curl"
          ? "Adjust so shoulders, hips, and wrists are visible."
          : exercise === "Pushup" || exercise === "Plank"
            ? "Adjust so shoulders, hips, and knees are visible."
          : "Adjust position so shoulders, hips, and ankles are visible.",
    };
  }

  const xs = tracked.map((lm) => lm.x * aspectRatio);
  const ys = tracked.map((lm) => lm.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const spanX = maxX - minX;
  const spanY = maxY - minY;

  const isHorizontal = exercise === "Pushup" || exercise === "Plank" || exercise === "Lunge";
  const bodySpan = isHorizontal ? spanX : spanY;

  let tooCloseThreshold = 0.86;
  let tooFarThreshold = 0.38;

  if (exercise === "Biceps Curl") {
    tooCloseThreshold = 0.76;
    tooFarThreshold = 0.32;
  } else if (exercise === "Pushup" || exercise === "Plank") {
    tooCloseThreshold = 0.86 * aspectRatio;
    tooFarThreshold = 0.36 * aspectRatio;
  } else if (exercise === "Lunge") {
    tooCloseThreshold = 0.86 * aspectRatio;
    tooFarThreshold = 0.44 * aspectRatio;
  }

  if (bodySpan > tooCloseThreshold) {
    return { status: "too-close", hint: "You are too close. Step back slightly." };
  }

  if (bodySpan < tooFarThreshold) {
    return { status: "too-far", hint: "You are too far. Move a bit closer." };
  }

  return { status: "good", hint: "Distance is ideal for tracking." };
};

const cloneLandmarks = (landmarks: NormalizedLandmarkList): NormalizedLandmarkList => {
  return landmarks.map((lm) => ({ ...lm })) as NormalizedLandmarkList;
};

const averageLandmarks = (history: NormalizedLandmarkList[]): NormalizedLandmarkList => {
  const frameCount = history.length;
  const pointsCount = history[0]?.length ?? 0;
  const averaged: NormalizedLandmark[] = [];

  for (let i = 0; i < pointsCount; i++) {
    let x = 0;
    let y = 0;
    let z = 0;
    let visibility = 0;

    for (let f = 0; f < frameCount; f++) {
      const lm = history[f][i];
      x += lm.x;
      y += lm.y;
      z += lm.z ?? 0;
      visibility += lm.visibility ?? 0;
    }

    averaged.push({
      x: x / frameCount,
      y: y / frameCount,
      z: z / frameCount,
      visibility: visibility / frameCount,
    });
  }

  return averaged as NormalizedLandmarkList;
};

const applyDeadband = (
  previous: NormalizedLandmarkList | null,
  next: NormalizedLandmarkList,
  epsilon = 0.0035
): NormalizedLandmarkList => {
  if (!previous || previous.length !== next.length) return next;

  const stabilized = next.map((lm, i) => {
    const prev = previous[i];
    return {
      ...lm,
      x: Math.abs(lm.x - prev.x) < epsilon ? prev.x : lm.x,
      y: Math.abs(lm.y - prev.y) < epsilon ? prev.y : lm.y,
      z: Math.abs((lm.z ?? 0) - (prev.z ?? 0)) < epsilon ? prev.z : lm.z,
    };
  });

  return stabilized as NormalizedLandmarkList;
};


// Use ONLY local assets to avoid CDN version mismatches.
// Local MediaPipe files are in public/mediapipe/pose/ and deployed with the app.
const POSE_ASSET_BASE_URLS = [
  "/mediapipe/pose",
] as const;

const angleFromThreePoints = (
  a: { x: number; y: number; z?: number },
  b: { x: number; y: number; z?: number },
  c: { x: number; y: number; z?: number },
  aspectRatio = 1.777
): number => {
  const scaleX = aspectRatio / 1.777;
  const ax = a.x * scaleX;
  const ay = a.y;
  const az = (a.z ?? 0) * scaleX;

  const bx = b.x * scaleX;
  const by = b.y;
  const bz = (b.z ?? 0) * scaleX;

  const cx = c.x * scaleX;
  const cy = c.y;
  const cz = (c.z ?? 0) * scaleX;

  const ab = { x: ax - bx, y: ay - by, z: az - bz };
  const cb = { x: cx - bx, y: cy - by, z: cz - bz };

  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const abMag = Math.sqrt(ab.x * ab.x + ab.y * ab.y + ab.z * ab.z);
  const cbMag = Math.sqrt(cb.x * cb.x + cb.y * cb.y + cb.z * cb.z);

  if (abMag === 0 || cbMag === 0) return 0;

  const cosine = clamp(dot / (abMag * cbMag), -1, 1);
  return toDegrees(Math.acos(cosine));
};

const getCameraErrorMessage = (error: unknown): string => {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError") {
      return "Camera permission was denied. Please allow camera access and try again.";
    }
    if (error.name === "NotReadableError") {
      return "Camera is already in use by another app. Close other apps using the camera and retry.";
    }
    if (error.name === "OverconstrainedError") {
      return "Your device could not satisfy the camera settings. Retrying with default camera settings may help.";
    }
    if (error.name === "SecurityError") {
      return "Camera access is blocked by browser security settings.";
    }
    if (error.name === "AbortError") {
      return "Camera startup was interrupted. Please try again.";
    }
  }

  if (error instanceof Error) {
    if (error.message.startsWith("pose-model-load-failed:")) {
      const detail = error.message.replace("pose-model-load-failed:", "").trim();
      return detail
        ? `Camera opened but AI model initialization failed (${detail}).`
        : "Camera opened but AI model initialization failed.";
    }

    if (error.message === "camera-unavailable") {
      return "No working camera was found on this device/browser. Try another browser or device.";
    }
    if (error.message === "video-playback-failed") {
      return "Camera opened but video playback failed in this browser. Please open this site in Chrome or Safari and retry.";
    }
    if (error.message === "video-metadata-timeout") {
      return "Camera opened but video stream did not initialize in time. Please retry.";
    }

    if (error.name) {
      return `Camera startup failed (${error.name}). Please retry.`;
    }
  }

  return "Camera could not be started. Check permissions and try again.";
};

const waitForVideoMetadata = (video: HTMLVideoElement, timeoutMs = 2500): Promise<void> => {
  if (video.readyState >= 1) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const onLoaded = () => {
      cleanup();
      resolve();
    };

    const onError = () => {
      cleanup();
      reject(new Error("video-metadata-timeout"));
    };

    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("video-metadata-timeout"));
    }, timeoutMs);

    const cleanup = () => {
      window.clearTimeout(timer);
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
    };

    video.addEventListener("loadedmetadata", onLoaded);
    video.addEventListener("error", onError);
  });
};

type RepExercise = Exclude<ExerciseType, "Plank">;

// Thresholds are joint angles in degrees (openness 0-1 for jumping jacks).
// Each rep must travel start -> bottom -> top; the gap between bottom and top
// is the hysteresis band that absorbs landmark jitter.
const REP_CONFIGS: Record<RepExercise, RepCounterConfig> = {
  Squat: { startThreshold: 150, bottomThreshold: 120, topThreshold: 155, minRepMs: 700 },
  Pushup: { startThreshold: 150, bottomThreshold: 120, topThreshold: 150, minRepMs: 600 },
  "Biceps Curl": { startThreshold: 140, bottomThreshold: 105, topThreshold: 138, minRepMs: 600 },
  Lunge: { startThreshold: 150, bottomThreshold: 120, topThreshold: 155, minRepMs: 700 },
  "Jumping Jack": {
    direction: "up",
    startThreshold: 0.3,
    bottomThreshold: 0.7,
    topThreshold: 0.25,
    stableFrames: 1,
    minRepMs: 350,
  },
};

const PLANK_PLACEHOLDER_CONFIG: RepCounterConfig = { startThreshold: 0, bottomThreshold: -1, topThreshold: 1 };

const createCounter = (exercise: ExerciseType): RepCounter =>
  new RepCounter(exercise === "Plank" ? PLANK_PLACEHOLDER_CONFIG : REP_CONFIGS[exercise]);

const PARTIAL_REP_HINTS: Record<RepExercise, string> = {
  Squat: "Partial squat not counted. Sit lower, aiming for thighs near parallel.",
  Pushup: "Partial pushup not counted. Lower your chest until elbows reach about 90 degrees.",
  "Biceps Curl": "Partial curl not counted. Curl all the way up to your shoulder.",
  Lunge: "Partial lunge not counted. Drop your back knee closer to the floor.",
  "Jumping Jack": "Partial jack not counted. Jump wider and bring your arms overhead.",
};

const REP_FORM_TIPS: Record<RepExercise, string> = {
  Squat: "Keep your chest up and knees over ankles.",
  Pushup: "Keep a straight line from shoulders to ankles.",
  "Biceps Curl": "Pin your elbows to your sides.",
  Lunge: "Keep your torso upright.",
  "Jumping Jack": "Reach fully overhead and land with feet wide.",
};

/** UI state (posture, phase, checks) is committed at most this often; counting runs every frame. */
const UI_COMMIT_INTERVAL_MS = 150;
const MAX_REP_HISTORY = 200;
const LANDMARK_HISTORY_FRAMES = 3;

const INITIAL_SMOOTHED_ANGLES = { knee: 170, elbow: 170, back: 170, plank: 170, curl: 170, lunge: 170, jack: 0 };

export interface UsePoseDetectionOptions {
  /** Used to scale calorie estimates. Defaults to 70 kg. */
  bodyWeightKg?: number;
}

export function usePoseDetection(
  selectedExercise: ExerciseType,
  options: UsePoseDetectionOptions = {}
): UsePoseDetectionResult {
  const { bodyWeightKg } = options;
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseRef = useRef<Pose | null>(null);
  const rafRef = useRef<number | null>(null);
  const loopActiveRef = useRef(false);
  const processingFrameRef = useRef(false);
  const repCounterRef = useRef<RepCounter>(createCounter(selectedExercise));
  const startTimeRef = useRef<number | null>(null);
  const frameCounterRef = useRef(0);
  const lastProcessedTsRef = useRef(0);
  const invalidFrameStreakRef = useRef(0);
  const landmarkHistoryRef = useRef<NormalizedLandmarkList[]>([]);
  const lastValidLandmarksRef = useRef<NormalizedLandmarkList | null>(null);
  const previousStabilizedLandmarksRef = useRef<NormalizedLandmarkList | null>(null);
  const lowLightRef = useRef(false);
  const smoothedAnglesRef = useRef({ ...INITIAL_SMOOTHED_ANGLES });
  const pendingUiRef = useRef<{ phase: string; checks: LiveMetricCheck[] }>({ phase: "idle", checks: [] });
  const lastUiCommitTsRef = useRef(0);
  const lastFeedbackTsRef = useRef(0);
  const lastFeedbackTextRef = useRef("");
  const plankAccumulatedMsRef = useRef(0);
  const plankPerfectAccumulatedMsRef = useRef(0);
  const plankPostureSumRef = useRef(0);
  const plankPostureCountRef = useRef(0);
  const plankLastTickTsRef = useRef<number | null>(null);
  const plankRunningRef = useRef(false);
  const plankCompletionAnnouncedRef = useRef(false);
  const plankMilestoneCountRef = useRef(0);

  const [cameraOn, setCameraOn] = useState(false);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [reps, setReps] = useState(0);
  const [repHistory, setRepHistory] = useState<RepRecord[]>([]);
  const [partialReps, setPartialReps] = useState(0);
  const [postureScore, setPostureScore] = useState(0);
  const [plankAvgPosture, setPlankAvgPosture] = useState(0);
  const [feedbackHistory, setFeedbackHistory] = useState<LiveFeedback[]>([
    { text: "Start camera to begin real-time AI coaching.", type: "success" },
  ]);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [distanceStatus, setDistanceStatus] = useState<DistanceStatus>("adjusting");
  const [distanceHint, setDistanceHint] = useState("Stand 1.5 to 2.5m (5 to 8ft) from camera.");
  const [currentPhase, setCurrentPhase] = useState("idle");
  const [liveChecks, setLiveChecks] = useState<LiveMetricCheck[]>([]);
  const [activePlankTime, setActivePlankTime] = useState(0);
  const [perfectPlankTime, setPerfectPlankTime] = useState(0);
  const [perfectPlankTimeAtTarget, setPerfectPlankTimeAtTarget] = useState(0);
  const [totalElapsedTime, setTotalElapsedTime] = useState(0);
  const [targetPlankTime, setTargetPlankTimeState] = useState(60);
  const [plankCompleted, setPlankCompleted] = useState(false);

  const calories = useMemo(
    () => estimateCalories(selectedExercise, reps, activePlankTime, bodyWeightKg),
    [activePlankTime, bodyWeightKg, reps, selectedExercise]
  );

  const repSummary = useMemo(() => summarizeReps(repHistory), [repHistory]);

  // Session-level form score: mean rep score, or mean posture while holding a plank.
  const sessionFormScore = selectedExercise === "Plank" ? plankAvgPosture : repSummary.avgScore;

  const setTargetPlankTime = useCallback((seconds: number) => {
    const safeSeconds = clamp(Math.round(seconds), 10, 600);
    setTargetPlankTimeState(safeSeconds);
  }, []);

  const playCompletionBeep = useCallback(() => {
    try {
      const AudioCtx = window.AudioContext ?? (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;

      const audioCtx = new AudioCtx();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();

      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);

      gainNode.gain.setValueAtTime(0.0001, audioCtx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.16, audioCtx.currentTime + 0.02);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.26);

      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);

      oscillator.start(audioCtx.currentTime);
      oscillator.stop(audioCtx.currentTime + 0.27);
      oscillator.onended = () => {
        void audioCtx.close();
      };
    } catch {
      // Ignore beep errors to avoid interrupting workout tracking.
    }
  }, []);

  /**
   * Coaching cues are rate-limited to one every 3s so the panel stays readable.
   * Priority messages (rep results) bypass the limit so no counted rep goes unannounced.
   */
  const pushFeedback = useCallback((feedback: LiveFeedback, priority = false) => {
    const now = Date.now();
    if (!priority) {
      if (now - lastFeedbackTsRef.current < 3000) return;
      if (feedback.text === lastFeedbackTextRef.current && now - lastFeedbackTsRef.current < 8000) return;
    }

    lastFeedbackTsRef.current = now;
    lastFeedbackTextRef.current = feedback.text;

    setFeedbackHistory((prev) => [feedback, ...prev].slice(0, 6));
  }, []);

  const resetSession = useCallback(() => {
    repCounterRef.current = createCounter(selectedExercise);
    startTimeRef.current = cameraOn ? performance.now() : null;
    landmarkHistoryRef.current = [];
    lastValidLandmarksRef.current = null;
    previousStabilizedLandmarksRef.current = null;
    invalidFrameStreakRef.current = 0;
    smoothedAnglesRef.current = { ...INITIAL_SMOOTHED_ANGLES };
    pendingUiRef.current = { phase: "idle", checks: [] };
    plankAccumulatedMsRef.current = 0;
    plankPerfectAccumulatedMsRef.current = 0;
    plankPostureSumRef.current = 0;
    plankPostureCountRef.current = 0;
    plankLastTickTsRef.current = null;
    plankRunningRef.current = false;
    plankCompletionAnnouncedRef.current = false;
    plankMilestoneCountRef.current = 0;
    setReps(0);
    setRepHistory([]);
    setPartialReps(0);
    setPostureScore(0);
    setPlankAvgPosture(0);
    setElapsedSeconds(0);
    setTotalElapsedTime(0);
    setActivePlankTime(0);
    setPerfectPlankTime(0);
    setPerfectPlankTimeAtTarget(0);
    setPlankCompleted(false);
    setCurrentPhase("idle");
    setLiveChecks([]);
    setDistanceStatus("adjusting");
    setDistanceHint("Stand 1.5 to 2.5m (5 to 8ft) from camera.");
    setFeedbackHistory([{ text: `Tracking reset for ${selectedExercise}.`, type: "success" }]);
  }, [cameraOn, selectedExercise]);

  const stopCamera = useCallback(() => {
    loopActiveRef.current = false;

    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }

    setCameraOn(false);
    processingFrameRef.current = false;
    plankRunningRef.current = false;
    plankLastTickTsRef.current = null;
  }, []);

  const handleCounterEvent = useCallback(
    (event: RepEvent, exercise: RepExercise) => {
      if (!event) return;

      if (event.type === "partial") {
        setPartialReps((prev) => prev + 1);
        pushFeedback({ text: PARTIAL_REP_HINTS[exercise], type: "warning" }, true);
        return;
      }

      const { rep } = event;
      setReps(rep.index);
      setRepHistory((prev) => [...prev, rep].slice(-MAX_REP_HISTORY));
      if (IS_DEV) console.debug("[PoseDebug] Rep counted", { exercise, ...rep });

      if (rep.score >= 90) {
        pushFeedback({ text: `Rep ${rep.index}: excellent form (${rep.score}%).`, type: "success" }, true);
      } else if (rep.score >= 75) {
        pushFeedback({ text: `Rep ${rep.index}: good rep (${rep.score}%).`, type: "success" }, true);
      } else {
        pushFeedback(
          { text: `Rep ${rep.index} counted at ${rep.score}% form. ${REP_FORM_TIPS[exercise]}`, type: "warning" },
          true
        );
      }
    },
    [pushFeedback]
  );

  const evaluateExercise = useCallback(
    (landmarks: NormalizedLandmarkList): { feedback: LiveFeedback; posture: number } => {
      const nowMs = performance.now();
      const counter = repCounterRef.current;
      const commitChecks = (phase: string, checks: LiveMetricCheck[]) => {
        pendingUiRef.current = { phase, checks };
      };

      const lShoulder = landmarks[LANDMARKS.leftShoulder];
      const rShoulder = landmarks[LANDMARKS.rightShoulder];
      const lElbow = landmarks[LANDMARKS.leftElbow];
      const rElbow = landmarks[LANDMARKS.rightElbow];
      const lWrist = landmarks[LANDMARKS.leftWrist];
      const rWrist = landmarks[LANDMARKS.rightWrist];
      const lHip = landmarks[LANDMARKS.leftHip];
      const rHip = landmarks[LANDMARKS.rightHip];
      const lKnee = landmarks[LANDMARKS.leftKnee];
      const rKnee = landmarks[LANDMARKS.rightKnee];
      const lAnkle = landmarks[LANDMARKS.leftAnkle];
      const rAnkle = landmarks[LANDMARKS.rightAnkle];

      const video = videoRef.current;
      const width = video?.videoWidth || 1280;
      const height = video?.videoHeight || 720;
      const aspectRatio = width / height;
      const xScale = aspectRatio / 1.777;
      const dy = (Math.abs(lHip.y - lShoulder.y) + Math.abs(rHip.y - rShoulder.y)) / 2;
      const dx = ((Math.abs(lHip.x - lShoulder.x) + Math.abs(rHip.x - rShoulder.x)) / 2) * xScale;
      const isVerticalTorso = dy > dx * 0.85;

      const leftKneeAngleRaw = angleFromThreePoints(lHip, lKnee, lAnkle, aspectRatio);
      const rightKneeAngleRaw = angleFromThreePoints(rHip, rKnee, rAnkle, aspectRatio);
      const leftElbowAngleRaw = angleFromThreePoints(lShoulder, lElbow, lWrist, aspectRatio);
      const rightElbowAngleRaw = angleFromThreePoints(rShoulder, rElbow, rWrist, aspectRatio);

      const leftLegVis = (lHip.visibility ?? 0) + (lKnee.visibility ?? 0) + (lAnkle.visibility ?? 0);
      const rightLegVis = (rHip.visibility ?? 0) + (rKnee.visibility ?? 0) + (rAnkle.visibility ?? 0);
      const leftArmVis = (lShoulder.visibility ?? 0) + (lElbow.visibility ?? 0) + (lWrist.visibility ?? 0);
      const rightArmVis = (rShoulder.visibility ?? 0) + (rElbow.visibility ?? 0) + (rWrist.visibility ?? 0);

      // Prefer the clearly more visible side (side-on views); otherwise average both.
      const kneeRaw =
        Math.abs(leftLegVis - rightLegVis) > 0.3
          ? leftLegVis > rightLegVis
            ? leftKneeAngleRaw
            : rightKneeAngleRaw
          : (leftKneeAngleRaw + rightKneeAngleRaw) / 2;

      const elbowRaw =
        Math.abs(leftArmVis - rightArmVis) > 0.3
          ? leftArmVis > rightArmVis
            ? leftElbowAngleRaw
            : rightElbowAngleRaw
          : (leftElbowAngleRaw + rightElbowAngleRaw) / 2;

      const backRaw =
        (angleFromThreePoints(lShoulder, lHip, lKnee, aspectRatio) + angleFromThreePoints(rShoulder, rHip, rKnee, aspectRatio)) / 2;

      const smoothed = smoothedAnglesRef.current;
      smoothed.knee = smoothValue(smoothed.knee, kneeRaw);
      smoothed.elbow = smoothValue(smoothed.elbow, elbowRaw);
      smoothed.back = smoothValue(smoothed.back, backRaw);

      const avgKneeAngle = smoothed.knee;
      const avgElbowAngle = smoothed.elbow;
      const backAngle = smoothed.back;

      const kneeAlignmentDelta = ((Math.abs(lKnee.x - lAnkle.x) + Math.abs(rKnee.x - rAnkle.x)) / 2) * xScale;

      if (selectedExercise === "Squat") {
        if (!isVerticalTorso) {
          counter.abortCycle();
          commitChecks("squat:waiting", []);
          return { feedback: { text: "Please stand up to perform squats.", type: "warning" }, posture: 0 };
        }
        const cfg = REP_CONFIGS.Squat;
        const depthScore = scoreInRange(avgKneeAngle, 85, 125, 50);
        const backScore = scoreFromRange(backAngle, 155, 55);
        const kneeAlignmentScore = clamp(100 - (kneeAlignmentDelta / 0.24) * 100, 0, 100);
        const posture = Math.round(depthScore * 0.45 + backScore * 0.35 + kneeAlignmentScore * 0.2);

        const { phase, event } = counter.update(avgKneeAngle, posture, nowMs);
        handleCounterEvent(event, "Squat");

        commitChecks(`squat:${phase}`, [
          { label: "Squat depth (knee angle)", current: `${avgKneeAngle.toFixed(0)}°`, threshold: `<= ${cfg.bottomThreshold}° at bottom`, pass: avgKneeAngle <= cfg.bottomThreshold },
          { label: "Full stand", current: `${avgKneeAngle.toFixed(0)}°`, threshold: `>= ${cfg.topThreshold}° to finish rep`, pass: avgKneeAngle >= cfg.topThreshold },
          { label: "Back angle", current: `${backAngle.toFixed(0)}°`, threshold: "> 145°", pass: backAngle > 145 },
          { label: "Knee tracking", current: `${kneeAlignmentScore.toFixed(0)}%`, threshold: ">= 50%", pass: kneeAlignmentScore >= 50 },
          { label: "Posture score", current: `${posture}%`, threshold: ">= 75%", pass: posture >= 75 },
        ]);

        let feedbackText = "Ready. Squat down with controlled form.";
        let feedbackType: FeedbackType = "success";
        if (phase === "top" && backAngle < 145) {
          feedbackText = "Keep your back straighter while standing.";
          feedbackType = "warning";
        } else if (phase === "descending") {
          feedbackText = backAngle < 140 ? "Keep your back straight as you lower." : "Lowering... aim for thighs parallel to the floor.";
          feedbackType = backAngle < 140 ? "warning" : "success";
        } else if (phase === "bottom") {
          feedbackText = "Good depth! Drive through your heels to stand up.";
        } else if (phase === "ascending") {
          feedbackText = backAngle < 140 ? "Keep your back straight as you rise." : "Rising... stand up fully to complete the rep.";
          feedbackType = backAngle < 140 ? "warning" : "success";
        }
        if (kneeAlignmentScore < 50 && feedbackType === "success") {
          feedbackText = "Keep knees aligned over your ankles.";
          feedbackType = "warning";
        }
        return { feedback: { text: feedbackText, type: feedbackType }, posture };
      }

      if (selectedExercise === "Pushup") {
        if (isVerticalTorso) {
          counter.abortCycle();
          commitChecks("pushup:waiting", []);
          return { feedback: { text: "Get into a horizontal pushup position. Standing posture does not count.", type: "warning" }, posture: 0 };
        }
        const cfg = REP_CONFIGS.Pushup;
        const leftFootPoint = preferVisiblePoint(lAnkle, lKnee);
        const rightFootPoint = preferVisiblePoint(rAnkle, rKnee);
        const plankLineRaw =
          (angleFromThreePoints(lShoulder, lHip, leftFootPoint, aspectRatio) + angleFromThreePoints(rShoulder, rHip, rightFootPoint, aspectRatio)) / 2;
        smoothed.plank = smoothValue(smoothed.plank, plankLineRaw);
        const plankLineAngle = smoothed.plank;
        const lineScore = scoreInRange(plankLineAngle, 150, 180, 35);
        const elbowDepthScore = scoreInRange(avgElbowAngle, 85, 120, 50);
        const posture = Math.round(lineScore * 0.6 + elbowDepthScore * 0.4);

        const { phase, event } = counter.update(avgElbowAngle, posture, nowMs);
        handleCounterEvent(event, "Pushup");

        commitChecks(`pushup:${phase}`, [
          { label: "Pushup depth (elbow angle)", current: `${avgElbowAngle.toFixed(0)}°`, threshold: `<= ${cfg.bottomThreshold}° at bottom`, pass: avgElbowAngle <= cfg.bottomThreshold },
          { label: "Lockout", current: `${avgElbowAngle.toFixed(0)}°`, threshold: `>= ${cfg.topThreshold}° to finish rep`, pass: avgElbowAngle >= cfg.topThreshold },
          { label: "Body line", current: `${plankLineAngle.toFixed(0)}°`, threshold: "150°-180°", pass: plankLineAngle >= 150 },
          { label: "Posture score", current: `${posture}%`, threshold: ">= 75%", pass: posture >= 75 },
        ]);

        let feedbackText = "Ready. Lower your chest toward the floor.";
        let feedbackType: FeedbackType = "success";
        if (lineScore < 60) {
          feedbackText = "Keep your hips in line with shoulders and ankles.";
          feedbackType = "warning";
        } else if (phase === "descending") {
          feedbackText = "Lowering... keep elbows controlled.";
        } else if (phase === "bottom") {
          feedbackText = "Great depth! Push back up to lock out.";
        } else if (phase === "ascending") {
          feedbackText = "Pushing up... extend your arms fully.";
        }
        return { feedback: { text: feedbackText, type: feedbackType }, posture };
      }

      if (selectedExercise === "Biceps Curl") {
        if (!isVerticalTorso) {
          counter.abortCycle();
          commitChecks("curl:waiting", []);
          return { feedback: { text: "Please stand up to perform biceps curls.", type: "warning" }, posture: 0 };
        }
        const cfg = REP_CONFIGS["Biceps Curl"];
        // Track the arm that is curling: the smaller elbow angle among visible arms.
        // This counts both simultaneous and alternating curls.
        const leftVisible = leftArmVis >= 1.2;
        const rightVisible = rightArmVis >= 1.2;
        const curlRaw =
          leftVisible && rightVisible
            ? Math.min(leftElbowAngleRaw, rightElbowAngleRaw)
            : leftVisible
              ? leftElbowAngleRaw
              : rightVisible
                ? rightElbowAngleRaw
                : elbowRaw;
        smoothed.curl = smoothValue(smoothed.curl, curlRaw);
        const curlAngle = smoothed.curl;

        const armPathDelta = ((Math.abs(lShoulder.x - lElbow.x) + Math.abs(rShoulder.x - rElbow.x)) / 2) * xScale;
        const elbowTuckScore = clamp(100 - (armPathDelta / 0.14) * 100, 0, 100);
        const rangeScore = scoreInRange(curlAngle, 40, 110, 70);
        const posture = Math.round(elbowTuckScore * 0.55 + rangeScore * 0.45);

        const { phase, event } = counter.update(curlAngle, posture, nowMs);
        handleCounterEvent(event, "Biceps Curl");

        commitChecks(`curl:${phase}`, [
          { label: "Curl top (elbow angle)", current: `${curlAngle.toFixed(0)}°`, threshold: `<= ${cfg.bottomThreshold}° at top`, pass: curlAngle <= cfg.bottomThreshold },
          { label: "Arm extension", current: `${curlAngle.toFixed(0)}°`, threshold: `>= ${cfg.topThreshold}° to finish rep`, pass: curlAngle >= cfg.topThreshold },
          { label: "Elbow tuck", current: `${elbowTuckScore.toFixed(0)}%`, threshold: ">= 60%", pass: elbowTuckScore >= 60 },
          { label: "Posture score", current: `${posture}%`, threshold: ">= 75%", pass: posture >= 75 },
        ]);

        let feedbackText = "Ready. Curl the weight up.";
        let feedbackType: FeedbackType = "success";
        if (elbowTuckScore < 50) {
          feedbackText = "Keep your elbows tucked close to your torso.";
          feedbackType = "warning";
        } else if (phase === "descending") {
          feedbackText = "Squeeze your biceps and curl up.";
        } else if (phase === "bottom") {
          feedbackText = "Top reached! Lower down with control.";
        } else if (phase === "ascending") {
          feedbackText = "Lowering slowly... extend your arms.";
        }
        return { feedback: { text: feedbackText, type: feedbackType }, posture };
      }

      if (selectedExercise === "Lunge") {
        if (!isVerticalTorso) {
          counter.abortCycle();
          commitChecks("lunge:waiting", []);
          return { feedback: { text: "Please stand up to perform lunges.", type: "warning" }, posture: 0 };
        }
        const cfg = REP_CONFIGS.Lunge;
        smoothed.lunge = smoothValue(smoothed.lunge, Math.min(leftKneeAngleRaw, rightKneeAngleRaw));
        const lungeKnee = smoothed.lunge;
        const depthScore = Math.max(scoreInRange(leftKneeAngleRaw, 75, 110, 35), scoreInRange(rightKneeAngleRaw, 75, 110, 35));
        const trunkScore = scoreFromRange(backAngle, 165, 35);
        const symmetryScore = clamp(100 - (Math.abs(leftKneeAngleRaw - rightKneeAngleRaw) / 70) * 100, 0, 100);
        const posture = Math.round(depthScore * 0.45 + trunkScore * 0.35 + symmetryScore * 0.2);

        const { phase, event } = counter.update(lungeKnee, posture, nowMs);
        handleCounterEvent(event, "Lunge");

        commitChecks(`lunge:${phase}`, [
          { label: "Lunge depth (knee angle)", current: `${lungeKnee.toFixed(0)}°`, threshold: `<= ${cfg.bottomThreshold}° at bottom`, pass: lungeKnee <= cfg.bottomThreshold },
          { label: "Return to stand", current: `${lungeKnee.toFixed(0)}°`, threshold: `>= ${cfg.topThreshold}° to finish rep`, pass: lungeKnee >= cfg.topThreshold },
          { label: "Torso angle", current: `${backAngle.toFixed(0)}°`, threshold: "> 145°", pass: backAngle > 145 },
          { label: "Posture score", current: `${posture}%`, threshold: ">= 75%", pass: posture >= 75 },
        ]);

        let feedbackText = "Ready. Step into a lunge.";
        let feedbackType: FeedbackType = "success";
        if (backAngle < 140) {
          feedbackText = "Keep your chest up and torso upright.";
          feedbackType = "warning";
        } else if (phase === "descending") {
          feedbackText = "Lowering... drop your back knee toward the floor.";
        } else if (phase === "bottom") {
          feedbackText = "Good depth! Push back to standing.";
        } else if (phase === "ascending") {
          feedbackText = "Pushing back... stand up fully.";
        }
        return { feedback: { text: feedbackText, type: feedbackType }, posture };
      }

      if (selectedExercise === "Jumping Jack") {
        if (!isVerticalTorso) {
          counter.abortCycle();
          commitChecks("jumping-jack:waiting", []);
          return { feedback: { text: "Please stand up to perform jumping jacks.", type: "warning" }, posture: 0 };
        }
        // Scale-invariant openness: arm raise angle plus ankle spread relative to
        // shoulder width, so counting does not depend on distance from the camera.
        const shoulderWidth = Math.max(Math.abs(lShoulder.x - rShoulder.x) * xScale, 0.02);
        const ankleRatio = (Math.abs(lAnkle.x - rAnkle.x) * xScale) / shoulderWidth;
        const armAngle =
          (angleFromThreePoints(lHip, lShoulder, lWrist, aspectRatio) + angleFromThreePoints(rHip, rShoulder, rWrist, aspectRatio)) / 2;
        const armOpen = clamp((armAngle - 35) / 110, 0, 1);
        const legOpen = clamp((ankleRatio - 0.7) / 0.9, 0, 1);
        smoothed.jack = smoothValue(smoothed.jack, armOpen * 0.5 + legOpen * 0.5, 0.65);
        const openness = smoothed.jack;

        const armScore = scoreInRange(armAngle, 140, 180, 60);
        const legScore = scoreInRange(ankleRatio, 1.4, 3, 0.8);
        const trunkScore = scoreFromRange(backAngle, 170, 40);
        const posture = Math.round(armScore * 0.4 + legScore * 0.4 + trunkScore * 0.2);

        const cfg = REP_CONFIGS["Jumping Jack"];
        const { phase, event } = counter.update(openness, posture, nowMs);
        handleCounterEvent(event, "Jumping Jack");

        commitChecks(`jumping-jack:${phase}`, [
          { label: "Arms overhead", current: `${armAngle.toFixed(0)}°`, threshold: ">= 140°", pass: armAngle >= 140 },
          { label: "Feet width", current: `${ankleRatio.toFixed(2)}x shoulders`, threshold: ">= 1.4x", pass: ankleRatio >= 1.4 },
          { label: "Openness", current: `${Math.round(openness * 100)}%`, threshold: `>= ${cfg.bottomThreshold * 100}% open, <= ${cfg.topThreshold * 100}% closed`, pass: openness >= cfg.bottomThreshold || openness <= cfg.topThreshold },
          { label: "Posture score", current: `${posture}%`, threshold: ">= 75%", pass: posture >= 75 },
        ]);

        let feedbackText = "Ready. Jump wide and raise your arms.";
        let feedbackType: FeedbackType = "success";
        if (phase === "bottom") {
          feedbackText = "Good open position! Jump back together.";
        } else if (phase === "descending" && openness < cfg.bottomThreshold) {
          feedbackText = "Jump wider and raise your arms higher.";
          feedbackType = "warning";
        }
        return { feedback: { text: feedbackText, type: feedbackType }, posture };
      }

      if (selectedExercise === "Plank") {
        if (isVerticalTorso) {
          plankRunningRef.current = false;
          plankLastTickTsRef.current = null;
          commitChecks("plank:paused", []);
          return { feedback: { text: "Get into a horizontal plank position. Standing posture does not count.", type: "warning" }, posture: 0 };
        }
        const leftFootPoint = preferVisiblePoint(lAnkle, lKnee);
        const rightFootPoint = preferVisiblePoint(rAnkle, rKnee);
        const plankLineRaw =
          (angleFromThreePoints(lShoulder, lHip, leftFootPoint, aspectRatio) + angleFromThreePoints(rShoulder, rHip, rightFootPoint, aspectRatio)) / 2;
        smoothed.plank = smoothValue(smoothed.plank, plankLineRaw);
        const plankLineAngle = smoothed.plank;
        const lineScore = scoreInRange(plankLineAngle, 155, 180, 25);
        const shoulderStackScore = clamp(100 - (Math.abs(lShoulder.x - lElbow.x) + Math.abs(rShoulder.x - rElbow.x)) * xScale * 160, 0, 100);
        const torsoYDelta = (Math.abs(lShoulder.y - lHip.y) + Math.abs(rShoulder.y - rHip.y)) / 2;
        const legYDelta = (Math.abs(lHip.y - leftFootPoint.y) + Math.abs(rHip.y - rightFootPoint.y)) / 2;
        const horizontalScore = Math.round(
          clamp(100 - (torsoYDelta / 0.22) * 100, 0, 100) * 0.6 +
          clamp(100 - (legYDelta / 0.28) * 100, 0, 100) * 0.4
        );
        const posture = Math.round(lineScore * 0.5 + shoulderStackScore * 0.25 + horizontalScore * 0.25);
        const postureValid = posture >= 55 && horizontalScore >= 35;

        if (postureValid) {
          if (!plankRunningRef.current || plankLastTickTsRef.current === null) {
            plankRunningRef.current = true;
            plankLastTickTsRef.current = nowMs;
          } else {
            // Cap per-frame credit so a stalled tab cannot add a large time jump.
            const deltaMs = clamp(nowMs - plankLastTickTsRef.current, 0, 500);
            plankAccumulatedMsRef.current += deltaMs;
            if (posture >= 85) plankPerfectAccumulatedMsRef.current += deltaMs;
            plankLastTickTsRef.current = nowMs;
          }
          plankPostureSumRef.current += posture;
          plankPostureCountRef.current += 1;
        } else {
          plankRunningRef.current = false;
          plankLastTickTsRef.current = null;
        }

        const activeSeconds = plankAccumulatedMsRef.current / 1000;
        commitChecks(plankRunningRef.current ? "plank:holding" : "plank:paused", [
          { label: "Body line", current: `${plankLineAngle.toFixed(0)}°`, threshold: "155°-180°", pass: plankLineAngle >= 155 },
          { label: "Shoulders over elbows", current: `${shoulderStackScore.toFixed(0)}%`, threshold: ">= 60%", pass: shoulderStackScore >= 60 },
          { label: "Horizontal body", current: `${horizontalScore}%`, threshold: ">= 35%", pass: horizontalScore >= 35 },
          { label: "Posture score", current: `${posture}%`, threshold: ">= 55% to run timer", pass: postureValid },
          { label: "Good-form hold", current: `${activeSeconds.toFixed(1)}s`, threshold: `>= ${targetPlankTime}s`, pass: activeSeconds >= targetPlankTime },
        ]);

        if (horizontalScore < 35) return { feedback: { text: "Get into floor plank position. Timer is paused.", type: "warning" }, posture };
        if (lineScore < 65) return { feedback: { text: "Keep your hips aligned with shoulders and ankles.", type: "warning" }, posture };
        if (shoulderStackScore < 60) return { feedback: { text: "Stack your shoulders over your elbows.", type: "warning" }, posture };
        if (!plankCompleted) {
          const remaining = Math.max(Math.ceil(targetPlankTime - activeSeconds), 0);
          return { feedback: { text: `Plank form looks good. Hold for ${remaining}s more.`, type: "success" }, posture };
        }
        return { feedback: { text: "Plank goal completed! Keep holding or log your workout.", type: "success" }, posture };
      }

      commitChecks("tracking", []);
      return {
        feedback: { text: "Tracking posture. Maintain steady controlled movement.", type: "success" },
        posture: Math.round(scoreFromRange(backAngle, 165, 45)),
      };
    },
    [handleCounterEvent, plankCompleted, selectedExercise, targetPlankTime]
  );

  const onPoseResults = useCallback(
    (results: Results) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;

      const width = video.videoWidth || 1280;
      const height = video.videoHeight || 720;
      const aspectRatio = width / height;

      // Resizing a canvas reallocates its buffer, so only do it when the size changes.
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, width, height);

      let activeLandmarks: NormalizedLandmarkList;

      if (results.poseLandmarks && results.poseLandmarks.length > 0) {
        invalidFrameStreakRef.current = 0;
        landmarkHistoryRef.current.push(cloneLandmarks(results.poseLandmarks));
        if (landmarkHistoryRef.current.length > LANDMARK_HISTORY_FRAMES) {
          landmarkHistoryRef.current.shift();
        }

        const averaged = averageLandmarks(landmarkHistoryRef.current);
        const stabilized = applyDeadband(previousStabilizedLandmarksRef.current, averaged);
        previousStabilizedLandmarksRef.current = stabilized;
        lastValidLandmarksRef.current = stabilized;
        activeLandmarks = stabilized;
      } else if (lastValidLandmarksRef.current && invalidFrameStreakRef.current < 8) {
        // Bridge brief detection dropouts visually only; stale landmarks never
        // feed the rep counter or the plank timer.
        invalidFrameStreakRef.current += 1;
        nativeDrawConnectors(ctx, lastValidLandmarksRef.current, getPoseConnections(), { color: "#22c55e", lineWidth: 3 });
        return;
      } else {
        invalidFrameStreakRef.current += 1;
        if (invalidFrameStreakRef.current === 8) {
          landmarkHistoryRef.current = [];
          plankRunningRef.current = false;
          plankLastTickTsRef.current = null;
          setDistanceStatus("adjusting");
          setDistanceHint("No body detected. Step into the frame.");
          pushFeedback({ text: "No body detected. Step back into the camera view.", type: "warning" });
        }
        return;
      }

      const distance = getDistanceStatus(selectedExercise, activeLandmarks, aspectRatio);
      setDistanceStatus(distance.status);
      setDistanceHint(distance.hint);

      const connections = getPoseConnections();
      if (hasMediaPipeDrawing) {
        try {
          mpDrawingUtils!.drawConnectors(ctx, activeLandmarks, connections, { color: "#22c55e", lineWidth: 3 });
          mpDrawingUtils!.drawLandmarks(ctx, activeLandmarks, { color: "#14b8a6", lineWidth: 2, radius: 3 });
        } catch {
          nativeDrawConnectors(ctx, activeLandmarks, connections, { color: "#22c55e", lineWidth: 3 });
          nativeDrawLandmarks(ctx, activeLandmarks, { color: "#14b8a6", lineWidth: 2, radius: 3 });
        }
      } else {
        nativeDrawConnectors(ctx, activeLandmarks, connections, { color: "#22c55e", lineWidth: 3 });
        nativeDrawLandmarks(ctx, activeLandmarks, { color: "#14b8a6", lineWidth: 2, radius: 3 });
      }

      const visibilityThreshold =
        selectedExercise === "Biceps Curl"
          ? lowLightRef.current ? 0.25 : 0.35
          : lowLightRef.current ? 0.3 : 0.45;

      const now = performance.now();
      const commitUi = now - lastUiCommitTsRef.current >= UI_COMMIT_INTERVAL_MS;

      if (!hasExerciseVisibility(selectedExercise, activeLandmarks, visibilityThreshold)) {
        // Unreliable joints: pause counting/timing instead of guessing.
        plankRunningRef.current = false;
        plankLastTickTsRef.current = null;
        const visibilityHelpText =
          selectedExercise === "Biceps Curl"
            ? "Adjust so shoulders, elbows, and wrists stay visible."
            : selectedExercise === "Pushup" || selectedExercise === "Plank"
              ? "Adjust so shoulders, hips, and knees are visible on one side."
              : selectedExercise === "Jumping Jack"
                ? "Adjust so both body sides (arms, hips, legs) stay visible."
                : "Adjust so one full body side (shoulder to ankle) is visible.";
        pushFeedback({ text: visibilityHelpText, type: "warning" });
        if (commitUi) {
          lastUiCommitTsRef.current = now;
          setCurrentPhase("waiting:visibility");
        }
        return;
      }

      const { posture, feedback } = evaluateExercise(activeLandmarks);
      pushFeedback(feedback);

      if (commitUi) {
        lastUiCommitTsRef.current = now;
        setPostureScore(posture);
        setCurrentPhase(pendingUiRef.current.phase);
        setLiveChecks(pendingUiRef.current.checks);
      }
    },
    [evaluateExercise, pushFeedback, selectedExercise]
  );

  const startDetectionLoop = useCallback(() => {
    const pose = poseRef.current;
    const video = videoRef.current;
    if (!pose || !video) return;

    loopActiveRef.current = true;

    const loop = async () => {
      if (!loopActiveRef.current || !videoRef.current || !poseRef.current) return;

      const now = performance.now();
      const targetIntervalMs = 1000 / 24;
      if (now - lastProcessedTsRef.current < targetIntervalMs) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      if (!processingFrameRef.current) {
        processingFrameRef.current = true;
        lastProcessedTsRef.current = now;
        try {
          const video = videoRef.current;
          if (video && video.readyState >= 2) {
            frameCounterRef.current += 1;
            const needsBrightnessSample = frameCounterRef.current % 12 === 0;

            if (!needsBrightnessSample && !lowLightRef.current) {
              // Normal path: hand the video frame straight to MediaPipe (no canvas copy).
              await poseRef.current.send({ image: video });
            } else {
              if (!processingCanvasRef.current) {
                processingCanvasRef.current = document.createElement("canvas");
              }
              const processingCanvas = processingCanvasRef.current;
              const width = video.videoWidth || 640;
              const height = video.videoHeight || 480;
              if (processingCanvas.width !== width) processingCanvas.width = width;
              if (processingCanvas.height !== height) processingCanvas.height = height;

              const pctx = processingCanvas.getContext("2d", { willReadFrequently: true });
              if (pctx) {
                pctx.filter = "none";
                pctx.drawImage(video, 0, 0, width, height);

                if (needsBrightnessSample) {
                  const isLowLight = estimateFrameBrightness(pctx, width, height) < 60;
                  if (isLowLight !== lowLightRef.current) {
                    lowLightRef.current = isLowLight;
                    if (isLowLight) {
                      pushFeedback({ text: "Low light detected. Turn on a light for better tracking.", type: "warning" });
                      poseRef.current.setOptions({ minDetectionConfidence: 0.35, minTrackingConfidence: 0.35 });
                    } else {
                      pushFeedback({ text: "Lighting improved. Tracking accuracy should be better now.", type: "success" });
                      poseRef.current.setOptions({ minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
                    }
                  }
                }

                if (lowLightRef.current) {
                  pctx.filter = "brightness(1.35) contrast(1.2)";
                  pctx.drawImage(video, 0, 0, width, height);
                  pctx.filter = "none";
                }

                await poseRef.current.send({ image: processingCanvas });
              }
            }
          }
        } catch {
          setErrorMessage("Unable to process camera frames. Please retry.");
        } finally {
          processingFrameRef.current = false;
        }
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
  }, [pushFeedback]);

  const ensurePose = useCallback(async (warmupImage?: HTMLVideoElement | HTMLCanvasElement) => {
    if (poseRef.current) return;

    const diagnostics: string[] = [];
    const mpWindow = window as unknown as { Pose?: new (config: { locateFile: (file: string) => string }) => Pose };

    for (const baseUrl of POSE_ASSET_BASE_URLS) {
      try {
        diagnostics.push(`\n> Trying: ${baseUrl}`);
        
        // Pre-check: verify critical assets exist before instantiating Pose
        const criticalAssets = ["pose_web.binarypb", "pose_solution_wasm_bin.js"];
        for (const asset of criticalAssets) {
          const testUrl = `${baseUrl}/${asset}`;
          try {
            const response = await fetch(testUrl, { method: "HEAD" });
            const contentLength = response.headers.get("content-length");
            const contentType = response.headers.get("content-type");
            diagnostics.push(
              `  ${asset}: ${response.status} | Type: ${contentType || "unknown"} | Size: ${contentLength || "unknown"}`
            );
            
            if (!response.ok) {
              throw new Error(`Asset HTTP error: ${response.status}`);
            }
          } catch (fetchErr) {
            diagnostics.push(`  ${asset}: [FAIL] Fetch failed - ${fetchErr instanceof Error ? fetchErr.message : String(fetchErr)}`);
            throw fetchErr;
          }
        }
        
        diagnostics.push(`[OK] Assets verified at ${baseUrl}`);
        diagnostics.push(`Checking global Pose availability...`);
        
        // Check if Pose is available before trying to instantiate
        if (typeof mpWindow.Pose === "undefined") {
          diagnostics.push("[WARN] window.Pose is undefined at this moment");
        } else {
          diagnostics.push(`[OK] window.Pose is defined (type: ${typeof mpWindow.Pose})`);
        }
        
        diagnostics.push(`Calling Pose constructor...`);

        // Prefer the runtime global `window.Pose` (UMD/WASM loader). If it's not
        // available, dynamically import the `@mediapipe/pose` module and use its
        // exported constructor. This avoids relying on a compile-time named import
        // which can be altered by bundlers.
        let pose: Pose | null = null;
        try {
          if (typeof mpWindow.Pose === "function") {
            diagnostics.push("  Using window.Pose global constructor");
            const GlobalCtor = mpWindow.Pose;
            pose = new GlobalCtor({
              locateFile: (file: string) => {
                const url = `${baseUrl}/${file}`;
                diagnostics.push(`  locateFile("${file}") → ${url}`);
                return url;
              },
            });
          } else {
            diagnostics.push("  window.Pose unavailable — dynamically importing @mediapipe/pose");
            const mod = await import("@mediapipe/pose");
            const rawMod = mod as unknown as { Pose?: new (config: { locateFile: (file: string) => string }) => Pose; default?: new (config: { locateFile: (file: string) => string }) => Pose };
            const ImportedCtor = typeof rawMod.Pose === "function" ? rawMod.Pose : rawMod.default ?? rawMod.Pose;
            if (typeof ImportedCtor !== "function") {
              throw new Error("Imported module does not expose a Pose constructor");
            }

            diagnostics.push("  Using imported Pose constructor");
            pose = new ImportedCtor({
              locateFile: (file: string) => {
                const url = `${baseUrl}/${file}`;
                diagnostics.push(`  locateFile("${file}") → ${url}`);
                return url;
              },
            });
          }

          diagnostics.push(`[OK] Pose constructor succeeded`);
        } catch (ctorErr) {
          diagnostics.push(`[FAIL] Pose constructor failed - ${ctorErr instanceof Error ? ctorErr.message : String(ctorErr)}`);
          throw ctorErr;
        }

        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        pose.onResults(onPoseResults);

        // Warm up run to verify model initialization
        if (warmupImage) {
          diagnostics.push("Running warmup with video frame...");
          await pose.send({ image: warmupImage });
          diagnostics.push("✓ Warmup successful");
        }

        poseRef.current = pose;
        if (IS_DEV) console.log("Pose initialized successfully:", diagnostics.join("\n"));
        return;
      } catch (error) {
        const errMsg = error instanceof Error ? error.message : String(error);
        const errName = error instanceof Error ? error.name : "Unknown";
        diagnostics.push(`[FAIL] Error (${errName}): ${errMsg}`);
        
        // Capture stack trace if available
        if (error instanceof Error && error.stack) {
          diagnostics.push(`Stack: ${error.stack.split('\n').slice(0, 3).join(' | ')}`);
        }
      }
    }

    const fullDiagnostics = diagnostics.join("\n");
    console.error("Pose initialization failed:", fullDiagnostics);
    
    // Check what Pose is at runtime
    console.error("Runtime check - typeof window.Pose:", typeof mpWindow.Pose);
    if (typeof mpWindow.Pose === "object") {
      console.error("window.Pose object keys:", Object.keys(mpWindow.Pose || {}).slice(0, 10));
    }
    
    const errorMsg = 
      `Files verification failed.\n${fullDiagnostics}\n\nTry:\n1. Clear browser cache (Ctrl+Shift+Delete)\n2. Do a hard refresh (Ctrl+Shift+R)\n3. Check browser DevTools Network tab for failed requests`;
    throw new Error(`pose-model-load-failed:${errorMsg}`);
  }, [onPoseResults]);

  useEffect(() => {
    if (!poseRef.current) return;
    // Keep MediaPipe callback in sync with latest selected exercise/state.
    poseRef.current.onResults(onPoseResults);
  }, [onPoseResults]);

  const startCamera = useCallback(async (overrideFacingMode?: "user" | "environment") => {
    setErrorMessage(null);

    if (!window.isSecureContext) {
      setErrorMessage("Camera requires a secure connection (HTTPS). Please open the deployed HTTPS URL.");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage("Camera is not supported in this browser.");
      return;
    }

    const activeMode = overrideFacingMode || facingMode;
    const constraintsList: MediaStreamConstraints[] = [
      {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: activeMode,
          frameRate: { ideal: 24, min: 15 },
        },
      },
      {
        video: {
          width: { ideal: 480 },
          height: { ideal: 360 },
          facingMode: activeMode,
          frameRate: { ideal: 20, min: 12 },
        },
      },
      {
        video: {
          width: { ideal: 320 },
          height: { ideal: 240 },
          facingMode: activeMode,
          frameRate: { ideal: 18, min: 10 },
        },
      },
      { video: { facingMode: activeMode } },
      { video: true },
    ];

    try {
      let stream: MediaStream | null = null;
      let lastCameraError: unknown = null;
      for (const constraints of constraintsList) {
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
          break;
        } catch (error) {
          lastCameraError = error;
          stream = null;
        }
      }

      if (!stream) {
        throw lastCameraError ?? new Error("camera-unavailable");
      }

      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) {
        setErrorMessage("Video element is unavailable.");
        return;
      }

      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.setAttribute("playsinline", "true");
      video.srcObject = stream;

      await waitForVideoMetadata(video);

      try {
        await video.play();
      } catch {
        // Some mobile browsers need a second play call after metadata is available.
        try {
          await video.play();
        } catch {
          throw new Error("video-playback-failed");
        }
      }

      await ensurePose(video);
      setCameraOn(true);
      startTimeRef.current = performance.now();
      setElapsedSeconds(0);
      lastProcessedTsRef.current = 0;
      frameCounterRef.current = 0;
      startDetectionLoop();
      pushFeedback({ text: "Camera started. AI detection is live.", type: "success" });
    } catch (error) {
      const inAppBrowserPattern = /(FBAN|FBAV|Instagram|Line\/[\d.]+|MicroMessenger|; wv\)|WebView)/i;
      const inAppHint = inAppBrowserPattern.test(navigator.userAgent)
        ? " If this page is opened in an in-app browser, use the menu and open it in Chrome or Safari."
        : "";

      setErrorMessage(`${getCameraErrorMessage(error)}${inAppHint}`);
      stopCamera();
    }
  }, [ensurePose, pushFeedback, startDetectionLoop, stopCamera, facingMode]);

  useEffect(() => {
    if (!cameraOn) return;

    const timer = window.setInterval(() => {
      if (!startTimeRef.current) return;
      const seconds = Math.floor((performance.now() - startTimeRef.current) / 1000);
      setElapsedSeconds(seconds);
      setTotalElapsedTime(seconds);
    }, 1000);

    return () => window.clearInterval(timer);
  }, [cameraOn]);

  useEffect(() => {
    if (!cameraOn || selectedExercise !== "Plank") return;

    const plankUiTimer = window.setInterval(() => {
      const activeSeconds = plankAccumulatedMsRef.current / 1000;
      const perfectSeconds = plankPerfectAccumulatedMsRef.current / 1000;
      setActivePlankTime(Number(activeSeconds.toFixed(1)));
      setPerfectPlankTime(Number(perfectSeconds.toFixed(1)));
      if (plankPostureCountRef.current > 0) {
        setPlankAvgPosture(Math.round(plankPostureSumRef.current / plankPostureCountRef.current));
      }

      // Goals are earned only by good-form hold time, never by wall-clock time.
      const milestoneCount = Math.floor(activeSeconds / targetPlankTime);
      if (milestoneCount >= 1) {
        setPlankCompleted(true);
        if (!plankCompletionAnnouncedRef.current) {
          plankCompletionAnnouncedRef.current = true;
          setPerfectPlankTimeAtTarget(Number(perfectSeconds.toFixed(1)));
          pushFeedback({ text: "Goal reached! Perfect plank recorded. You can log workout now.", type: "success" });
        }
      }

      if (milestoneCount > plankMilestoneCountRef.current) {
        const newMilestones = milestoneCount - plankMilestoneCountRef.current;
        plankMilestoneCountRef.current = milestoneCount;

        for (let i = 0; i < newMilestones; i += 1) {
          playCompletionBeep();
        }
      }

    }, 300);

    return () => window.clearInterval(plankUiTimer);
  }, [cameraOn, playCompletionBeep, selectedExercise, targetPlankTime, pushFeedback]);

  useEffect(() => {
    resetSession();
  }, [resetSession, selectedExercise]);

  useEffect(() => {
    return () => {
      stopCamera();
      if (poseRef.current) {
        poseRef.current.close();
        poseRef.current = null;
      }
    };
  }, [stopCamera]);

  const toggleCameraFacing = useCallback(async () => {
    const nextMode = facingMode === "user" ? "environment" : "user";
    setFacingMode(nextMode);
    if (cameraOn) {
      stopCamera();
      setTimeout(async () => {
        try {
          await startCamera(nextMode);
        } catch (err) {
          console.error("Failed to restart camera after toggle:", err);
        }
      }, 300);
    }
  }, [facingMode, cameraOn, stopCamera, startCamera]);

  return {
    videoRef,
    canvasRef,
    cameraOn,
    errorMessage,
    reps,
    repHistory,
    partialReps,
    repSummary,
    sessionFormScore,
    postureScore,
    calories,
    feedbackHistory,
    elapsedSeconds,
    distanceStatus,
    distanceHint,
    currentPhase,
    liveChecks,
    activePlankTime,
    perfectPlankTime,
    perfectPlankTimeAtTarget,
    totalElapsedTime,
    targetPlankTime,
    plankCompleted,
    setTargetPlankTime,
    startCamera,
    stopCamera,
    resetSession,
    facingMode,
    toggleCameraFacing,
  };
}
