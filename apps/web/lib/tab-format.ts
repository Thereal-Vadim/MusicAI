export type SourceType = "upload" | "youtube";

export interface NoteConfidence {
  audio: number;
  vision: number;
  judge: number;
  overall: number;
}

export interface JudgeInfo {
  in_scale: boolean;
  in_chord: boolean;
  snapped: boolean;
  snap_reason?: string | null;
  flags: string[];
}

export interface TabNote {
  id: string;
  pitch: string;
  original_pitch?: string | null;
  start_ms: number;
  duration_ms: number;
  string: number;
  fret: number;
  pitch_midi?: number;
  confidence: NoteConfidence;
  judge: JudgeInfo;
  flags: string[];
}

export interface TabMeasure {
  index: number;
  start_ms: number;
  confidence: number;
  chord?: string | null;
  notes: TabNote[];
}

export interface TabDocument {
  version: "1";
  job_id?: string | null;
  meta: {
    bpm: number;
    key?: string | null;
    mode?: string | null;
    tuning: string[];
    source: {
      type: SourceType;
      url?: string | null;
      youtube_id?: string | null;
      filename?: string | null;
    };
    pipeline_version?: string;
    overall_confidence: number;
  };
  tracks: Array<{
    instrument: "guitar";
    measures: TabMeasure[];
  }>;
}

export interface JobStatus {
  id: string;
  status: string;
  stage: string;
  stage_detail: string;
  draft_id?: string | null;
}

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const PIPELINE_STAGES = [
  "ingest",
  "separate",
  "transcribe",
  "vision",
  "fusion",
  "judge",
  "draft",
  "done",
] as const;

export function isConflictNote(note: TabNote): boolean {
  return (
    note.judge.snapped ||
    note.judge.flags.includes("out_of_harmony") ||
    note.flags.includes("audio_vision_mismatch") ||
    note.confidence.overall < 0.75
  );
}

export function conflictReason(note: TabNote): string {
  if (note.judge.snap_reason) return note.judge.snap_reason;
  if (note.flags.includes("audio_vision_mismatch")) return "audio_vision_mismatch";
  if (note.judge.flags.includes("out_of_harmony")) return "out_of_harmony";
  if (note.judge.flags.includes("auto_corrected")) return "auto_corrected";
  return "low_confidence";
}
