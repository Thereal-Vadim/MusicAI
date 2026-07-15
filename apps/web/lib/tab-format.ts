export type SourceType = "upload" | "youtube";
export type GuitarPart = "combined" | "solo" | "rhythm";

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

export interface NoteTechnique {
  palm_mute?: boolean;
  slide?: "up" | "down" | "into_from_below" | "into_from_above";
  vibrato?: boolean;
  tie?: boolean;
  ghost?: boolean;
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
  technique?: NoteTechnique | null;
}

export interface TabMeasure {
  index: number;
  start_ms: number;
  confidence: number;
  chord?: string | null;
  time_signature?: [number, number] | null;
  section?: string | null;
  tempo_bpm?: number | null;
  notes: TabNote[];
}

export interface QualityMeta {
  notes_total: number;
  snapped_count: number;
  high_confidence_count: number;
  conflict_count: number;
  snapped_pct: number;
  high_confidence_pct: number;
  conflict_pct: number;
  mean_overall: number;
  key_confidence: number;
  reference_url?: string | null;
  reference_match_pct?: number | null;
  reference_mismatch_count?: number;
}

export interface TabDocument {
  version: "1";
  job_id?: string | null;
  meta: {
    title?: string | null;
    artist?: string | null;
    album?: string | null;
    bpm: number;
    key?: string | null;
    mode?: string | null;
    tuning: string[];
    guitar_part?: GuitarPart;
    source: {
      type: SourceType;
      url?: string | null;
      youtube_id?: string | null;
      filename?: string | null;
    };
    pipeline_version?: string;
    overall_confidence: number;
    quality?: QualityMeta | null;
  };
  tracks: Array<{
    instrument: "guitar";
    name?: string | null;
    measures: TabMeasure[];
  }>;
}

export interface LiveLogLine {
  ts?: string;
  level?: string;
  stage?: string;
  job_id?: string;
  message: string;
}

export interface StageProgress {
  name: string;
  target_pct: number;
  duration_sec?: number | null;
}

export interface JobStatus {
  id: string;
  status: string;
  stage: string;
  stage_detail: string;
  progress_pct: number;
  elapsed_sec?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
  stage_durations?: Record<string, number>;
  stages?: StageProgress[];
  draft_id?: string | null;
}

export interface StemPreviewItem {
  id: string;
  label: string;
  filename: string;
}

export interface JobStemsInfo {
  guitar_part: GuitarPart;
  items: StemPreviewItem[];
}

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const PIPELINE_STAGES = [
  "ingest",
  "separate",
  "guitar_demix",
  "demix_validate",
  "transcribe",
  "vision",
  "fusion",
  "judge",
  "draft",
  "done",
] as const;

export const STAGE_TARGET_PCT: Record<string, number> = {
  ingest: 8,
  separate: 16,
  guitar_demix: 20,
  demix_validate: 22,
  transcribe: 42,
  vision: 62,
  fusion: 72,
  judge: 82,
  draft: 95,
  done: 100,
};

export function isConflictNote(note: TabNote): boolean {
  if (note.flags.includes("reference_mismatch")) return true;
  if (note.judge.flags.includes("unplayable_position")) return true;
  if (note.judge.flags.includes("too_many_simultaneous_notes")) return true;
  if (note.judge.flags.includes("chord_span_exceeded")) return true;
  if (note.judge.flags.includes("temporal_violation")) return true;
  if (!note.judge.in_scale && !note.judge.snapped) return true;
  return false;
}

export function conflictReason(note: TabNote): string {
  if (note.flags.includes("reference_mismatch")) return "reference_mismatch";
  if (note.judge.snap_reason) return note.judge.snap_reason;
  if (note.flags.includes("audio_vision_mismatch")) return "audio_vision_mismatch";
  if (note.judge.flags.includes("out_of_harmony")) return "out_of_harmony";
  if (note.judge.flags.includes("auto_corrected")) return "auto_corrected";
  return "low_confidence";
}
