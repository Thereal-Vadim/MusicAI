export type NoteMatchStatus =
  | "match"
  | "pitch_miss"
  | "timing_miss"
  | "fret_miss"
  | "string_miss"
  | "unmatched_ref"
  | "extra_pred";

export interface BenchmarkComparison {
  reference_id: string;
  reference_url?: string | null;
  metrics: {
    pitch_f1: number;
    pitch_accuracy: number;
    fret_accuracy: number;
    string_accuracy: number;
    timing_accuracy: number;
    overall_f1: number;
    matched: number;
    reference_count: number;
    predicted_count: number;
    alignments?: Array<{
      status: NoteMatchStatus;
      ref_start_ms?: number | null;
      pred_start_ms?: number | null;
      pitch_midi?: number | null;
      ref_fret?: number | null;
      pred_fret?: number | null;
      ref_string?: number | null;
      pred_string?: number | null;
      timing_delta_ms?: number | null;
      predicted_note_id?: string | null;
    }>;
  };
  note_statuses: Record<string, NoteMatchStatus>;
}

export const BENCHMARK_STATUS_LABELS: Record<NoteMatchStatus, string> = {
  match: "Точное совпадение",
  pitch_miss: "Неверная высота",
  timing_miss: "Сдвиг по времени",
  fret_miss: "Неверный лад",
  string_miss: "Неверная струна",
  unmatched_ref: "Пропущена (эталон)",
  extra_pred: "Лишняя нота",
};

export const BENCHMARK_STATUS_COLORS: Record<NoteMatchStatus, string> = {
  match: "#2f9e44",
  pitch_miss: "#e03131",
  timing_miss: "#f08c00",
  fret_miss: "#e64980",
  string_miss: "#be4bdb",
  unmatched_ref: "#868e96",
  extra_pred: "#495057",
};
