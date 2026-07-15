import { describe, expect, it } from "vitest";
import { conflictReason, isConflictNote, type TabNote } from "../lib/tab-format";

const baseNote: TabNote = {
  id: "1",
  pitch: "E4",
  start_ms: 0,
  duration_ms: 200,
  string: 1,
  fret: 0,
  confidence: { audio: 0.9, vision: 0.8, judge: 0.95, overall: 0.88 },
  judge: { in_scale: true, in_chord: true, snapped: false, flags: [] },
  flags: [],
};

describe("tab conflict helpers", () => {
  it("detects snapped notes as conflicts", () => {
    const note = {
      ...baseNote,
      judge: { ...baseNote.judge, snapped: true, snap_reason: "out_of_harmony_low_audio_conf" },
    };
    expect(isConflictNote(note)).toBe(true);
    expect(conflictReason(note)).toBe("out_of_harmony_low_audio_conf");
  });

  it("detects audio vision mismatch", () => {
    const note = { ...baseNote, flags: ["audio_vision_mismatch"] };
    expect(isConflictNote(note)).toBe(true);
    expect(conflictReason(note)).toBe("audio_vision_mismatch");
  });
});
