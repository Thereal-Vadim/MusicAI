"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { TabDocument, TabNote } from "@/lib/tab-format";
import { conflictReason, isConflictNote } from "@/lib/tab-format";
import {
  loadYouTubeApi,
  syncAudioElement,
  syncYouTubePlayer,
  usePlaybackClock,
  type YTPlayer,
} from "@/lib/playback/sync";

interface TabEditorProps {
  document: TabDocument;
  onEditNote: (noteId: string, patch: { string?: number; fret?: number }) => Promise<void>;
}

const STRING_Y = [20, 40, 60, 80, 100, 120];

export function TabEditor({ document, onEditNote }: TabEditorProps) {
  const notes = useMemo(
    () => document.tracks.flatMap((t) => t.measures.flatMap((m) => m.notes)),
    [document]
  );
  const conflictIds = useMemo(
    () => notes.filter(isConflictNote).map((n) => n.id),
    [notes]
  );
  const [conflictIndex, setConflictIndex] = useState(0);
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const { currentMs, playing, play, pause, seek } = usePlaybackClock(0);
  const [ytReady, setYtReady] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const ytPlayerRef = useRef<YTPlayer | null>(null);
  const source = document.meta.source;

  useEffect(() => {
    if (source.type !== "youtube" || !source.youtube_id) {
      setYtReady(false);
      return;
    }
    let cancelled = false;
    setYtReady(false);
    ytPlayerRef.current = null;

    loadYouTubeApi().then(() => {
      if (cancelled || !window.YT?.Player) return;
      new window.YT.Player("yt-player", {
        videoId: source.youtube_id!,
        events: {
          onReady: (event) => {
            if (cancelled) return;
            ytPlayerRef.current = event.target;
            setYtReady(true);
          },
        },
      });
    });

    return () => {
      cancelled = true;
      ytPlayerRef.current = null;
      setYtReady(false);
    };
  }, [source.type, source.youtube_id]);

  useEffect(() => {
    syncAudioElement(audioRef.current, currentMs, playing);
    if (ytReady) {
      syncYouTubePlayer(ytPlayerRef.current, currentMs, playing);
    }
  }, [currentMs, playing, ytReady]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Tab") {
        e.preventDefault();
        if (!conflictIds.length) return;
        const next = (conflictIndex + 1) % conflictIds.length;
        setConflictIndex(next);
        const note = notes.find((n) => n.id === conflictIds[next]);
        if (note) {
          setSelectedNoteId(note.id);
          seek(note.start_ms);
        }
      }
      if (e.key === " " && selectedNoteId) {
        e.preventDefault();
      }
      if (/^[0-9]$/.test(e.key) && selectedNoteId) {
        onEditNote(selectedNoteId, { fret: Number(e.key) }).catch(console.error);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [conflictIndex, conflictIds, notes, onEditNote, seek, selectedNoteId]);

  const maxMs = Math.max(...notes.map((n) => n.start_ms + n.duration_ms), 1000);

  return (
    <div>
      <div className="card" style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <button onClick={() => (playing ? pause() : play())}>{playing ? "Pause" : "Play"}</button>
          <input
            type="range"
            min={0}
            max={maxMs}
            value={currentMs}
            onChange={(e) => seek(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span className="muted">{(currentMs / 1000).toFixed(1)}s</span>
        </div>
        {source.type === "youtube" && source.youtube_id ? (
          <div id="yt-player" style={{ marginTop: "1rem", width: "100%", minHeight: 360 }} />
        ) : (
          <audio ref={audioRef} controls style={{ width: "100%", marginTop: "1rem" }} />
        )}
        <p className="muted" style={{ marginTop: "0.75rem" }}>
          Конфликтов: {conflictIds.length}. Нажмите Tab для перехода к следующему спорному месту.
        </p>
      </div>

      <div className="card">
        <svg viewBox="0 0 900 160" width="100%" height="220" role="img" aria-label="Guitar tab">
          {STRING_Y.map((y, idx) => (
            <g key={idx}>
              <line x1="40" y1={y} x2="860" y2={y} stroke="#445" strokeWidth="1" />
              <text x="8" y={y + 4} fill="#9aa7bd" fontSize="12">
                {idx + 1}
              </text>
            </g>
          ))}
          <line x1={40 + (currentMs / maxMs) * 820} y1="10" x2={40 + (currentMs / maxMs) * 820} y2="140" stroke="#6ea8fe" strokeWidth="2" />
          {notes.map((note) => renderNote(note, maxMs, selectedNoteId, setSelectedNoteId))}
        </svg>
      </div>
    </div>
  );
}

function renderNote(
  note: TabNote,
  maxMs: number,
  selectedNoteId: string | null,
  setSelectedNoteId: (id: string) => void
) {
  const x = 40 + (note.start_ms / maxMs) * 820;
  const y = STRING_Y[note.string - 1];
  const conflict = isConflictNote(note);
  const fill = conflict ? "#f0c040" : "#51cf66";
  const selected = selectedNoteId === note.id;

  return (
    <g
      key={note.id}
      transform={`translate(${x}, ${y - 8})`}
      onClick={() => setSelectedNoteId(note.id)}
      style={{ cursor: "pointer" }}
    >
      <title>{conflict ? conflictReason(note) : note.pitch}</title>
      <rect
        x="-10"
        y="-12"
        width="24"
        height="24"
        rx="4"
        fill={fill}
        stroke={selected ? "#fff" : "#000"}
        strokeWidth={selected ? 2 : 0}
      />
      <text x="-4" y="4" fontSize="11" fill="#081018" fontWeight="700">
        {note.fret}
      </text>
    </g>
  );
}
