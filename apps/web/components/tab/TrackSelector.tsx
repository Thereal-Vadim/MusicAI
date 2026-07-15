"use client";

import type { AlphaTabScoreTrack } from "./alphatab-types";

export type TrackSelectorProps = {
  tracks: AlphaTabScoreTrack[];
  activeIndex: number | null;
  muted: Record<number, boolean>;
  solo: Record<number, boolean>;
  volumes: Record<number, number>;
  onSelect: (track: AlphaTabScoreTrack) => void;
  onToggleMute: (track: AlphaTabScoreTrack) => void;
  onToggleSolo: (track: AlphaTabScoreTrack) => void;
  onVolume: (track: AlphaTabScoreTrack, volume: number) => void;
};

function GuitarIcon({ active }: { active: boolean }) {
  const stroke = active ? "#2f9e44" : "#495057";
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
      <path
        d="M8 20c0-2.5 1.5-4 3.5-4.5L14 6l2.5 9.5C18.5 16 20 17.5 20 20c0 2.2-1.8 4-4 4h-4c-2.2 0-4-1.8-4-4z"
        fill="none"
        stroke={stroke}
        strokeWidth="1.6"
      />
      <circle cx="14" cy="20" r="2.2" fill={stroke} />
      <path d="M14 6V3.5" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function HeadphonesIcon({ active }: { active: boolean }) {
  const stroke = active ? "#2f9e44" : "#868e96";
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
      <path
        d="M4 11a6 6 0 0 1 12 0v5h-2.5a1.5 1.5 0 0 1-1.5-1.5V13a1.5 1.5 0 0 1 1.5-1.5H16M4 11v5h2.5A1.5 1.5 0 0 0 8 14.5V13A1.5 1.5 0 0 0 6.5 11.5H4"
        fill="none"
        stroke={stroke}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SpeakerIcon({ muted }: { muted: boolean }) {
  const stroke = muted ? "#c92a2a" : "#868e96";
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
      <path
        d="M3.5 8.5v3h2.5L10 15V5L6 8.5H3.5z"
        fill="none"
        stroke={stroke}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      {muted ? (
        <path d="M13 8l4 4M17 8l-4 4" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" />
      ) : (
        <path
          d="M13 7.5a3.5 3.5 0 0 1 0 5M15.2 5.5a6 6 0 0 1 0 9"
          fill="none"
          stroke={stroke}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      )}
    </svg>
  );
}

export function TrackSelector({
  tracks,
  activeIndex,
  muted,
  solo,
  volumes,
  onSelect,
  onToggleMute,
  onToggleSolo,
  onVolume,
}: TrackSelectorProps) {
  if (tracks.length === 0) return null;

  return (
    <div className="track-selector" role="listbox" aria-label="Tracks">
      {tracks.map((track) => {
        const index = track.index;
        const isActive = activeIndex === index;
        const isMuted = Boolean(muted[index]);
        const isSolo = Boolean(solo[index]);
        const volume = volumes[index] ?? 1;

        return (
          <div
            key={index}
            role="option"
            aria-selected={isActive}
            className={`track-selector-row${isActive ? " is-active" : ""}${isMuted ? " is-muted" : ""}`}
            onClick={() => onSelect(track)}
          >
            <span className={`track-selector-dot${isActive ? " is-on" : ""}`} />
            <GuitarIcon active={isActive} />
            <div className="track-selector-meta">
              <span className="track-selector-name">{track.name || `Track ${index + 1}`}</span>
            </div>
            <div className="track-selector-actions" onClick={(e) => e.stopPropagation()}>
              <input
                type="range"
                className="track-selector-volume"
                min={0}
                max={1}
                step={0.05}
                value={volume}
                aria-label={`Volume ${track.name || index}`}
                onChange={(e) => onVolume(track, Number(e.target.value))}
              />
              <button
                type="button"
                className={`track-selector-btn${isSolo ? " is-on" : ""}`}
                aria-pressed={isSolo}
                title="Solo"
                onClick={() => onToggleSolo(track)}
              >
                <HeadphonesIcon active={isSolo} />
              </button>
              <button
                type="button"
                className={`track-selector-btn${isMuted ? " is-on mute" : ""}`}
                aria-pressed={isMuted}
                title="Mute"
                onClick={() => onToggleMute(track)}
              >
                <SpeakerIcon muted={isMuted} />
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
