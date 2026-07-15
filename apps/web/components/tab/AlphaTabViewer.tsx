"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ALPHATAB_CDN, loadAlphaTab } from "@/lib/load-alphatab";
import type { BenchmarkComparison } from "@/lib/benchmark-format";
import type { AlphaTabApi, AlphaTabScoreTrack } from "./alphatab-types";
import { TrackSelector } from "./TrackSelector";
import "./tab-sheet.css";

const CDN = ALPHATAB_CDN;

interface AlphaTabViewerProps {
  draftId: string;
  tex: string;
  gp5Url: string;
  trackName?: string | null;
  comparison?: BenchmarkComparison | null;
}

const PLAYER_PLAYING = 1;

function normalizeTracks(raw: AlphaTabScoreTrack[] | undefined): AlphaTabScoreTrack[] {
  if (!raw?.length) return [];
  return raw.map((track, index) => ({
    ...track,
    index: typeof track.index === "number" ? track.index : index,
    name: track.name || `Track ${index + 1}`,
  }));
}

export function AlphaTabViewer({ draftId, tex, gp5Url, trackName, comparison }: AlphaTabViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<AlphaTabApi | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(100);
  const [positionSec, setPositionSec] = useState(0);
  const [durationSec, setDurationSec] = useState(0);
  const [tracks, setTracks] = useState<AlphaTabScoreTrack[]>([]);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [muted, setMuted] = useState<Record<number, boolean>>({});
  const [solo, setSolo] = useState<Record<number, boolean>>({});
  const [volumes, setVolumes] = useState<Record<number, number>>({});

  useEffect(() => {
    let cancelled = false;
    let api: AlphaTabApi | null = null;

    async function init() {
      if (!containerRef.current) return;
      setLoading(true);
      setError(null);
      setTracks([]);
      setActiveIndex(null);
      setMuted({});
      setSolo({});
      setVolumes({});

      try {
        const [alphaTab, gp5Buffer] = await Promise.all([
          loadAlphaTab(),
          fetch(gp5Url)
            .then(async (res) => (res.ok ? res.arrayBuffer() : null))
            .catch(() => null),
        ]);
        if (cancelled || !containerRef.current) return;

        containerRef.current.innerHTML = "";

        const settings = {
          core: {
            tex: false,
            scriptFile: `${CDN}/alphaTab.min.js`,
            fontDirectory: `${CDN}/font/`,
            logLevel: alphaTab.LogLevel.None,
          },
          display: {
            scale: 1,
            stretchForce: 0.85,
            layoutMode: alphaTab.LayoutMode.Page,
            staveProfile: alphaTab.StaveProfile.Tab,
            barsPerRow: 4,
            padding: [24, 32],
            resources: {
              staffLineColor: "#212529",
              barSeparatorColor: "#adb5bd",
              mainGlyphColor: "#212529",
              secondaryGlyphColor: "#495057",
              scoreInfoColor: "#212529",
              barNumberColor: "#868e96",
              fretboardNoteColor: "#212529",
              fretboardStringColor: "#495057",
              fretboardFretColor: "#868e96",
              noteCircleColor: "#212529",
            },
          },
          notation: {
            elements: {
              scoreTitle: false,
              scoreSubTitle: false,
              scoreArtist: false,
              scoreAlbum: false,
              scoreWords: false,
              scoreMusic: false,
              scoreWordsAndMusic: false,
              scoreCopyright: false,
              guitarTuning: false,
              trackNames: false,
              chordDiagrams: false,
            },
          },
          player: {
            enablePlayer: true,
            playerMode: alphaTab.PlayerMode.EnabledAutomatic,
            enableCursor: true,
            enableUserInteraction: true,
            soundFont: `${CDN}/soundfont/sonivox.sf2`,
            scrollElement: scrollRef.current ?? "html,body",
          },
        } satisfies Record<string, unknown>;

        api = new alphaTab.AlphaTabApi(containerRef.current, settings) as unknown as AlphaTabApi;
        apiRef.current = api;

        api.playerStateChanged.on((args) => {
          setPlaying(args.state === PLAYER_PLAYING);
        });

        api.scoreLoaded.on((score) => {
          setDurationSec(Math.max(0, api?.endTime ?? 0));
          const loadedTracks = normalizeTracks(score?.tracks ?? api?.score?.tracks);
          setTracks(loadedTracks);
          const volumeMap: Record<number, number> = {};
          for (const track of loadedTracks) volumeMap[track.index] = 1;
          setVolumes(volumeMap);
          if (loadedTracks.length > 0) {
            setActiveIndex(loadedTracks[0].index);
            try {
              api?.renderTracks([loadedTracks[0]]);
            } catch {
              /* older builds may only support load-time track filter */
            }
          }
        });

        api.error.on((err) => {
          const details =
            typeof err === "object" && err !== null && "message" in err
              ? String((err as { message?: string }).message)
              : String(err);
          setError(
            details.includes("diagnostics")
              ? `${details} (проверьте GP5 fallback или перезагрузите draft)`
              : details
          );
          setLoading(false);
        });

        api.renderStarted.on(() => {
          setLoading(false);
        });

        let loaded = false;
        if (gp5Buffer && gp5Buffer.byteLength > 100) {
          // Load all tracks so TrackSelector mute/solo/volume can mix them.
          loaded = api.load(new Uint8Array(gp5Buffer));
        }

        if (!loaded) {
          if (!tex.trim()) {
            if (!cancelled) setLoading(true);
            return;
          }
          api.tex(tex);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "alphaTab failed to load");
          setLoading(false);
        }
      }
    }

    init();

    const tick = window.setInterval(() => {
      const current = apiRef.current;
      if (!current) return;
      setPositionSec(current.timePosition ?? 0);
      if (current.endTime > 0) setDurationSec(current.endTime);
    }, 250);

    return () => {
      cancelled = true;
      window.clearInterval(tick);
      apiRef.current = null;
      api?.destroy();
    };
  }, [draftId, gp5Url, tex]);

  useEffect(() => {
    if (apiRef.current) {
      apiRef.current.playbackSpeed = speed / 100;
    }
  }, [speed]);

  const onPlayPause = () => apiRef.current?.playPause();
  const onStop = () => apiRef.current?.stop();

  const onSeek = (value: number) => {
    if (!apiRef.current) return;
    apiRef.current.timePosition = value;
    setPositionSec(value);
  };

  const onSelectTrack = useCallback((track: AlphaTabScoreTrack) => {
    setActiveIndex(track.index);
    try {
      apiRef.current?.renderTracks([track]);
    } catch {
      /* ignore */
    }
  }, []);

  const onToggleMute = useCallback((track: AlphaTabScoreTrack) => {
    setMuted((prev) => {
      const nextMuted = !prev[track.index];
      try {
        apiRef.current?.changeTrackMute([track], nextMuted);
      } catch {
        /* ignore */
      }
      return { ...prev, [track.index]: nextMuted };
    });
  }, []);

  const onToggleSolo = useCallback((track: AlphaTabScoreTrack) => {
    setSolo((prev) => {
      const nextSolo = !prev[track.index];
      try {
        apiRef.current?.changeTrackSolo([track], nextSolo);
      } catch {
        /* ignore */
      }
      return { ...prev, [track.index]: nextSolo };
    });
  }, []);

  const onVolume = useCallback((track: AlphaTabScoreTrack, volume: number) => {
    setVolumes((prev) => ({ ...prev, [track.index]: volume }));
    try {
      apiRef.current?.changeTrackVolume([track], volume);
    } catch {
      /* ignore */
    }
  }, []);

  return (
    <div className="tab-sheet">
      <div className="tab-sheet-body">
        <TrackSelector
          tracks={tracks}
          activeIndex={activeIndex}
          muted={muted}
          solo={solo}
          volumes={volumes}
          onSelect={onSelectTrack}
          onToggleMute={onToggleMute}
          onToggleSolo={onToggleSolo}
          onVolume={onVolume}
        />
        <div ref={scrollRef} className="tab-sheet-scroll">
          {loading ? <p className="tab-sheet-loading">Загрузка табулатуры…</p> : null}
          {error ? <p className="tab-sheet-error">{error}</p> : null}
          <div ref={containerRef} className="tab-sheet-canvas" />
        </div>
      </div>

      {comparison ? (
        <div className="tab-sheet-benchmark-bar">
          <span className="tab-sheet-benchmark-chip">
            Overall F1 <strong>{(comparison.metrics.overall_f1 * 100).toFixed(0)}%</strong>
          </span>
          <span className="tab-sheet-benchmark-chip">
            Pitch <strong>{(comparison.metrics.pitch_f1 * 100).toFixed(0)}%</strong>
          </span>
          <span className="tab-sheet-benchmark-chip">
            Fret <strong>{(comparison.metrics.fret_accuracy * 100).toFixed(0)}%</strong>
          </span>
          <span className="tab-sheet-benchmark-chip">
            Timing <strong>{(comparison.metrics.timing_accuracy * 100).toFixed(0)}%</strong>
          </span>
          <span className="tab-sheet-benchmark-chip muted">
            {comparison.metrics.matched}/{comparison.metrics.reference_count} нот эталона
          </span>
        </div>
      ) : null}

      <div className="tab-sheet-controls">
        <button type="button" onClick={onPlayPause}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" className="secondary" onClick={onStop}>
          Stop
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(durationSec, 1)}
          step={0.05}
          value={Math.min(positionSec, durationSec || 0)}
          onChange={(e) => onSeek(Number(e.target.value))}
        />
        <span className="tab-sheet-meta">
          {positionSec.toFixed(1)}s / {durationSec.toFixed(1)}s
        </span>
        <label className="tab-sheet-meta" style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          Speed
          <input
            type="range"
            min={50}
            max={150}
            step={5}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            style={{ width: 80 }}
          />
          {speed}%
        </label>
        {trackName && tracks.length <= 1 ? <span className="tab-sheet-meta">{trackName}</span> : null}
      </div>
    </div>
  );
}
