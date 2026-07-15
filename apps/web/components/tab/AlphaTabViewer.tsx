"use client";

import { useEffect, useRef, useState } from "react";
import { ALPHATAB_CDN, loadAlphaTab } from "@/lib/load-alphatab";
import "./tab-sheet.css";

const CDN = ALPHATAB_CDN;

interface AlphaTabViewerProps {
  draftId: string;
  tex: string;
  gp5Url: string;
  trackName?: string | null;
}

type AlphaTabApi = {
  destroy: () => void;
  load: (data: unknown, trackIndexes?: number[]) => boolean;
  tex: (tex: string) => void;
  playPause: () => void;
  stop: () => void;
  playbackSpeed: number;
  playerStateChanged: { on: (cb: (args: { state: number }) => void) => void };
  scoreLoaded: { on: (cb: () => void) => void };
  renderStarted: { on: (cb: () => void) => void };
  error: { on: (cb: (err: { message: string }) => void) => void };
  timePosition: number;
  endTime: number;
};

const PLAYER_PLAYING = 1;

export function AlphaTabViewer({ draftId, tex, gp5Url, trackName }: AlphaTabViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<AlphaTabApi | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(100);
  const [positionSec, setPositionSec] = useState(0);
  const [durationSec, setDurationSec] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let api: AlphaTabApi | null = null;

    async function init() {
      if (!containerRef.current) return;
      setLoading(true);
      setError(null);

      try {
        const alphaTab = await loadAlphaTab();
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

        api.scoreLoaded.on(() => {
          setDurationSec(Math.max(0, api?.endTime ?? 0));
        });

        api.error.on((err) => {
          setError(err.message);
          setLoading(false);
        });

        api.renderStarted.on(() => {
          setLoading(false);
        });

        let loaded = false;
        try {
          const gp5Res = await fetch(gp5Url);
          if (gp5Res.ok) {
            const buffer = await gp5Res.arrayBuffer();
            if (buffer.byteLength > 100) {
              loaded = api.load(new Uint8Array(buffer));
            }
          }
        } catch {
          // fall through to alphaTex
        }

        if (!loaded) {
          if (!tex.trim()) {
            throw new Error("Не удалось загрузить GP5 и alphaTex пуст.");
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

  return (
    <div className="tab-sheet">
      <div ref={scrollRef} className="tab-sheet-scroll">
        {loading ? <p className="tab-sheet-loading">Загрузка табулатуры…</p> : null}
        {error ? <p className="tab-sheet-error">{error}</p> : null}
        <div ref={containerRef} className="tab-sheet-canvas" />
      </div>

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
        {trackName ? <span className="tab-sheet-meta">{trackName}</span> : null}
      </div>
    </div>
  );
}
