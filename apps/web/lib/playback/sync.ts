"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export function usePlaybackClock(initialMs = 0) {
  const [currentMs, setCurrentMs] = useState(initialMs);
  const [playing, setPlaying] = useState(false);
  const rafRef = useRef<number | null>(null);
  const startRef = useRef<number>(0);
  const baseRef = useRef<number>(initialMs);

  const tick = useCallback(() => {
    const elapsed = performance.now() - startRef.current;
    setCurrentMs(baseRef.current + elapsed);
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => {
    if (playing) {
      startRef.current = performance.now();
      rafRef.current = requestAnimationFrame(tick);
    } else if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, tick]);

  const play = () => setPlaying(true);
  const pause = () => {
    if (playing) {
      baseRef.current = currentMs;
      setPlaying(false);
    }
  };
  const seek = (ms: number) => {
    baseRef.current = ms;
    setCurrentMs(ms);
    if (playing) startRef.current = performance.now();
  };

  return { currentMs, playing, play, pause, seek, setPlaying };
}

export function syncAudioElement(audio: HTMLAudioElement | null, currentMs: number, playing: boolean) {
  if (!audio) return;
  const targetSec = currentMs / 1000;
  if (Math.abs(audio.currentTime - targetSec) > 0.08) {
    audio.currentTime = targetSec;
  }
  if (playing && audio.paused) audio.play().catch(() => undefined);
  if (!playing && !audio.paused) audio.pause();
}

declare global {
  interface Window {
    YT?: {
      Player: new (
        elementId: string,
        options: {
          videoId: string;
          events?: { onReady?: (e: { target: YTPlayer }) => void };
        }
      ) => YTPlayer;
      PlayerState: { PLAYING: number; PAUSED: number };
    };
    onYouTubeIframeAPIReady?: () => void;
  }
}

export interface YTPlayer {
  playVideo: () => void;
  pauseVideo: () => void;
  seekTo: (seconds: number, allowSeekAhead: boolean) => void;
  getCurrentTime: () => number;
  getPlayerState: () => number;
}

export function loadYouTubeApi(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  if (window.YT?.Player) return Promise.resolve();

  return new Promise((resolve) => {
    const existing = document.getElementById("youtube-iframe-api");
    if (!existing) {
      const tag = document.createElement("script");
      tag.id = "youtube-iframe-api";
      tag.src = "https://www.youtube.com/iframe_api";
      document.body.appendChild(tag);
    }
    window.onYouTubeIframeAPIReady = () => resolve();
  });
}

export function isYTPlayerReady(player: YTPlayer | null | undefined): player is YTPlayer {
  return (
    !!player &&
    typeof player.getCurrentTime === "function" &&
    typeof player.getPlayerState === "function" &&
    typeof player.seekTo === "function"
  );
}

export function syncYouTubePlayer(player: YTPlayer | null, currentMs: number, playing: boolean) {
  if (!isYTPlayerReady(player)) return;
  const targetSec = currentMs / 1000;
  try {
    if (Math.abs(player.getCurrentTime() - targetSec) > 0.15) {
      player.seekTo(targetSec, true);
    }
    const state = player.getPlayerState();
    const YT_PLAYING = window.YT?.PlayerState.PLAYING ?? 1;
    if (playing && state !== YT_PLAYING) player.playVideo();
    if (!playing && state === YT_PLAYING) player.pauseVideo();
  } catch {
    // Player can still be initializing between onReady and first sync tick.
  }
}
