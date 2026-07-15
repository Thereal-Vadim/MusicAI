"""Media ingest: upload normalization and YouTube download."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

YOUTUBE_ID_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)

log = logging.getLogger("musicai.ingest")


def extract_youtube_id(url: str) -> str | None:
    match = YOUTUBE_ID_PATTERN.search(url)
    return match.group(1) if match else None


def _yt_dlp_cmd() -> list[str]:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def _resolve_js_runtime() -> list[str]:
    """yt-dlp needs an external JS runtime for YouTube challenge solving."""
    for runtime in ("deno", "node", "bun"):
        if shutil.which(runtime):
            log.info("Using yt-dlp JS runtime: %s", runtime)
            return ["--js-runtimes", runtime]
    log.warning(
        "No JS runtime (deno/node/bun) on PATH — YouTube downloads may fail with HTTP 403. "
        "Install Node 20+ or Deno and retry."
    )
    return []


def _yt_dlp_common_args(*, ffmpeg: str | None) -> list[str]:
    args: list[str] = [
        *_resolve_js_runtime(),
        "--remote-components",
        "ejs:github",
        "--extractor-args",
        "youtube:player_client=default,-android_sdkless",
        "--no-playlist",
    ]
    if ffmpeg:
        args.extend(["--ffmpeg-location", ffmpeg])
    return args


def _resolve_ffmpeg() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            log.info("Using bundled ffmpeg from imageio-ffmpeg: %s", bundled)
            return bundled
    except ImportError:
        log.debug("imageio-ffmpeg not installed")
    return None


def normalize_audio(input_path: Path, output_path: Path, sample_rate: int = 44100) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        log.warning("No ffmpeg available; using raw download %s", input_path)
        shutil.copy2(input_path, output_path)
        return output_path
    try:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(output_path),
        ]
        log.info("ffmpeg normalize: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning("ffmpeg failed (%s), copying input", exc)
        shutil.copy2(input_path, output_path)
    return output_path


def download_youtube(
    url: str,
    output_dir: Path,
    *,
    allow_placeholder: bool = False,
    max_duration_sec: int | None = 120,
) -> tuple[Path, Path | None, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    youtube_id = extract_youtube_id(url)
    if not youtube_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    audio_out = output_dir / f"{youtube_id}.wav"
    video_out = output_dir / f"{youtube_id}.mp4"
    ytdlp = _yt_dlp_cmd()
    ffmpeg = _resolve_ffmpeg()
    has_ffmpeg = ffmpeg is not None
    common_args = _yt_dlp_common_args(ffmpeg=ffmpeg)

    duration_args: list[str] = []
    if max_duration_sec is not None and has_ffmpeg:
        duration_args = ["--download-sections", f"*0-{max_duration_sec}"]
    elif max_duration_sec is not None:
        log.warning(
            "ffmpeg not found; downloading full audio (no %ss cap). Install ffmpeg for partial clips.",
            max_duration_sec,
        )

    log.info("Downloading YouTube audio id=%s url=%s", youtube_id, url)
    audio_cmd = [
        *ytdlp,
        *common_args,
        *duration_args,
        "-x",
        "--audio-format",
        "wav",
        "-o",
        str(output_dir / f"{youtube_id}.%(ext)s"),
        url,
    ]
    audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
    if audio_result.returncode != 0:
        log.error("yt-dlp audio stderr: %s", audio_result.stderr[-2000:])
        if not allow_placeholder:
            raise RuntimeError(f"yt-dlp audio download failed: {audio_result.stderr[-500:]}")
    else:
        log.info("yt-dlp audio stdout tail: %s", audio_result.stdout[-500:])

    wav_candidates = list(output_dir.glob(f"{youtube_id}.*"))
    audio_path = next(
        (p for p in wav_candidates if p.suffix.lower() in {".wav", ".m4a", ".webm", ".opus"}),
        None,
    )
    if audio_path is None or not audio_path.exists():
        if not allow_placeholder:
            raise RuntimeError("YouTube audio file not found after yt-dlp")
        audio_path = _write_placeholder(output_dir, youtube_id)
    elif audio_path.suffix.lower() != ".wav":
        normalize_audio(audio_path, audio_out)
        audio_path = audio_out

    log.info("Downloading YouTube video id=%s", youtube_id)
    video_cmd = [
        *ytdlp,
        *common_args,
        *duration_args,
        "-f",
        "best[height<=720]/best",
        "-o",
        str(video_out),
        url,
    ]
    video_result = subprocess.run(video_cmd, capture_output=True, text=True)
    video_path: Path | None = video_out if video_out.exists() else None
    if video_result.returncode != 0:
        log.warning("yt-dlp video failed: %s", video_result.stderr[-1000:])
    else:
        video_path = video_out if video_out.exists() else None
        log.info("Video path=%s", video_path)

    log.info("Ingest complete audio=%s video=%s", audio_path, video_path)
    return audio_path, video_path, youtube_id


def _write_placeholder(output_dir: Path, youtube_id: str) -> Path:
    import numpy as np
    import soundfile as sf

    audio_path = output_dir / f"{youtube_id}_placeholder.wav"
    log.warning("Writing placeholder tone audio to %s", audio_path)
    sr = 44100
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(audio_path), tone, sr)
    return audio_path
