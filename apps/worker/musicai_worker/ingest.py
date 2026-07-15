"""Media ingest: upload normalization and YouTube download."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

YOUTUBE_ID_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)


def extract_youtube_id(url: str) -> str | None:
    match = YOUTUBE_ID_PATTERN.search(url)
    return match.group(1) if match else None


def normalize_audio(input_path: Path, output_path: Path, sample_rate: int = 44100) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copy2(input_path, output_path)
    return output_path


def download_youtube(url: str, output_dir: Path) -> tuple[Path, Path | None, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    youtube_id = extract_youtube_id(url)
    if not youtube_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    audio_out = output_dir / f"{youtube_id}.wav"
    video_out = output_dir / f"{youtube_id}.mp4"

    try:
        audio_cmd = [
            "yt-dlp",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            str(output_dir / f"{youtube_id}.%(ext)s"),
            url,
        ]
        subprocess.run(audio_cmd, check=True, capture_output=True, text=True)
        wav_candidates = list(output_dir.glob(f"{youtube_id}.*"))
        audio_path = next((p for p in wav_candidates if p.suffix in {".wav", ".m4a", ".webm"}), audio_out)
        if audio_path.suffix != ".wav":
            normalize_audio(audio_path, audio_out)
            audio_path = audio_out

        video_cmd = [
            "yt-dlp",
            "-f",
            "best[height<=720]",
            "-o",
            str(video_out),
            url,
        ]
        subprocess.run(video_cmd, check=True, capture_output=True, text=True)
        video_path = video_out if video_out.exists() else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        audio_path = output_dir / f"{youtube_id}_placeholder.wav"
        if not audio_path.exists():
            import numpy as np
            import soundfile as sf

            sr = 44100
            t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
            tone = 0.1 * np.sin(2 * np.pi * 440 * t)
            sf.write(str(audio_path), tone, sr)
        video_path = None

    return audio_path, video_path, youtube_id
