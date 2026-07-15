"""Pipeline progress percentages and live status file helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_STAGES = [
    "ingest",
    "separate",
    "dereverb",
    "guitar_demix",
    "demix_validate",
    "transcribe",
    "vision",
    "fusion",
    "judge",
    "draft",
    "done",
]

# (start_pct, end_pct) for each stage
STAGE_PCT_RANGE: dict[str, tuple[int, int]] = {
    "queued": (0, 0),
    "ingest": (0, 8),
    "separate": (8, 14),
    "dereverb": (14, 18),
    "guitar_demix": (18, 22),
    "demix_validate": (22, 24),
    "transcribe": (24, 44),
    "vision": (44, 62),
    "fusion": (62, 72),
    "judge": (72, 82),
    "draft": (82, 95),
    "done": (100, 100),
    "failed": (0, 0),
}


def stage_progress_pct(stage: str, sub_progress: float = 0.0) -> int:
    lo, hi = STAGE_PCT_RANGE.get(stage, (0, 0))
    if stage == "done":
        return 100
    sub = max(0.0, min(1.0, sub_progress))
    return int(round(lo + (hi - lo) * sub))


def stage_label_pct(stage: str) -> int:
    """Display percentage shown next to a stage row (end of stage range)."""
    _, hi = STAGE_PCT_RANGE.get(stage, (0, 0))
    return hi if stage != "done" else 100


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobLiveStatus:
    stage: str = "queued"
    detail: str = ""
    progress_pct: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_sec: float = 0.0
    stage_durations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "detail": self.detail,
            "progress_pct": self.progress_pct,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "stage_durations": self.stage_durations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobLiveStatus:
        return cls(
            stage=str(data.get("stage", "queued")),
            detail=str(data.get("detail", "")),
            progress_pct=int(data.get("progress_pct", 0)),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            elapsed_sec=float(data.get("elapsed_sec", 0.0)),
            stage_durations=dict(data.get("stage_durations") or {}),
        )


class JobStatusWriter:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.status_path = work_dir / "status.json"
        self._started_at: str | None = None
        self._stage_durations: dict[str, float] = {}

    def mark_started(self) -> None:
        self._started_at = utc_now_iso()
        self.write("queued", "Job queued", sub_progress=0.0)

    def write(
        self,
        stage: str,
        detail: str = "",
        *,
        sub_progress: float = 0.0,
        finished: bool = False,
    ) -> None:
        started_at = self._started_at or utc_now_iso()
        if self._started_at is None:
            self._started_at = started_at

        started_dt = datetime.fromisoformat(started_at)
        elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()

        status = JobLiveStatus(
            stage=stage,
            detail=detail,
            progress_pct=stage_progress_pct(stage, sub_progress),
            started_at=started_at,
            finished_at=utc_now_iso() if finished else None,
            elapsed_sec=elapsed,
            stage_durations=dict(self._stage_durations),
        )
        self.status_path.write_text(json.dumps(status.to_dict(), indent=2), encoding="utf-8")

    def record_stage_duration(self, stage: str, seconds: float) -> None:
        self._stage_durations[stage] = round(seconds, 2)


def read_job_status(work_dir: Path) -> JobLiveStatus | None:
    path = work_dir / "status.json"
    if not path.exists():
        return None
    try:
        return JobLiveStatus.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def read_live_logs(log_path: Path, from_line: int = 0) -> tuple[list[dict[str, Any]], int]:
    if not log_path.exists():
        return [], 0
    lines = log_path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    slice_lines = lines[max(0, from_line) :]
    entries: list[dict[str, Any]] = []
    for line in slice_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"ts": "", "level": "INFO", "stage": "-", "message": line})
    return entries, total
