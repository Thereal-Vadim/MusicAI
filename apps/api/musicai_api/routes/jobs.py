"""Job API routes."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from musicai_api.db.models import Draft, Job
from musicai_api.db.session import get_session
from musicai_api.services.job_runner import spawn_job_process
from musicai_api.services.stem_assets import StemAssetError, list_job_stems, resolve_stem_audio
from musicai_api.settings import settings
from tab_schema.job_progress import (
    PIPELINE_STAGES,
    read_job_status,
    read_live_logs,
    stage_label_pct,
)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

DEFAULT_TUNING = ["E2", "A2", "D3", "G3", "B3", "E4"]
GuitarPart = Literal["combined", "solo", "rhythm"]


class YouTubeJobRequest(BaseModel):
    youtube_url: str
    tuning: list[str] = Field(default_factory=lambda: list(DEFAULT_TUNING))
    guitar_part: GuitarPart = "combined"
    rights_confirmed: bool = False


class StageProgress(BaseModel):
    name: str
    target_pct: int
    duration_sec: float | None = None


class JobResponse(BaseModel):
    id: str
    status: str
    stage: str
    stage_detail: str
    progress_pct: int = 0
    elapsed_sec: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    stage_durations: dict[str, float] = Field(default_factory=dict)
    stages: list[StageProgress] = Field(default_factory=list)
    draft_id: str | None = None


class JobCreateResponse(BaseModel):
    job: JobResponse


class JobLogsResponse(BaseModel):
    lines: list[dict[str, object]]
    total_lines: int
    from_line: int


class StemItemResponse(BaseModel):
    id: str
    label: str
    filename: str


class JobStemsResponse(BaseModel):
    guitar_part: str
    items: list[StemItemResponse]


def _work_dir_for_job(job: Job) -> Path:
    raw = Path(job.work_dir)
    if raw.is_absolute():
        return raw
    # Legacy rows store paths like data/jobs/<id>
    if raw.parts[:2] == ("data", "jobs"):
        return (settings.musicai_data_dir / "jobs" / raw.name).resolve()
    return (settings.jobs_dir / raw.name).resolve()


def _build_stage_list(live_durations: dict[str, float]) -> list[StageProgress]:
    return [
        StageProgress(
            name=stage,
            target_pct=stage_label_pct(stage),
            duration_sec=live_durations.get(stage),
        )
        for stage in PIPELINE_STAGES
    ]


def _job_response(job: Job, draft_id: str | None = None) -> JobResponse:
    live = read_job_status(_work_dir_for_job(job))
    durations = live.stage_durations if live else {}
    return JobResponse(
        id=job.id,
        status=job.status,
        stage=live.stage if live else job.stage,
        stage_detail=live.detail if live else job.stage_detail,
        progress_pct=live.progress_pct if live else 0,
        elapsed_sec=live.elapsed_sec if live else None,
        started_at=live.started_at if live else None,
        finished_at=live.finished_at if live else None,
        stage_durations=durations,
        stages=_build_stage_list(durations),
        draft_id=draft_id,
    )


@router.post("", response_model=JobCreateResponse)
async def create_upload_job(
    file: UploadFile = File(...),
    tuning: str = Form(default=json.dumps(DEFAULT_TUNING)),
    guitar_part: GuitarPart = Form(default="combined"),
    session: AsyncSession = Depends(get_session),
) -> JobCreateResponse:
    job_id = str(uuid.uuid4())
    work_dir = (settings.jobs_dir / job_id).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    upload_path = settings.uploads_dir / f"{job_id}_{file.filename}"
    with upload_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    tuning_list = json.loads(tuning)
    job_meta = {
        "source": {"type": "upload", "path": str(upload_path), "filename": file.filename},
        "tuning": tuning_list,
        "guitar_part": guitar_part,
    }
    (work_dir / "job.json").write_text(json.dumps(job_meta))

    job = Job(
        id=job_id,
        status="queued",
        stage="queued",
        source_type="upload",
        source_ref=file.filename or "upload",
        tuning_json=json.dumps(tuning_list),
        work_dir=str(work_dir),
    )
    session.add(job)
    await session.commit()

    spawn_job_process(job_id, work_dir)
    return JobCreateResponse(job=_job_response(job))


@router.post("/youtube", response_model=JobCreateResponse)
async def create_youtube_job(
    body: YouTubeJobRequest,
    session: AsyncSession = Depends(get_session),
) -> JobCreateResponse:
    if not settings.youtube_ingest_enabled:
        raise HTTPException(status_code=403, detail="YouTube ingest is disabled")
    if not body.rights_confirmed:
        raise HTTPException(status_code=400, detail="Rights confirmation required")

    job_id = str(uuid.uuid4())
    work_dir = (settings.jobs_dir / job_id).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    job_meta = {
        "source": {"type": "youtube", "url": body.youtube_url},
        "tuning": body.tuning,
        "guitar_part": body.guitar_part,
    }
    (work_dir / "job.json").write_text(json.dumps(job_meta))

    job = Job(
        id=job_id,
        status="queued",
        stage="queued",
        source_type="youtube",
        source_ref=body.youtube_url,
        tuning_json=json.dumps(body.tuning),
        work_dir=str(work_dir),
    )
    session.add(job)
    await session.commit()

    spawn_job_process(job_id, work_dir)
    return JobCreateResponse(job=_job_response(job))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)) -> JobResponse:
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    draft_id = None
    if job.status == "done":
        draft = await session.scalar(select(Draft).where(Draft.job_id == job_id))
        draft_id = draft.id if draft else None

    return _job_response(job, draft_id=draft_id)


@router.get("/{job_id}/logs", response_model=JobLogsResponse)
async def get_job_logs(
    job_id: str,
    from_line: int = 0,
    session: AsyncSession = Depends(get_session),
) -> JobLogsResponse:
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    log_path = _work_dir_for_job(job) / "logs" / "live.jsonl"
    lines, total = read_live_logs(log_path, from_line=from_line)
    return JobLogsResponse(lines=lines, total_lines=total, from_line=from_line)


@router.get("/{job_id}/stems", response_model=JobStemsResponse)
async def get_job_stems(job_id: str, session: AsyncSession = Depends(get_session)) -> JobStemsResponse:
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    data = list_job_stems(_work_dir_for_job(job))
    if not data:
        raise HTTPException(status_code=404, detail="Stems not ready")

    return JobStemsResponse(
        guitar_part=data["guitar_part"],
        items=[StemItemResponse(**item) for item in data["items"]],
    )


@router.get("/{job_id}/stems/audio/{stem_id}")
async def get_job_stem_audio(
    job_id: str,
    stem_id: str,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        audio_path, media_type = resolve_stem_audio(_work_dir_for_job(job), stem_id)  # type: ignore[arg-type]
    except StemAssetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        audio_path,
        media_type=media_type,
        filename=audio_path.name,
        headers={"Accept-Ranges": "bytes"},
    )
