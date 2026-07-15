"""Job API routes."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from musicai_api.db.models import Draft, Job
from musicai_api.db.session import get_session
from musicai_api.services.job_runner import spawn_job_process
from musicai_api.settings import settings

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

DEFAULT_TUNING = ["E2", "A2", "D3", "G3", "B3", "E4"]


class YouTubeJobRequest(BaseModel):
    youtube_url: str
    tuning: list[str] = Field(default_factory=lambda: list(DEFAULT_TUNING))
    rights_confirmed: bool = False


class JobResponse(BaseModel):
    id: str
    status: str
    stage: str
    stage_detail: str
    draft_id: str | None = None


class JobCreateResponse(BaseModel):
    job: JobResponse


@router.post("", response_model=JobCreateResponse)
async def create_upload_job(
    file: UploadFile = File(...),
    tuning: str = Form(default=json.dumps(DEFAULT_TUNING)),
    session: AsyncSession = Depends(get_session),
) -> JobCreateResponse:
    job_id = str(uuid.uuid4())
    work_dir = settings.jobs_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    upload_path = settings.uploads_dir / f"{job_id}_{file.filename}"
    with upload_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    tuning_list = json.loads(tuning)
    job_meta = {
        "source": {"type": "upload", "path": str(upload_path), "filename": file.filename},
        "tuning": tuning_list,
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
    return JobCreateResponse(job=JobResponse(id=job_id, status="queued", stage="queued", stage_detail=""))


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
    work_dir = settings.jobs_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    job_meta = {
        "source": {"type": "youtube", "url": body.youtube_url},
        "tuning": body.tuning,
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
    return JobCreateResponse(job=JobResponse(id=job_id, status="queued", stage="queued", stage_detail=""))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)) -> JobResponse:
    job = await session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    draft_id = None
    if job.status == "done":
        draft = await session.scalar(select(Draft).where(Draft.job_id == job_id))
        draft_id = draft.id if draft else None

    return JobResponse(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_detail=job.stage_detail,
        draft_id=draft_id,
    )
