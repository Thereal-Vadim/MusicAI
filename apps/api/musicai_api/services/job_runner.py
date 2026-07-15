"""Background job runner."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from musicai_api.db.models import Draft, Job
from tab_schema.job_progress import read_job_status
from tab_schema.models import TabDocument


async def _sync_live_status(job_id: str, work_dir: Path, proc: asyncio.subprocess.Process) -> None:
    from musicai_api.db.session import SessionLocal

    while proc.returncode is None:
        live = read_job_status(work_dir)
        if live:
            async with SessionLocal() as session:
                job = await session.get(Job, job_id)
                if job and job.status == "running":
                    job.stage = live.stage
                    job.stage_detail = live.detail
                    await session.commit()
        await asyncio.sleep(0.8)

    live = read_job_status(work_dir)
    if live:
        async with SessionLocal() as session:
            job = await session.get(Job, job_id)
            if job:
                job.stage = live.stage
                job.stage_detail = live.detail
                await session.commit()


async def run_job_in_background(job_id: str, work_dir: Path) -> None:
    from musicai_api.db.session import SessionLocal

    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job:
            return
        job.status = "running"
        job.stage = "queued"
        await session.commit()

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "musicai_worker.cli",
        "--job-id",
        job_id,
        "--work-dir",
        str(work_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    sync_task = asyncio.create_task(_sync_live_status(job_id, work_dir, proc))
    stdout, stderr = await proc.communicate()
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass

    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job:
            return

        live = read_job_status(work_dir)
        if live:
            job.stage = live.stage
            job.stage_detail = live.detail

        draft_path = work_dir / "draft.json"
        if proc.returncode == 0 and draft_path.exists():
            document = TabDocument.model_validate_json(draft_path.read_text())
            job.status = "done"
            job.stage = "done"
            existing = await session.scalar(select(Draft).where(Draft.job_id == job_id))
            if existing:
                existing.document_json = document.model_dump_json()
            else:
                session.add(Draft(job_id=job_id, document_json=document.model_dump_json()))
        else:
            job.status = "failed"
            job.stage = "failed"
            job.error = (stderr.decode() if stderr else stdout.decode())[:2000]

        await session.commit()


def spawn_job_process(job_id: str, work_dir: Path) -> None:
    asyncio.create_task(run_job_in_background(job_id, work_dir))
