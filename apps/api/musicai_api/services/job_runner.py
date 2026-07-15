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
from tab_schema.models import TabDocument


async def run_job_in_background(job_id: str, work_dir: Path) -> None:
    from musicai_api.db.session import SessionLocal

    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job:
            return
        job.status = "running"
        job.stage = "ingest"
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
    stdout, stderr = await proc.communicate()

    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job:
            return

        status_path = work_dir / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            job.stage = status.get("stage", job.stage)
            job.stage_detail = status.get("detail", "")

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
