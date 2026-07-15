"""Worker CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from musicai_worker.pipeline import TranscriptionPipeline
from tab_schema.job_progress import JobStatusWriter


async def process_job(job_id: str, work_dir: Path) -> None:
    meta_path = work_dir / "job.json"
    if not meta_path.exists():
        print(f"Job metadata not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    pipeline = TranscriptionPipeline()
    status_writer = JobStatusWriter(work_dir)
    status_writer.mark_started()

    async def progress(
        stage: str,
        detail: str = "",
        *,
        sub_progress: float = 0.0,
        finished: bool = False,
        stage_duration_sec: float | None = None,
    ) -> None:
        if stage_duration_sec is not None:
            status_writer.record_stage_duration(stage, stage_duration_sec)
        status_writer.write(stage, detail, sub_progress=sub_progress, finished=finished)

    await pipeline.run(
        job_id=job_id,
        work_dir=work_dir,
        source=meta["source"],
        tuning=meta.get("tuning"),
        guitar_part=meta.get("guitar_part", "combined"),
        progress_callback=progress,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MusicAI worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    asyncio.run(process_job(args.job_id, Path(args.work_dir)))


if __name__ == "__main__":
    main()
