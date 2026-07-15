"""Worker CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from musicai_worker.pipeline import TranscriptionPipeline


async def process_job(job_id: str, work_dir: Path) -> None:
    meta_path = work_dir / "job.json"
    if not meta_path.exists():
        print(f"Job metadata not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    import json

    meta = json.loads(meta_path.read_text())
    pipeline = TranscriptionPipeline()

    async def progress(stage: str, detail: str = "") -> None:
        status_path = work_dir / "status.json"
        status_path.write_text(json.dumps({"stage": stage, "detail": detail}))

    await pipeline.run(
        job_id=job_id,
        work_dir=work_dir,
        source=meta["source"],
        tuning=meta.get("tuning"),
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
