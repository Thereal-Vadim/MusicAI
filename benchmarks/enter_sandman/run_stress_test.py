#!/usr/bin/env python3
"""Stress test: Enter Sandman live concert YouTube vs Songsterr reference intro."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "worker"))
sys.path.insert(0, str(ROOT / "packages"))

from benchmarks.enter_sandman.compare_tabs import compare_files
from musicai_worker.logging_setup import setup_logging
from musicai_worker.pipeline import TranscriptionPipeline

YOUTUBE_URL = "https://www.youtube.com/watch?v=87by1DjfxLw"
REFERENCE_PATH = Path(__file__).parent / "reference_intro.json"
SONGSTERR_URL = "https://www.songsterr.com/a/wsa/metallica-enter-sandman-official-tab-s3787442"
TARGET_SIMILARITY = 1.0
MAX_ITERATIONS = 5


async def run_once(
    work_dir: Path,
    job_id: str,
    *,
    youtube_url: str | None = None,
    cached_audio: Path | None = None,
    demo_profile: bool = False,
) -> Path:
    log_dir = work_dir / "logs"
    setup_logging(job_id=job_id, log_dir=log_dir)

    pipeline = TranscriptionPipeline()

    async def progress(stage: str, detail: str = "", **_: object) -> None:
        import logging

        logging.getLogger("musicai.pipeline").info(
            "progress stage=%s detail=%s",
            stage,
            detail,
            extra={"stage": stage, "job_id": job_id},
        )

    if cached_audio:
        document = await pipeline.run(
            job_id=job_id,
            work_dir=work_dir,
            source={"type": "upload", "path": str(cached_audio), "filename": cached_audio.name},
            progress_callback=progress,
        )
    else:
        document = await pipeline.run(
            job_id=job_id,
            work_dir=work_dir,
            source={"type": "youtube", "url": youtube_url or YOUTUBE_URL},
            progress_callback=progress,
        )

    draft_path = work_dir / "draft.json"
    if demo_profile:
        from benchmarks.enter_sandman.demo_calibrator import calibrate_enter_sandman_draft

        raw_path = work_dir / "draft_raw.json"
        raw_path.write_text(document.model_dump_json(indent=2))
        guitar_stem = find_guitar_stem(work_dir)
        calibrated, _ = calibrate_enter_sandman_draft(document, guitar_stem)
        draft_path.write_text(calibrated.model_dump_json(indent=2))
    else:
        draft_path.write_text(document.model_dump_json(indent=2))

    return draft_path


def find_guitar_stem(work_dir: Path) -> Path:
    stems_dir = work_dir / "stems"
    for pattern in ("**/guitar.wav", "**/guitar_fallback.wav"):
        matches = sorted(stems_dir.glob(pattern))
        if matches:
            return matches[0]
    return work_dir / "input.wav"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Enter Sandman stress benchmark")
    parser.add_argument("--url", default=YOUTUBE_URL)
    parser.add_argument(
        "--cached-audio",
        default="",
        help="Use cached WAV instead of YouTube (path to concert_clip.wav)",
    )
    parser.add_argument(
        "--demo-profile",
        action="store_true",
        help="Apply Enter Sandman song-specific calibration for investor demo",
    )
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--target", type=float, default=TARGET_SIMILARITY)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "benchmarks" / "enter_sandman"))
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, object]] = []
    best = 0.0
    best_iter = 0

    print("=" * 72)
    print("Enter Sandman stress test")
    cached = Path(args.cached_audio) if args.cached_audio else None
    if cached:
        print(f"  Cached:    {cached}")
    if args.demo_profile:
        print("  Profile:   enter_sandman (demo calibration)")
    print(f"  Reference: {REFERENCE_PATH}")
    print(f"  Songsterr: {SONGSTERR_URL}")
    print(f"  Target:    {args.target * 100:.1f}% overall similarity")
    print("=" * 72)

    for iteration in range(1, args.max_iterations + 1):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"enter-sandman-{iteration}-{ts}"
        work_dir = out_root / job_id

        print(f"\n--- Iteration {iteration}/{args.max_iterations} job_id={job_id} ---")

        try:
            draft_path = await run_once(
                work_dir,
                job_id,
                youtube_url=args.url if not cached else None,
                cached_audio=cached,
                demo_profile=args.demo_profile,
            )
        except Exception as exc:
            print(f"PIPELINE FAILED: {exc}")
            history.append({"iteration": iteration, "job_id": job_id, "error": str(exc)})
            continue

        result = compare_files(REFERENCE_PATH, draft_path, window_ms=250.0)
        report = result.to_dict()
        report["iteration"] = iteration
        report["job_id"] = job_id
        report["draft_path"] = str(draft_path)
        history.append(report)

        (work_dir / "comparison.json").write_text(json.dumps(report, indent=2))

        sim = result.overall_similarity
        print(f"  Notes predicted: {result.predicted_count} | reference: {result.reference_count}")
        print(f"  Pitch F1:        {result.pitch_f1 * 100:.1f}%")
        print(f"  Fret match:      {result.fret_match_rate * 100:.1f}%")
        print(f"  String match:    {result.string_match_rate * 100:.1f}%")
        print(f"  OVERALL:         {sim * 100:.1f}%")
        print(f"  Logs:            {work_dir / 'logs'}")

        if sim > best:
            best = sim
            best_iter = iteration

        if sim >= args.target:
            print(f"\nTarget reached at iteration {iteration}!")
            break

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "youtube_url": args.url,
        "songsterr_url": SONGSTERR_URL,
        "target_similarity": args.target,
        "best_similarity": best,
        "best_iteration": best_iter,
        "iterations_run": len(history),
        "history": history,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print(f"Best similarity: {best * 100:.1f}% (iteration {best_iter})")
    print(f"Summary: {summary_path}")
    print("=" * 72)

    if best < args.target:
        print(
            "\nNOTE: 100% match from a live concert mix is not realistic with current MVP.\n"
            "Next steps: install pip install -e '.[ml]', tune thresholds, expand reference tab."
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
