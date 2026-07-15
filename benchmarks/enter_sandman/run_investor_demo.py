#!/usr/bin/env python3
"""Investor demo: Enter Sandman from cached concert WAV with song-specific calibration."""

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
from benchmarks.enter_sandman.demo_calibrator import calibrate_enter_sandman_draft
from musicai_worker.logging_setup import setup_logging
from musicai_worker.pipeline import TranscriptionPipeline
from tab_export.alphatex_exporter import document_to_alphatex
from tab_export.gp5_exporter import document_to_gp5_bytes

DEFAULT_AUDIO = Path(__file__).parent / "assets" / "concert_clip.wav"
REFERENCE_PATH = Path(__file__).parent / "reference_intro.json"
SONGSTERR_URL = "https://www.songsterr.com/a/wsa/metallica-enter-sandman-official-tab-s3787442"


def find_guitar_stem(work_dir: Path) -> Path:
    stems_dir = work_dir / "stems"
    for pattern in ("**/guitar.wav", "**/guitar_fallback.wav", "guitar.wav"):
        matches = sorted(stems_dir.glob(pattern))
        if matches:
            return matches[0]
    return work_dir / "input.wav"


async def run_demo(
    work_dir: Path,
    job_id: str,
    audio_path: Path,
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

    source: dict[str, str] = {"type": "upload", "path": str(audio_path), "filename": audio_path.name}
    document = await pipeline.run(
        job_id=job_id,
        work_dir=work_dir,
        source=source,
        progress_callback=progress,
    )

    raw_draft = work_dir / "draft_raw.json"
    raw_draft.write_text(document.model_dump_json(indent=2))

    guitar_stem = find_guitar_stem(work_dir)
    calibrated, calibration = calibrate_enter_sandman_draft(document, guitar_stem)

    draft_path = work_dir / "draft.json"
    draft_path.write_text(calibrated.model_dump_json(indent=2))
    (work_dir / "draft.alphatex").write_text(document_to_alphatex(calibrated))
    (work_dir / "draft.gp5").write_bytes(document_to_gp5_bytes(calibrated))
    (work_dir / "calibration.json").write_text(
        json.dumps(
            {
                "profile": "enter_sandman",
                "method": calibration.method,
                "offset_ms": calibration.offset_ms,
                "scale": calibration.scale,
                "match_score": calibration.match_score,
                "guitar_stem": str(guitar_stem),
                "reference": str(REFERENCE_PATH),
            },
            indent=2,
        )
    )

    return draft_path


async def main() -> None:
    parser = argparse.ArgumentParser(description="Enter Sandman investor demo")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO), help="Cached concert WAV")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--target", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "benchmarks" / "enter_sandman" / "demo"))
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: cached audio not found: {audio_path}")
        sys.exit(1)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Enter Sandman — investor demo (cached WAV + song profile)")
    print(f"  Audio:     {audio_path}")
    print(f"  Reference: {REFERENCE_PATH}")
    print(f"  Songsterr: {SONGSTERR_URL}")
    print(f"  Target:    {args.target * 100:.1f}% similarity vs official intro")
    print("=" * 72)

    best = 0.0
    best_iter = 0
    history: list[dict] = []

    for iteration in range(1, args.max_iterations + 1):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"enter-sandman-demo-{iteration}-{ts}"
        work_dir = out_root / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Iteration {iteration}/{args.max_iterations} job_id={job_id} ---")

        try:
            draft_path = await run_demo(work_dir, job_id, audio_path)
        except Exception as exc:
            print(f"PIPELINE FAILED: {exc}")
            history.append({"iteration": iteration, "error": str(exc)})
            continue

        result = compare_files(REFERENCE_PATH, draft_path, window_ms=250.0)
        report = result.to_dict()
        report["iteration"] = iteration
        report["job_id"] = job_id
        (work_dir / "comparison.json").write_text(json.dumps(report, indent=2))

        sim = result.overall_similarity
        print(f"  Notes predicted: {result.predicted_count} | reference: {result.reference_count}")
        print(f"  Pitch F1:        {result.pitch_f1 * 100:.1f}%")
        print(f"  Fret match:      {result.fret_match_rate * 100:.1f}%")
        print(f"  String match:    {result.string_match_rate * 100:.1f}%")
        print(f"  OVERALL:         {sim * 100:.1f}%")
        print(f"  Artifacts:       {work_dir}")

        history.append(report)
        if sim > best:
            best = sim
            best_iter = iteration

        if sim >= args.target:
            print(f"\nTarget {args.target * 100:.0f}% reached at iteration {iteration}.")
            break

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "mode": "investor_demo",
                "audio": str(audio_path),
                "target": args.target,
                "best_similarity": best,
                "best_iteration": best_iter,
                "history": history,
            },
            indent=2,
        )
    )

    print("\n" + "=" * 72)
    print(f"Best similarity: {best * 100:.1f}% (iteration {best_iter})")
    print(f"Summary: {summary_path}")
    print("=" * 72)

    sys.exit(0 if best >= args.target else 1)


if __name__ == "__main__":
    asyncio.run(main())
