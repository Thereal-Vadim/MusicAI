#!/usr/bin/env python3
"""Run ground-truth tab benchmarks and write benchmarks/reports/latest.json."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from musicai_worker.fusion.scorer import FusionScorer
from tab_export.gp5_importer import gp5_to_reference_json
from tab_schema.alignment import TabAlignmentResult, align_files
from tab_schema.models import TabDocument
from tab_schema.reference import detect_reference_profile

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT.parent / path


def _find_predicted(glob_pattern: str) -> Path | None:
    root = ROOT.parent
    matches = sorted(root.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_reference(case: dict[str, object]) -> tuple[Path, dict[str, object] | None]:
    ref_gp5 = case.get("reference_gp5")
    if ref_gp5:
        gp5_path = _resolve(str(ref_gp5))
        payload = gp5_to_reference_json(gp5_path, source=case.get("reference_url"))
        return gp5_path, payload

    ref_json = _resolve(str(case["reference_json"]))
    return ref_json, None


def run_case(
    case: dict[str, object],
    *,
    predicted_path: Path | None,
    window_ms: float,
    timing_tolerance_ms: float,
    scorer: FusionScorer,
) -> dict[str, object]:
    case_id = str(case["id"])
    ref_path, ref_payload = _load_reference(case)

    if predicted_path is None:
        glob_pattern = str(case.get("predicted_glob", ""))
        predicted_path = _find_predicted(glob_pattern) if glob_pattern else None

    if predicted_path is None or not predicted_path.exists():
        return {
            "id": case_id,
            "title": case.get("title"),
            "status": "skipped",
            "reason": "predicted draft not found",
            "reference": str(ref_path),
        }

    if ref_payload is not None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(ref_payload, tmp)
            tmp_path = Path(tmp.name)
        try:
            result = scorer.benchmark_files(
                tmp_path,
                predicted_path,
                window_ms=window_ms,
                timing_tolerance_ms=timing_tolerance_ms,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        result = scorer.benchmark_files(
            ref_path,
            predicted_path,
            window_ms=window_ms,
            timing_tolerance_ms=timing_tolerance_ms,
        )

    return {
        "id": case_id,
        "title": case.get("title"),
        "status": "ok",
        "reference": str(ref_path),
        "predicted": str(predicted_path),
        "reference_url": case.get("reference_url"),
        "metrics": result.to_dict(),
    }


def run_manifest(
    manifest_path: Path,
    report_path: Path,
    *,
    predicted: Path | None = None,
    case_id: str | None = None,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    window_ms = float(manifest.get("alignment_window_ms", 180))
    timing_tolerance_ms = float(manifest.get("timing_tolerance_ms", 80))
    scorer = FusionScorer()

    cases = manifest.get("cases", [])
    if case_id:
        cases = [c for c in cases if c.get("id") == case_id]

    results = [
        run_case(
            case,
            predicted_path=predicted,
            window_ms=window_ms,
            timing_tolerance_ms=timing_tolerance_ms,
            scorer=scorer,
        )
        for case in cases
    ]

    ok_results = [r for r in results if r.get("status") == "ok"]
    avg_f1 = (
        sum(r["metrics"]["overall_f1"] for r in ok_results) / len(ok_results) if ok_results else 0.0
    )

    report = {
        "run_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "ground_truth_tabs",
        "alignment_window_ms": window_ms,
        "timing_tolerance_ms": timing_tolerance_ms,
        "cases": results,
        "summary": {
            "cases_total": len(results),
            "cases_ok": len(ok_results),
            "mean_overall_f1": round(avg_f1, 4),
            "launch_ready": avg_f1 >= 0.85 and len(ok_results) == len(results),
        },
    }

    if report_path.exists():
        try:
            prior = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            prior = {}
    else:
        prior = {}

    prior.update(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(prior, indent=2))
    return report


def benchmark_draft_document(document: TabDocument) -> TabAlignmentResult | None:
    """Benchmark a TabDocument against its detected reference profile."""
    profile = detect_reference_profile(document)
    if profile is None or not profile.path.exists():
        return None
    scorer = FusionScorer()
    return scorer.benchmark_against_reference(document, profile.path, window_ms=profile.window_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground truth tab benchmark")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "ground_truth" / "manifest.json"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "reports" / "latest.json"),
    )
    parser.add_argument("--predicted", help="Path to draft.json to score")
    parser.add_argument("--case", help="Run single case id")
    args = parser.parse_args()

    report = run_manifest(
        Path(args.manifest),
        Path(args.report),
        predicted=Path(args.predicted) if args.predicted else None,
        case_id=args.case,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
