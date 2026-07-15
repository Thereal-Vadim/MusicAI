"""Job progress helpers tests."""

from tab_schema.job_progress import stage_label_pct, stage_progress_pct


def test_stage_progress_pct_ranges():
    assert stage_progress_pct("ingest", 0.0) == 0
    assert stage_progress_pct("ingest", 1.0) == 8
    assert stage_progress_pct("transcribe", 0.5) == 32
    assert stage_progress_pct("done", 1.0) == 100


def test_stage_label_pct():
    assert stage_label_pct("judge") == 82
    assert stage_label_pct("done") == 100
