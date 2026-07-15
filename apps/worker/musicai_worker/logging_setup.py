"""Centralized verbose logging for pipeline diagnostics."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "stage=%(stage)s | job=%(job_id)s | %(message)s"
)


class ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "stage"):
            record.stage = "-"
        if not hasattr(record, "job_id"):
            record.job_id = "-"
        return super().format(record)


class PipelineLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            record.stage = "-"
        if not hasattr(record, "job_id"):
            record.job_id = "-"
        return True


def setup_logging(
    job_id: str = "-",
    log_dir: Path | None = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    musicai_logger = logging.getLogger("musicai")
    musicai_logger.handlers.clear()
    musicai_logger.setLevel(level)
    musicai_logger.propagate = False

    formatter = ContextFormatter(LOG_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(PipelineLogFilter())
    musicai_logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{job_id}.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(PipelineLogFilter())
        musicai_logger.addHandler(file_handler)

    return musicai_logger


def get_logger(name: str, job_id: str = "-", stage: str = "-") -> logging.LoggerAdapter:
    base = logging.getLogger(f"musicai.{name}")

    class _Adapter(logging.LoggerAdapter):
        def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            extra = kwargs.setdefault("extra", {})
            extra.setdefault("job_id", self.extra.get("job_id", "-"))
            extra.setdefault("stage", self.extra.get("stage", "-"))
            return msg, kwargs

    return _Adapter(base, {"job_id": job_id, "stage": stage})


def log_artifact(log_dir: Path, name: str, payload: Any) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")
    return path


@contextmanager
def stage_timer(
    logger: logging.LoggerAdapter,
    stage: str,
    detail: str = "",
) -> Iterator[dict[str, float]]:
    metrics: dict[str, float] = {}
    start = time.perf_counter()
    logger.info("START %s %s", stage, detail)
    try:
        yield metrics
    finally:
        elapsed = time.perf_counter() - start
        metrics["elapsed_sec"] = elapsed
        logger.info("END %s elapsed=%.3fs", stage, elapsed)
