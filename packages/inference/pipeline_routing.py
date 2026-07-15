"""Resolve pipeline stage backends from YAML config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from inference.settings import InferenceSettings, inference_settings

FingeringOptimizer = Literal["aco", "dp"]
ENSEMBLE_PRIMARY_PREFIX = "ensemble/"


@dataclass(frozen=True)
class EnsembleSteps:
    step_1: str
    step_2: str


@dataclass(frozen=True)
class CoarseSeparationRouting:
    """Coarse stem routing with optional RoFormer → Demucs cascade."""

    primary: str
    fallbacks: tuple[str, ...] = ()
    ensemble: EnsembleSteps | None = None

    @property
    def is_ensemble(self) -> bool:
        return (
            self.primary.startswith(ENSEMBLE_PRIMARY_PREFIX)
            and self.ensemble is not None
        )

    @property
    def chain(self) -> tuple[str, ...]:
        """Linear fallback chain (excludes virtual ensemble primary)."""
        if self.is_ensemble:
            return self.fallbacks
        return (self.primary, *self.fallbacks)


@dataclass(frozen=True)
class StageRouting:
    primary: str
    fallbacks: tuple[str, ...] = ()

    @property
    def chain(self) -> tuple[str, ...]:
        return (self.primary, *self.fallbacks)


@dataclass(frozen=True)
class ACOSettings:
    n_ants: int = 24
    n_iterations: int = 50
    max_fret_span: int = 4
    alpha: float = 1.0
    beta: float = 2.5
    rho: float = 0.45
    q: float = 100.0
    open_string_bonus: float = 2.0
    string_skip_penalty: float = 3.5
    position_shift_penalty: float = 1.0
    large_shift_threshold: int = 5
    large_shift_extra: float = 4.0
    same_string_repeat_penalty: float = 2.0
    chord_time_tolerance_ms: float = 15.0
    impossible_penalty: float = 1e9


@dataclass(frozen=True)
class FingeringRouting:
    optimizer: FingeringOptimizer = "dp"
    aco: ACOSettings = field(default_factory=ACOSettings)


@dataclass(frozen=True)
class PipelineRouting:
    coarse_separation: CoarseSeparationRouting
    guitar_demix: StageRouting
    fingering: FingeringRouting

    def to_dict(self) -> dict[str, Any]:
        coarse_payload: dict[str, Any] = {
            "primary": self.coarse_separation.primary,
            "fallbacks": list(self.coarse_separation.fallbacks),
        }
        if self.coarse_separation.ensemble:
            coarse_payload["ensemble"] = {
                "step_1": self.coarse_separation.ensemble.step_1,
                "step_2": self.coarse_separation.ensemble.step_2,
            }

        return {
            "coarse_separation": coarse_payload,
            "guitar_demix": {
                "primary": self.guitar_demix.primary,
                "fallbacks": list(self.guitar_demix.fallbacks),
            },
            "fingering": {
                "optimizer": self.fingering.optimizer,
                "aco": self.fingering.aco.__dict__,
            },
        }


def _default_config_path(settings: InferenceSettings) -> Path:
    if settings.pipeline_config_path:
        return settings.pipeline_config_path
    return Path(__file__).parent / "config" / "pipeline.yaml"


def _parse_ensemble(raw: dict[str, Any]) -> EnsembleSteps | None:
    ensemble_raw = raw.get("ensemble")
    if not isinstance(ensemble_raw, dict):
        return None
    step_1 = ensemble_raw.get("step_1")
    step_2 = ensemble_raw.get("step_2")
    if step_1 and step_2:
        return EnsembleSteps(step_1=str(step_1), step_2=str(step_2))
    return None


def load_pipeline_routing(
    config_path: str | Path | None = None,
    settings: InferenceSettings | None = None,
) -> PipelineRouting:
    cfg = settings or inference_settings
    path = Path(config_path) if config_path else _default_config_path(cfg)
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) if path.exists() else {}

    coarse = raw.get("coarse_separation", {})
    demix = raw.get("guitar_demix", {})
    finger = raw.get("fingering", {})
    aco_raw = finger.get("aco", {})

    optimizer = finger.get("optimizer", cfg.fingering_optimizer)
    if optimizer not in ("aco", "dp"):
        optimizer = "dp"

    return PipelineRouting(
        coarse_separation=CoarseSeparationRouting(
            primary=coarse.get("primary", "demucs/htdemucs_6s"),
            fallbacks=tuple(coarse.get("fallbacks", [])),
            ensemble=_parse_ensemble(coarse),
        ),
        guitar_demix=StageRouting(
            primary=demix.get("primary", "casa/v1"),
            fallbacks=tuple(demix.get("fallbacks", [])),
        ),
        fingering=FingeringRouting(
            optimizer=optimizer,
            aco=ACOSettings(
                n_ants=int(aco_raw.get("n_ants", cfg.aco_n_ants)),
                n_iterations=int(aco_raw.get("n_iterations", cfg.aco_n_iterations)),
                max_fret_span=int(aco_raw.get("max_fret_span", cfg.aco_max_fret_span)),
                alpha=float(aco_raw.get("alpha", 1.0)),
                beta=float(aco_raw.get("beta", 2.5)),
                rho=float(aco_raw.get("rho", 0.45)),
                q=float(aco_raw.get("q", 100.0)),
                open_string_bonus=float(aco_raw.get("open_string_bonus", 2.0)),
                string_skip_penalty=float(aco_raw.get("string_skip_penalty", 3.5)),
            ),
        ),
    )
