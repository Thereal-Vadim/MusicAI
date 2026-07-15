"""BS-RoFormer adapter unit tests (no GPU / checkpoint required)."""

from pathlib import Path

import pytest

from inference.adapters.bs_roformer_adapter import (
    BSRoFormerAdapter,
    _BSRoFormerRuntime,
    classify_stem_filename,
    standardize_coarse_stems,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("track_(Vocals)_model.wav", "vocals"),
        ("song_vocal_stem.wav", "vocals"),
        ("mix_(Instrumental)_bs_roformer.wav", "guitar"),
        ("htdemucs_guitar.wav", "guitar"),
        ("drums_output.wav", "drums"),
        ("bass_stem.wav", "bass"),
        ("unknown.wav", None),
    ],
)
def test_classify_stem_filename(filename: str, expected: str | None) -> None:
    assert classify_stem_filename(Path(filename).stem) == expected


def test_standardize_coarse_stems_copies_to_fixed_layout(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    standard_dir = tmp_path / "standard"
    raw_dir.mkdir()
    vocals = raw_dir / "track_(Vocals)_model.wav"
    guitar = raw_dir / "track_(Instrumental)_model.wav"
    vocals.write_bytes(b"vocals-bytes")
    guitar.write_bytes(b"guitar-bytes")

    coarse = standardize_coarse_stems([vocals, guitar], standard_dir)

    assert set(coarse.keys()) == {"vocals", "guitar"}
    assert coarse["vocals"] == standard_dir / "vocals.wav"
    assert coarse["guitar"] == standard_dir / "guitar.wav"
    assert coarse["vocals"].read_bytes() == b"vocals-bytes"
    assert coarse["guitar"].read_bytes() == b"guitar-bytes"


def test_runtime_singleton_per_checkpoint_device() -> None:
    ckpt = Path("/tmp/test_model.ckpt")
    a = _BSRoFormerRuntime.get(ckpt, "mps", 256)
    b = _BSRoFormerRuntime.get(ckpt, "mps", 256)
    c = _BSRoFormerRuntime.get(ckpt, "cpu", 256)
    assert a is b
    assert a is not c
    assert not a.model_loaded


def test_adapter_healthcheck_requires_checkpoint_and_dependency(tmp_path: Path) -> None:
    adapter = BSRoFormerAdapter(checkpoint_path=Path("/nonexistent/model.ckpt"))
    assert adapter.healthcheck() is False

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"fake")
    adapter_with_ckpt = BSRoFormerAdapter(checkpoint_path=ckpt)
    assert adapter_with_ckpt.healthcheck() is _audio_separator_installed()


def _audio_separator_installed() -> bool:
    from inference.adapters.bs_roformer_adapter import _audio_separator_available

    return _audio_separator_available()


def test_adapter_describe_reports_lazy_state() -> None:
    adapter = BSRoFormerAdapter(
        checkpoint_path=Path("/tmp/model.ckpt"),
        device="mps",
    )
    info = adapter.describe()
    assert info["backend"] == "bs_roformer"
    assert info["device"] == "mps"
    assert info["model_loaded"] is False
