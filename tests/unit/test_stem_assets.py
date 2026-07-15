from __future__ import annotations

import json

import pytest

from musicai_worker.stem_manifest import write_stems_manifest


def test_write_stems_manifest_includes_isolated_when_different(tmp_path) -> None:
    work_dir = tmp_path / "job"
    work_dir.mkdir()
    input_audio = work_dir / "input.wav"
    demucs = work_dir / "stems" / "htdemucs_6s" / "input" / "guitar.wav"
    isolated = work_dir / "stems" / "parts" / "guitar_solo.wav"
    demucs.parent.mkdir(parents=True)
    isolated.parent.mkdir(parents=True)
    input_audio.write_bytes(b"in")
    demucs.write_bytes(b"de")
    isolated.write_bytes(b"so")

    path = write_stems_manifest(
        work_dir,
        guitar_part="solo",
        demucs_stem=demucs,
        transcription_stem=isolated,
        input_audio=input_audio,
    )
    data = json.loads(path.read_text())
    ids = {item["id"] for item in data["items"]}
    assert ids == {"input", "demucs", "isolated"}


def test_write_stems_manifest_skips_duplicate_isolated(tmp_path) -> None:
    work_dir = tmp_path / "job"
    work_dir.mkdir()
    stem = work_dir / "stems" / "guitar.wav"
    stem.parent.mkdir(parents=True)
    stem.write_bytes(b"x")
    input_audio = work_dir / "input.wav"
    input_audio.write_bytes(b"in")

    path = write_stems_manifest(
        work_dir,
        guitar_part="combined",
        demucs_stem=stem,
        transcription_stem=stem,
        input_audio=input_audio,
    )
    data = json.loads(path.read_text())
    ids = [item["id"] for item in data["items"]]
    assert ids == ["input", "demucs"]


def test_resolve_stem_audio_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    from musicai_api.services.stem_assets import StemAssetError, resolve_stem_audio

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"secret")
    manifest = {
        "guitar_part": "combined",
        "items": [{"id": "demucs", "label": "x", "relative_path": "../secret.wav"}],
    }
    (work_dir / "stems").mkdir()
    (work_dir / "stems" / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(StemAssetError):
        resolve_stem_audio(work_dir, "demucs")
