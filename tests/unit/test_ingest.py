from __future__ import annotations

from unittest.mock import patch

from musicai_worker.ingest import _resolve_js_runtime, _yt_dlp_common_args


def test_yt_dlp_common_args_includes_youtube_extractor_and_ffmpeg() -> None:
    with patch(
        "musicai_worker.ingest.shutil.which",
        side_effect=lambda name: "/usr/bin/node" if name == "node" else None,
    ):
        args = _yt_dlp_common_args(ffmpeg="/opt/ffmpeg")

    assert args[:2] == ["--js-runtimes", "node"]
    assert "--remote-components" in args
    assert "ejs:github" in args
    assert "youtube:player_client=default,-android_sdkless" in args
    assert "--ffmpeg-location" in args
    assert args[args.index("--ffmpeg-location") + 1] == "/opt/ffmpeg"


def test_resolve_js_runtime_prefers_deno() -> None:
    with patch(
        "musicai_worker.ingest.shutil.which",
        side_effect=lambda name: "/usr/bin/deno" if name == "deno" else None,
    ):
        assert _resolve_js_runtime() == ["--js-runtimes", "deno"]


def test_resolve_js_runtime_empty_when_missing() -> None:
    with patch("musicai_worker.ingest.shutil.which", return_value=None):
        assert _resolve_js_runtime() == []
