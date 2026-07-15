"""MediaPipe hand tracking adapter (Tasks API / HandLandmarker)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from inference.adapters.base import BaseModelAdapter
from inference.adapters.mediapipe_model import default_model_cache_path, resolve_hand_landmarker_model
from inference.schemas.model_io import HandFrame, HandLandmark, VisionInput, VisionOutput

log = logging.getLogger("musicai.mediapipe")

INDEX_TIP_LANDMARK = 8
VISIBILITY_FALLBACK_THRESHOLD = 0.3


class MediaPipeAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_id: str = "mediapipe/hands",
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_path: Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = model_path
        self.runtime = "local"

    def healthcheck(self) -> bool:
        try:
            from mediapipe.tasks.python.vision import HandLandmarker  # noqa: F401
        except ImportError:
            return False

        if self.model_path and self.model_path.exists():
            return True

        try:
            return default_model_cache_path().exists()
        except Exception:
            return False

    async def predict(self, input_data: VisionInput) -> VisionOutput:
        return await asyncio.to_thread(self._analyze, input_data)

    def _empty_output(self, *, fallback: bool = True) -> VisionOutput:
        return VisionOutput(frames=[], model_id=self.model_id, fallback_audio_only=fallback)

    def _analyze(self, input_data: VisionInput) -> VisionOutput:
        if input_data.video is None or not input_data.video.exists():
            return self._empty_output()

        try:
            import cv2
            from mediapipe.tasks.python.core import base_options as base_options_module
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
            from mediapipe.tasks.python.vision.core import image as mp_image_module
        except ImportError:
            log.warning("MediaPipe/OpenCV unavailable; using audio-only vision fallback")
            return self._empty_output()

        try:
            model_file = resolve_hand_landmarker_model(self.model_path)
        except Exception as exc:
            log.warning("MediaPipe hand model unavailable: %s", exc)
            return self._empty_output()

        cap = cv2.VideoCapture(str(input_data.video))
        if not cap.isOpened():
            return self._empty_output()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(int(fps / input_data.sample_fps), 1)
        frames: list[HandFrame] = []

        options = HandLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(model_file)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        try:
            with HandLandmarker.create_from_options(options) as landmarker:
                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % frame_interval != 0:
                        frame_idx += 1
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp_image_module.Image(
                        image_format=mp_image_module.ImageFormat.SRGB,
                        data=rgb,
                    )
                    timestamp_ms = int((frame_idx / fps) * 1000.0)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    frames.append(hand_frame_from_result(result, float(timestamp_ms)))
                    frame_idx += 1
        finally:
            cap.release()

        fallback = not frames or all(f.visibility_score < VISIBILITY_FALLBACK_THRESHOLD for f in frames)
        if fallback:
            log.info("MediaPipe detected no reliable hand landmarks; audio-only fusion")
        else:
            log.info("MediaPipe hand tracking frames=%d", len(frames))
        return VisionOutput(
            frames=frames,
            model_id=self.model_id,
            fallback_audio_only=fallback,
        )

    @staticmethod
    def _estimate_fret_zone(x: float, y: float) -> int:
        zone = int((x * 0.7 + (1.0 - y) * 0.3) * 24)
        return max(0, min(zone, 24))

    def describe(self) -> dict[str, object]:
        base = super().describe()
        base["backend"] = "mediapipe_tasks"
        base["model"] = str(self.model_path) if self.model_path else str(default_model_cache_path())
        return base


def hand_frame_from_result(result: Any, timestamp_ms: float) -> HandFrame:
    landmarks: list[HandLandmark] = []
    visibility = 0.0
    fret_zone: int | None = None

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        for lm in hand:
            landmarks.append(
                HandLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=float(getattr(lm, "visibility", 1.0) or 1.0),
                )
            )
        if landmarks:
            visibility = min(lm.visibility for lm in landmarks)
            if len(landmarks) > INDEX_TIP_LANDMARK:
                tip = landmarks[INDEX_TIP_LANDMARK]
                fret_zone = MediaPipeAdapter._estimate_fret_zone(tip.x, tip.y)

    return HandFrame(
        timestamp_ms=timestamp_ms,
        landmarks=landmarks,
        fret_zone=fret_zone,
        visibility_score=visibility,
    )
