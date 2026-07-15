"""MediaPipe hand tracking adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path

from inference.adapters.base import BaseModelAdapter
from inference.schemas.model_io import HandFrame, HandLandmark, VisionInput, VisionOutput


class MediaPipeAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_id: str = "mediapipe/hands",
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.runtime = "local"

    def healthcheck(self) -> bool:
        try:
            import mediapipe  # noqa: F401

            return True
        except ImportError:
            return False

    async def predict(self, input_data: VisionInput) -> VisionOutput:
        return await asyncio.to_thread(self._analyze, input_data)

    def _analyze(self, input_data: VisionInput) -> VisionOutput:
        if input_data.video is None or not input_data.video.exists():
            return VisionOutput(frames=[], model_id=self.model_id, fallback_audio_only=True)

        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            return VisionOutput(frames=[], model_id=self.model_id, fallback_audio_only=True)

        cap = cv2.VideoCapture(str(input_data.video))
        if not cap.isOpened():
            return VisionOutput(frames=[], model_id=self.model_id, fallback_audio_only=True)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(int(fps / input_data.sample_fps), 1)
        frames: list[HandFrame] = []

        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            landmarks: list[HandLandmark] = []
            visibility = 0.0
            fret_zone: int | None = None

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                for lm in hand.landmark:
                    landmarks.append(
                        HandLandmark(x=lm.x, y=lm.y, z=lm.z, visibility=getattr(lm, "visibility", 1.0))
                    )
                visibility = min(l.visibility for l in landmarks) if landmarks else 0.0
                index_tip = landmarks[8] if len(landmarks) > 8 else None
                if index_tip:
                    fret_zone = self._estimate_fret_zone(index_tip.x, index_tip.y)

            timestamp_ms = (frame_idx / fps) * 1000.0
            frames.append(
                HandFrame(
                    timestamp_ms=timestamp_ms,
                    landmarks=landmarks,
                    fret_zone=fret_zone,
                    visibility_score=visibility,
                )
            )
            frame_idx += 1

        cap.release()
        hands.close()

        fallback = not frames or all(f.visibility_score < 0.3 for f in frames)
        return VisionOutput(
            frames=frames,
            model_id=self.model_id,
            fallback_audio_only=fallback,
        )

    @staticmethod
    def _estimate_fret_zone(x: float, y: float) -> int:
        zone = int((x * 0.7 + (1.0 - y) * 0.3) * 24)
        return max(0, min(zone, 24))
