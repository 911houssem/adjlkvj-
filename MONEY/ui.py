"""طبقة الواجهة: مُرمِّز WebSocket الخاص بالمتصفح + جسر الحالة والمقاييس.

بروتوكول الخط (WebSocket واحد):

* المتصفح → الخادم: رسائل ثنائية = PCM16 مونو 16 كيلوهرتز.
* الخادم → المتصفح: رسائل ثنائية = PCM16 مونو 24 كيلوهرتز، ورسائل نصية
  = أحداث JSON (``status`` / ``user_text`` / ``bot_text`` / ``metrics`` /
  ``error`` / ``tts_fallback``).

:class:`UIBridge` يوضع في مسارين من خط الأنابيب (قبل المجمِّع وقبل المخرج):
كائن واحد يراقب جهة المستخدم (كلام، نص، مقاييس ASR، أخطاء) والآخر يراقب
جهة الرد (بداية/نهاية الصوت، نص الرد، مقاييس LLM وTTS). يشتركان في
:class:`SessionState` لحساب زمن الاستجابة الكلي E2E.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData, TTFAMetricsData, TTFATMetricsData, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


class BrowserFrameSerializer(FrameSerializer):
    """يترجم إطارات Pipecat إلى بروتوكول المتصفح والعكس."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            # صوت خام PCM16 بالعينات نفسها (المعدّل يحدده الناقل من إعداداته).
            return bytes(frame.audio)
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message, ensure_ascii=False)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            if not data:
                return None
            return InputAudioRawFrame(
                audio=bytes(data), sample_rate=INPUT_SAMPLE_RATE, num_channels=1
            )
        # رسائل نصية من المتصفح: لا شيء مطلوب حالياً (كتم الميكروفون يتم
        # في المتصفح بإيقاف الإرسال)، فنتجاهلها مع تسجيل للتشخيص.
        logger.trace(f"رسالة نصية غير معروفة من المتصفح: {data[:200]}")
        return None


@dataclass
class SessionState:
    """حالة الجلسة المشتركة بين جسري الواجهة."""

    # نهاية كلام المستخدم (time.time) — نقطة البدء لقياس زمن الاستجابة الكلي.
    speech_end: float | None = None
    # أول عيّنة صوت من الرد في الدورة الحالية.
    first_audio_at: float | None = None
    status: str = "idle"
    # آخر قيمة لكل مقياس (تُرسَل كاملة مع كل حدث metrics ليستسهل دمجها في الواجهة).
    metrics: dict[str, float] = field(default_factory=dict)


def _bucket(processor: str) -> str | None:
    """يصنّف مصدر المقاييس حسب اسم المعالج (jais_asr / audar_tts / OpenAILLMService)."""
    name = processor.lower()
    if "asr" in name or "stt" in name:
        return "asr"
    if "tts" in name:
        return "tts"
    if "llm" in name:
        return "llm"
    return None


def _metric_updates(data: list[MetricsData]) -> Iterator[tuple[str, float]]:
    """يحوّل بيانات المقاييس إلى (اسم مقياس بالمللي ثانية، قيمة)."""
    for item in data:
        bucket = _bucket(item.processor)
        if bucket is None:
            continue
        if isinstance(item, TTFAMetricsData):
            yield f"{bucket}_ttfa_ms", item.ttfa * 1000
        elif isinstance(item, TTFATMetricsData):
            # زمن أول رمز إجابة = أول بايت + زمن التفكير (يظهر أثر reasoning).
            yield f"{bucket}_ttft_ms", item.ttfat * 1000
        elif isinstance(item, TTFBMetricsData):
            yield f"{bucket}_ttfb_ms", item.value * 1000


class UIBridge(FrameProcessor):
    """يرسل حالة الواجهة ونصوص المحادثة والمقاييس إلى المتصفح كأحداث JSON."""

    def __init__(self, state: SessionState, role: Literal["input", "output"], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._role = role

    # ------------------------------------------------------------------ #
    # معالجة الإطارات
    # ------------------------------------------------------------------ #

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        try:
            if self._role == "input":
                await self._handle_input(frame)
            else:
                await self._handle_output(frame)
        except Exception as exc:  # الواجهة لا تُسقط الجلسة أبداً
            logger.exception(f"UIBridge: فشل إرسال حدث للواجهة: {exc}")
        await self.push_frame(frame, direction)

    async def _handle_input(self, frame: Frame) -> None:
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._state.speech_end = None
            self._state.first_audio_at = None
            await self._set_status("listening")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # timestamp هنا وقت الجدار، وstop_secs هو تأخير توقف الأصوات؛
            # الفرق يعطي لحظة نهاية الكلام الفعلية التي نقيس منها E2E.
            self._state.speech_end = frame.timestamp - frame.stop_secs
            await self._set_status("thinking")
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return
            asr_ms = self._ms_since_speech_end()
            if asr_ms is not None:
                self._state.metrics["asr_ms"] = asr_ms
            await self._emit(
                {
                    "type": "user_text",
                    "text": text,
                    "asr_ms": round(asr_ms, 1) if asr_ms is not None else None,
                }
            )
            await self._emit_metrics()
        elif isinstance(frame, MetricsFrame):
            updates = dict(_metric_updates(frame.data))
            if any(key.startswith("asr") for key in updates):
                self._state.metrics.update(updates)
                await self._emit_metrics()
        elif isinstance(frame, ErrorFrame):
            await self._emit(
                {
                    "type": "error",
                    "stage": _bucket(getattr(getattr(frame, "processor", None), "name", "")) or "server",
                    "message": str(frame.error),
                }
            )

    async def _handle_output(self, frame: Frame) -> None:
        if isinstance(frame, TTSStartedFrame):
            await self._set_status("speaking")
        elif isinstance(frame, TTSStoppedFrame):
            await self._set_status("listening")
        elif isinstance(frame, TTSAudioRawFrame):
            if self._state.first_audio_at is None:
                self._state.first_audio_at = time.time()
                e2e = self._ms_since_speech_end()
                if e2e is not None:
                    self._state.metrics["e2e_ms"] = e2e
                    logger.info(f"زمن الاستجابة الكلي E2E: {e2e:.0f}ms")
                    await self._emit_metrics()
        elif isinstance(frame, TTSTextFrame):
            text = frame.text.strip()
            if text:
                await self._emit({"type": "bot_text", "text": text})
        elif isinstance(frame, MetricsFrame):
            updates = dict(_metric_updates(frame.data))
            if updates:
                self._state.metrics.update(updates)
                await self._emit_metrics()

    # ------------------------------------------------------------------ #
    # مساعدات
    # ------------------------------------------------------------------ #

    def _ms_since_speech_end(self) -> float | None:
        if self._state.speech_end is None:
            return None
        return (time.time() - self._state.speech_end) * 1000

    async def _set_status(self, status: str) -> None:
        if self._state.status == status:
            return
        self._state.status = status
        await self._emit({"type": "status", "state": status})

    async def _emit_metrics(self) -> None:
        await self._emit(
            {"type": "metrics", "metrics": {k: round(v, 1) for k, v in self._state.metrics.items()}}
        )

    async def _emit(self, payload: dict[str, Any]) -> None:
        await self.push_frame(OutputTransportMessageUrgentFrame(message=payload))
