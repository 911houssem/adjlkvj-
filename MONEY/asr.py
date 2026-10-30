"""Jais-ASR — خدمة التعرّف على الكلام العربي داخل خط أنابيب Pipecat.

يُحمَّل الموديل (``IFM/Jais-ASR``) مرة واحدة عند إقلاع الخادم عبر
:class:`JaisASREngine`، وتُنشأ جلسة ASR واحدة لكل اتصال تشارك نفس الموديل.

الخدمتان المخصصتان هنا تنفذان عقدين من Pipecat:

* ``SegmentedSTTService``: يجمّع الصوت حتى تهدأ أصوات VAD ثم ينادِ ``run_stt``
  مرة واحدة لكل جملة (بدلاً من تشغيل الموديل على كل 20 مللي ثانية).
* ``wants_wav_segments = False``: الموديل محلي ويقرأ PCM خاماً، وحاوية WAV
  ستجعله يقرأ ترويسة الـ 44 بايت كأنها صوت.
* ``can_generate_metrics() = True``: يفعّل قياس TTFB الذي يقيس زمن تحويل
  الكلام إلى نص (من نهاية الكلام حتى وصول النص).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

# يغطي جملة تصل إلى نحو 30 ثانية. لو امتلأت النتيجة يظهر truncated=True
# في سجل الموديل، فالمستخدم يعرف أن عليه تقصير كلامه.
MAX_NEW_TOKENS = 256

# مهلة تقرير TTFB بعد نهاية الكلام: تنتظر النص الأخير ثم تقيس.
TTFB_TIMEOUT_SECS = 5.0


class JaisASREngine:
    """موديل Jais-ASR محمّل مرة واحدة، مشترك بين كل جلسات الخادم."""

    def __init__(self, pipe: Any, model_id: str) -> None:
        self._pipe = pipe
        self._model_id = model_id
        # مسار الاستدعاء في ThreadPool متعاقب من جلسات متعددة؛ الموديل غير
        # آمن للتزامن المزدوج على نفس الكائن، فنؤمّن الاستدعاء.
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_id: str) -> "JaisASREngine":
        """يحمّل الموديل وخطوط الأنابيب — يُستدعى مرة واحدة عند الإقلاع."""
        import torch
        from transformers import pipeline as hf_pipeline

        started = time.monotonic()
        logger.info(f"جارٍ تحميل موديل ASR [{model_id}] (قد يستغرق دقائق أول مرة)...")

        # bf16 هو ما يوصي به بطاقة الموديل؛ بطاقات أقدم بلا دعم bf16 تستخدم fp16،
        # ومعالج بلا CUDA يستخدم float32.
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
            logger.warning("لا يوجد CUDA: سيعمل ASR على المعالج وسيكون بطيئاً.")

        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            trust_remote_code=True,
            device_map="auto",
            dtype=dtype,
        )
        logger.info(f"ASR جاهز خلال {time.monotonic() - started:.1f}s")
        return cls(pipe, model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    def transcribe(self, array: np.ndarray, sample_rate: int) -> str:
        """يفريغ موجة صوتية عائمة النقطة إلى نص (استدعاء محظور على الخيط)."""
        with self._lock:
            result = self._pipe(
                {"array": array, "sampling_rate": sample_rate},
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        return str(result.get("text", "")).strip()

    def transcribe_file(self, path: str) -> str:
        """يفريغ ملف صوت — يُستخدم لاستخراج نص الصوت المرجعي إذا كان فارغاً."""
        with self._lock:
            result = self._pipe(path, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return str(result.get("text", "")).strip()

    def unload(self) -> None:
        """يحرّر ذاكرة الموديل عند إيقاف الخادم."""
        import gc

        import torch

        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class JaisASRService(SegmentedSTTService):
    """تحويل كلام المستخدم (PCM 16 كيلوهرتز) إلى نص عربي."""

    def __init__(self, engine: JaisASREngine, *, sample_rate: int = 16000, **kwargs: Any) -> None:
        super().__init__(
            sample_rate=sample_rate,
            # صمت بسيط بعد نهاية الكلام حتى لا تُهمل الكلمة الأخيرة، لكنه يضاف
            # إلى طول المقطع فجعلته أصغر من الافتراضي (0.5) لتقليل زمن الاستجابة.
            trailing_silence_secs=0.3,
            stt_ttfb_timeout=TTFB_TIMEOUT_SECS,
            name="jais_asr",
            **kwargs,
        )
        self._engine = engine

    @property
    def wants_wav_segments(self) -> bool:
        """PCM خام وليس WAV — راجع توثيق ``SegmentedSTTService``."""
        return False

    def can_generate_metrics(self) -> bool:
        """نقيس TTFB لكل جملة (يُفعَّل مع ``enable_metrics`` في خط الأنابيب)."""
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if not audio:
            return

        array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        started = time.monotonic()
        try:
            # توليد الموديل محلي وحاجز (GIL لا يكفي مع CUDA) فننفّذه في خيط.
            text = await asyncio.to_thread(
                self._engine.transcribe, array, self.sample_rate
            )
        except Exception as exc:
            logger.exception(f"فشل التعرّف على الكلام: {exc}")
            await self.push_error(f"تعذّر تحويل الصوت إلى نص ({exc})", exception=exc)
            return

        elapsed = (time.monotonic() - started) * 1000
        if not text:
            logger.debug(f"ASR فارغ خلال {elapsed:.0f}ms")
            return

        logger.info(f"ASR خلال {elapsed:.0f}ms: {text}")
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), None)
