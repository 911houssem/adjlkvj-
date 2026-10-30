"""Audar-TTS-V1-Flash — خدمة النطق العربي مع استنساخ الصوت داخل Pipecat.

بنية العمل:

* :class:`AudarVoice` — النماذج المشتركة (``llama-cpp-python`` لنموذج GGUF +
  ``NeuCodec`` لفك رموز الصوت + رموز الصوت المرجعي). تُحمَّل مرة واحدة عند
  إقلاع الخادم وتُشارَك بين كل الجلسات.
* :class:`AudarTTSService` — خدمة Pipecat التي تنادي ``run_tts`` على كل جملة.

قرارات غير بديهية:

1. **GPU افتراضياً**: ``TTS_GPU_LAYERS=99`` تدفع كل الطبقات إلى GPU.
2. **بثّ على مقاطع 400 مللي ثانية**: رموز NeuCodec تخرج بسرعة 50 رمز/ثانية
   (رمز = 20 مللي ثانية صوت). نراقب التوليد أثناء عمله ونفكّ أول 20 رمزاً
   (400 مللي ثانية) فور جاهزيتها بدل انتظار اكتمال الجملة.
3. **نوافذ متداخلة عند الفك**: فكّ NeuCodec شبكة غير سببية، وفكّ كل مقطع
   منفرداً يترك أطرافاً مشوّهة؛ لذلك نعيد فكّ النافذة مع 200 مللي ثانية من
   الماضي و100 مللي ثانية من المستقبل ثم نحتفظ بالجزء المطلوب فقط.
4. **توليد في خيط منفصل**: ``llm.generate`` حاجبة (blocking)، ولا يجوز أن
   توقف حلقة asyncio؛ الخيط يضع المقاطع في طابور يصل إلى المُولِّد غير المتزامن.
5. **المقاطعة (barge-in)**: حدث ``on_audio_context_interrupted`` يضيء
   ``threading.Event`` فيخرج خيط التوليد خلال رمز واحد.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.services.tts_service import TTSService

# --------------------------------------------------------------------------- #
# ثوابت النموذج والبثّ
# --------------------------------------------------------------------------- #

OUTPUT_SAMPLE_RATE = 24000  # مخرجات NeuCodec 24 كيلوهرتز
REF_SAMPLE_RATE = 16000  # مدخلات مُرمِّز NeuCodec 16 كيلوهرتز
TOKENS_PER_SECOND = 50  # 50 رمز/ثانية → رمز واحد = 20 مللي ثانية
TOKEN_SECONDS = 1.0 / TOKENS_PER_SECOND

CHUNK_MS = 400  # حجم مقطع البثّ المطلوب
CHUNK_TOKENS = max(1, round(CHUNK_MS / 1000 * TOKENS_PER_SECOND))  # 20 رمزاً
PAST_TOKENS = 10  # 200 مللي ثانية سياق يسار يُعاد فكّها ثم تُقطع
FUTURE_TOKENS = 5  # 100 مللي ثانية سياق يمين (تؤخّر الإرسال قليلاً)
MAX_TOKENS = 2048  # سقف بطاقة النموذج ≈ 41 ثانية صوت لكل جملة
N_CTX = 4096  # يتسع لمرجع 15 ثانية (≈750 رمز) + الجملة + المخرجات

# معاملات العيّن المعتمدة في بطاقة النموذج.
SAMPLING = {"temp": 1.0, "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.1}

# بروتوكول الح conditioning في بطاقة النموذج حرفياً (لا قالب محادثة ولا grammar).
SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")
CODEC_REPO = "neuphonic/neucodec"


def build_prompt(target_text: str, ref_text: str, ref_codes: list[int]) -> str:
    """يبني موجة الشرط: نص الصوت المرجعي + رموزه ثم النص المطلوب نطقه."""
    ref = "".join(f"<|speech_{c}|>" for c in ref_codes)
    return "user: Convert the text to speech:" + ref_text + ref + target_text + "\nassistant:"


def _pcm16_bytes(wav: np.ndarray) -> bytes:
    """حوّل صوتاً عائمة النقطة إلى PCM16 little-endian كما يتوقعه المتصفح."""
    clipped = np.clip(wav, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


class AudarVoice:
    """النماذج المشتركة: GGUF عبر llama.cpp + NeuCodec + شرط الصوت المرجعي."""

    def __init__(
        self,
        *,
        llm: Any,
        codec: Any,
        ref_codes: list[int],
        ref_text: str,
        eos_id: int,
        ref_seconds: float,
    ) -> None:
        self._llm = llm
        self._codec = codec
        self._ref_codes = ref_codes
        self._ref_text = ref_text
        self._eos_id = eos_id
        self._ref_seconds = ref_seconds
        # llama-cpp غير آمن للتزامن على نفس الكائن، وكودّ NeuCodec تشاركه
        # الجلسات أيضاً؛ قفل واحد يغطي جملة كاملة (توليد + فك).
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        repo: str,
        gguf: str,
        gpu_layers: int,
        ref_wav: Path,
        ref_text: str,
    ) -> "AudarVoice":
        """يحمّل النموذجين ويشترط صوتاً مرجعياً — يُستدعى مرة واحدة عند الإقلاع."""
        import soundfile as sf
        import torch
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from neucodec import NeuCodec

        started = time.monotonic()

        # يقبل مسار ملف GGUF محلياً أو اسم ملف داخل مستودع HuggingFace.
        gguf_path = gguf if Path(gguf).is_file() else hf_hub_download(repo, gguf)
        logger.info(f"جارٍ تحميل TTS GGUF [{gguf_path}] مع GPU layers={gpu_layers}...")
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=N_CTX,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        codec = NeuCodec.from_pretrained(CODEC_REPO).eval().to(device)
        if device.type == "cpu":
            logger.warning("لا يوجد CUDA: سيعمل فكّ NeuCodec على المعالج وسيبطئ أول صوت.")

        # بطاقة النموذج توصي بمرجع 5–15 ثانية بمونو؛ نحوّل الملف إلى 16 كيلوهرتز
        # مونو بنفسنا لأن encode_code لا يحوّل القنوات ولا المعدل للتراكب.
        data, rate = sf.read(str(ref_wav), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if rate != REF_SAMPLE_RATE:
            import soxr

            data = soxr.resample(data, rate, REF_SAMPLE_RATE)
        ref_seconds = data.shape[0] / REF_SAMPLE_RATE
        if not 3.0 <= ref_seconds <= 20.0:
            logger.warning(
                f"مدة الصوت المرجعي {ref_seconds:.1f}s خارج النطاق الموصى به (5–15s)؛ "
                "استنساخ الصوت قد يكون أضعف أو أبطأ."
            )

        with torch.inference_mode():
            codes = codec.encode_code(torch.from_numpy(data)[None, None, :])
        ref_codes = [int(c) for c in codes.squeeze().tolist()]

        # البطاقة تتوقف عند رمز  خاص؛ نتحقق أنه رمز واحد ونعود إلى eos إن لم يكن.
        eot = llm.tokenize(b"", add_bos=False, special=True)
        if len(eot) == 1:
            eos_id = int(eot[0])
        else:
            logger.warning(f"رمز  لا يفرَّد ({len(eot)} رموز)؛ استخدام eos افتراضي.")
            eos_id = int(llm.token_eos())

        logger.info(
            f"TTS جاهز خلال {time.monotonic() - started:.1f}s: "
            f"{len(ref_codes)} رمز مرجعي ({ref_seconds:.1f}s) على {device.type}"
        )
        return cls(
            llm=llm,
            codec=codec,
            ref_codes=ref_codes,
            ref_text=ref_text,
            eos_id=eos_id,
            ref_seconds=ref_seconds,
        )

    # ------------------------------------------------------------------ #
    # توليد الصوت
    # ------------------------------------------------------------------ #

    def stream(self, text: str, cancel: threading.Event) -> Iterator[bytes]:
        """يولّد PCM16/24kHz متدفقاً كمقاطع ≈400 مللي ثانية.

        تعمل في خيط عمل؛ تعود فوراً عند إشارة ``cancel`` (مقاطعة المستخدم).
        """
        prompt = build_prompt(text, self._ref_text, self._ref_codes)
        with self._lock:
            prompt_ids = self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
            speech: list[int] = []  # رموز الصوت المولّدة حتى الآن
            generated = 0
            emitted = 0  # كم رمزاً حوِّل إلى صوت مُرسَل فعلاً
            buf = ""  # النص الخام المولّد (نستخرج رموز الصوت منه بالتعبير النمطي)

            def flush(final: bool) -> Iterator[bytes]:
                """يفكّ ويرسل كل مقاطع 400 مللي ثانية الجاهزة."""
                nonlocal emitted
                count = len(speech)
                while True:
                    remaining = count - emitted
                    if remaining <= 0:
                        return
                    # قبل نهاية الجملة لا نرسل جزئياً: ننتظر مقطعاً كاملاً + سياق يمين.
                    if not final and remaining < CHUNK_TOKENS + FUTURE_TOKENS:
                        return
                    win_start = max(0, emitted - PAST_TOKENS)
                    target = min(count, emitted + CHUNK_TOKENS)
                    win_end = min(count, target + FUTURE_TOKENS)
                    # حدّد الحدود قبل الفكّ حتى يتقدم المؤشر حتى لو كان المقطع فارغاً.
                    lo_ratio = emitted - win_start
                    hi_ratio = target - win_start
                    emitted = target
                    window = speech[win_start:win_end]
                    if not window:
                        continue
                    wav = self._decode(window)
                    if wav.size == 0:
                        continue
                    # إزاحة نسبية بدل افتراض 480 عينة/رمز تماماً، تتحمّل أي فرق
                    # طفيف في طول مخرجات الشبكة قرب حواف النافذة.
                    span = win_end - win_start
                    lo = int(round(wav.size * lo_ratio / span))
                    hi = int(round(wav.size * hi_ratio / span))
                    piece = wav[lo:hi]
                    if piece.size:
                        yield _pcm16_bytes(piece)

            for token_id in self._llm.generate(prompt_ids, **SAMPLING):
                if cancel.is_set():
                    return
                if token_id == self._eos_id:
                    break
                generated += 1
                if generated >= MAX_TOKENS:
                    logger.warning("TTS بلغ سقف الرموز (2048)؛ إيقاف الجملة الحالية.")
                    break
                buf += self._llm.detokenize([token_id], special=True).decode("utf-8", "ignore")
                speech = [int(m.group(1)) for m in SPEECH_RE.finditer(buf)]
                yield from flush(final=False)

            if cancel.is_set():
                return
            yield from flush(final=True)

    def _decode(self, window: list[int]) -> np.ndarray:
        """يفكّ نافذة رموز صوت إلى موجة 24 كيلوهرتز (داخل قفل النماذج)."""
        import torch

        with torch.inference_mode():
            codes = torch.tensor(window, dtype=torch.int64)[None, None, :]
            recon = self._codec.decode_code(codes)
        return recon.detach().float().cpu().numpy()[0, 0]

    def unload(self) -> None:
        """يحرّر ذاكرة النموذجين عند إيقاف الخادم."""
        import gc

        import torch

        self._llm = None
        self._codec = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AudarTTSService(TTSService):
    """خدمة Pipecat التي تحوّل جملة الرد إلى صوت عربي متدفق."""

    def __init__(self, voice: AudarVoice, *, sample_rate: int = OUTPUT_SAMPLE_RATE, **kwargs: Any) -> None:
        super().__init__(
            sample_rate=sample_rate,
            # الإطار TTSStartedFrame/TTSStoppedFrame يتولاهما الصنف الأساسي.
            push_start_frame=True,
            push_stop_frames=True,
            name="audar_tts",
            **kwargs,
        )
        self._voice = voice
        self._cancels: dict[str, threading.Event] = {}

    def can_generate_metrics(self) -> bool:
        """نقيس TTFB وTTFA لأول صوت (يُفعَّل مع ``enable_metrics``)."""
        return True

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        """المقاطعة (barge-in): المستخدم بدأ الكلام فنوقف التوليد فوراً."""
        cancel = self._cancels.get(context_id)
        if cancel:
            cancel.set()
            logger.debug("إيقاف توليد TTS بفعل مقاطعة المستخدم")

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        await self.start_tts_usage_metrics(text)

        cancel = threading.Event()
        self._cancels[context_id] = cancel
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def offer(item: Any) -> None:
            """جدولة عنصر على الحلقة من خيط التوليد."""
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass  # الحلقة أُغلقت أثناء إنهاء الجلسة

        threading.Thread(
            target=self._worker,
            args=(text, cancel, offer),
            name="audar-tts",
            daemon=True,
        ).start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                kind, payload = item
                if kind == "audio":
                    yield TTSAudioRawFrame(
                        payload, self.sample_rate, 1, context_id=context_id
                    )
                elif kind == "error":
                    # فشل هذه الجملة: نعرض نصها في الواجهة بدلاً من الصوت، ثم
                    # نمرّر الخطأ إلى الصنف الأساسي ليُسجَّل ويُعالَج.
                    await self.push_frame(
                        OutputTransportMessageUrgentFrame(
                            message={
                                "type": "tts_fallback",
                                "text": text,
                                "detail": str(payload),
                            }
                        )
                    )
                    yield ErrorFrame(error=f"Audar TTS: {payload}")
                    return
        finally:
            cancel.set()
            self._cancels.pop(context_id, None)

    def _worker(self, text: str, cancel: threading.Event, offer: Callable[[Any], None]) -> None:
        """خيط التوليد: ينادي نموذج GGUF ويمرّر المقاطع إلى حلقة asyncio."""
        try:
            for pcm in self._voice.stream(text, cancel):
                if cancel.is_set():
                    return
                offer(("audio", pcm))
        except Exception as exc:
            logger.exception(f"فشل توليد الصوت لهذه الجملة: {exc}")
            offer(("error", exc))
            return
        offer(None)
