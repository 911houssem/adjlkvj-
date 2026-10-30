"""إعدادات وكيل الصوت العربي.

كل إعدادات الأداة تأتي من متغيرات البيئة (ملف ``.env``)، وقيم افتراضية
مطابقة لما هو مذكور في ``.env.example`` حتى تعمل الأداة بعد ``cp .env.example .env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# برومبت النظام: عربي، ردود قصيرة، بدون أي تنسيق (Markdown) — النص يُقرأ صوتاً.
DEFAULT_SYSTEM_PROMPT = (
    "أنت مساعد صوتي عربي. أجب بجملة واحدة قصيرة من ثلاث إلى اثنتي عشرة كلمة، "
    "بلغة عربية فصيحة واضحة. النص الذي تكتبه يُحوَّل إلى كلام مباشرة، لذلك يُمنع "
    "إدخال أي تنسيق: لا Markdown، لا نجوم، لا قوائم، لا رموز، ولا كود. "
    "إن كان السؤال غير واضح اطلب توضيحاً في جملة واحدة."
)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"متغير البيئة {name} يجب أن يكون عدداً صحيحاً، وصلت القيمة: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"متغير البيئة {name} يجب أن يكون عدراً، وصلت القيمة: {raw!r}") from exc


@dataclass(frozen=True)
class Config:
    """لقطة للإعدادات بعد قراءتها من البيئة والتحقق منها."""

    # Jais-ASR (التفريغ الصوتي)
    asr_model: str
    # Jais-3 عبر vLLM (النموذج اللغوي)
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_reasoning_effort: str
    # Audar-TTS (النطق)
    tts_repo: str
    tts_gguf: str
    tts_gpu_layers: int
    # الصوت المرجعي لاستنساخ الصوت
    ref_wav: Path
    ref_text: str
    # الواجهة والخادم
    system_prompt: str
    host: str
    port: int
    # سرعات الصوت: المتصفح يرسل 16 كيلوهرتز ويستقبل 24 كيلوهرتز.
    audio_in_rate: int = 16000
    audio_out_rate: int = 24000


def load_config() -> Config:
    """يقرأ ``.env`` ثم يبني ``Config``، مع رسائل خطأ واضحة لكل مشكلة."""

    load_dotenv()

    ref_wav = Path(_env("REF_WAV", "ref.wav")).expanduser()
    if not ref_wav.is_file():
        raise FileNotFoundError(
            f"ملف الصوت المرجعي غير موجود: {ref_wav}. "
            "ضع ملف WAV مدته 5 إلى 15 ثانية بـ 16 كيلوهرتز (أو أي معدل) ومونو، "
            "ثم عيّن مساره في REF_WAV."
        )

    gpu_layers = _env_int("TTS_GPU_LAYERS", 99)
    if gpu_layers == 0:
        # 0 = كل الطبقات على المعالج؛ مقبول لكنه بطيء جداً، فننبّه صراحة.
        print(
            "[تحذير] TTS_GPU_LAYERS=0 يعني تشغيل TTS على المعالج (CPU)؛ "
            "زمن أول صوت سيتضاعف. استخدم 99 لتفعيل GPU."
        )

    return Config(
        asr_model=_env("ASR_MODEL", "IFM/Jais-ASR"),
        llm_model=_env("LLM_MODEL", "inception42/Jais-3-7B"),
        llm_base_url=_env("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/"),
        # vLLM لا يتحقق من المفتاح ما لم تضبط --api-key، أي قيمة تصلح.
        llm_api_key=_env("LLM_API_KEY", "sk-local"),
        llm_reasoning_effort=_env("LLM_REASONING_EFFORT", "low"),
        tts_repo=_env("TTS_REPO", "audarai/Audar-TTS-V1-Flash"),
        tts_gguf=_env("TTS_GGUF", "Audar-TTS-V1-Flash-Q4_K_M.gguf"),
        tts_gpu_layers=gpu_layers,
        ref_wav=ref_wav,
        ref_text=_env("REF_TEXT", ""),
        system_prompt=_env("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        host=_env("HOST", "127.0.0.1"),
        port=_env_int("PORT", 7860),
    )
