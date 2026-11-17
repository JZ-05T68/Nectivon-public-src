"""Optional local-only WAV transcription. No downloads, cloud client or audio upload."""

from __future__ import annotations

import io
import json
import logging
import wave
from importlib.util import find_spec
from pathlib import Path

from .models import StudyError


def availability(model: Path | None) -> tuple[bool, str]:
    """Check local availability without loading a model or discovering remote resources."""
    if find_spec("vosk") is None:
        return False, "未安装本地 Vosk 语音识别，可改用人工输入。音频不会上传。"
    if model is None or not model.is_dir():
        return False, "尚未指定已下载的本地语音模型，可改用人工输入。"
    if not model.is_absolute() or str(model).startswith(("\\\\", "//")):
        return False, "语音模型必须是本机绝对路径，可改用人工输入。"
    return True, "使用本机 Vosk 模型，识别文字可编辑，确认后才保存为学习材料。"


def transcribe(data: bytes, model: Path) -> str:
    """Decode at most ten minutes of mono PCM16 WAV using an explicitly supplied local model."""
    ready, message = availability(model)
    if not ready:
        raise StudyError(message)
    if len(data) > 20 * 1024 * 1024:
        raise StudyError("录音过长，请分段录制或人工输入。")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise StudyError("请使用单声道 16 位 PCM WAV，或直接人工输入。")
            if wav.getframerate() not in {8000, 16000, 22050, 44100, 48000}:
                raise StudyError("录音采样率不支持。")
            if wav.getnframes() / wav.getframerate() > 600:
                raise StudyError("每段录音最长十分钟，请分段录制。")
            from vosk import KaldiRecognizer, Model

            recognizer = KaldiRecognizer(Model(str(model.resolve())), wav.getframerate())
            segments = []
            while block := wav.readframes(4000):
                if recognizer.AcceptWaveform(block):
                    segments.append(json.loads(recognizer.Result()).get("text", ""))
            segments.append(json.loads(recognizer.FinalResult()).get("text", ""))
    except (wave.Error, EOFError):
        raise StudyError("录音格式无法读取，请重新录音或人工输入。") from None
    except StudyError:
        raise
    except Exception as exc:  # noqa: BLE001 - local engine errors must offer the manual fallback
        logging.getLogger(__name__).error("local_voice failed type=%s", type(exc).__name__)
        raise StudyError("本地语音识别不可用，请改用人工输入。音频未上传。") from None
    result = "\n".join(segment for segment in segments if segment)
    if not result.strip():
        raise StudyError("未识别到可用文字，请人工输入。音频未上传。")
    return result
