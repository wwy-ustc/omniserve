"""Local speech recognition with bounded audio decoding."""

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
AUDIO_MODEL = "openai/whisper-tiny"
AUDIO_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


def audio_format(data):
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "matroska"
    if data.startswith(b"OggS"):
        return "ogg"
    if data[4:8] == b"ftyp":
        return "mov"
    if data.startswith(b"fLaC"):
        return "flac"
    if data.startswith(b"FORM") and data[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if data.startswith(b"ID3") or (
        len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    ):
        return "mp3"
    raise ValueError("无法识别音频格式。请使用 WAV、MP3、M4A、WebM、OGG 或 FLAC。")


class VoiceRecognizer:
    def __init__(self, local_files_only):
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            from imageio_ffmpeg import get_ffmpeg_exe

            self.ffmpeg = get_ffmpeg_exe()
        self.processor = AutoProcessor.from_pretrained(
            AUDIO_MODEL, revision=AUDIO_REVISION, local_files_only=local_files_only,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            AUDIO_MODEL, revision=AUDIO_REVISION, local_files_only=local_files_only, dtype=torch.float32,
        ).eval()

    def decode(self, data):
        audio_type = audio_format(data)
        # MP4 recordings can put their index after the audio, so provide seekable input.
        with TemporaryDirectory(prefix="omniserve-audio-") as folder:
            source = Path(folder) / "input"
            source.write_bytes(data)
            result = subprocess.run(
                [
                    self.ffmpeg, "-nostdin", "-v", "error",
                    "-protocol_whitelist", "file,pipe", "-f", audio_type,
                    "-i", str(source), "-map", "0:a:0",
                    "-t", str(MAX_AUDIO_SECONDS + 1), "-ac", "1",
                    "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
                ],
                capture_output=True, timeout=15, check=False,
            )
        if result.returncode or not result.stdout:
            raise ValueError("无法解码音频。请检查文件是否完整并包含音轨。")
        samples = np.frombuffer(result.stdout, dtype="<f4")
        if samples.size > SAMPLE_RATE * MAX_AUDIO_SECONDS:
            raise ValueError(f"单条录音最长 {MAX_AUDIO_SECONDS} 秒。")
        return samples

    @torch.inference_mode()
    def transcribe(self, data):
        samples = self.decode(data)
        features = self.processor(
            samples, sampling_rate=SAMPLE_RATE,
            return_attention_mask=True, return_tensors="pt",
        )
        tokens = self.model.generate(**features, task="transcribe")
        text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        if not text:
            raise ValueError("未识别到可转写的语音。")
        return text
