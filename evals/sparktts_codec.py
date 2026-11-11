"""Minimal Spark-TTS round-trip adapter for keyed and channel experiments."""

import tempfile

import soundfile as sf
import torch


class SparkTTSTokenizerWrapper:
    def __init__(self, model_dir, device, audio_tokenizer=None):
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        self.tok = (audio_tokenizer if audio_tokenizer is not None
                    else BiCodecTokenizer(model_dir=model_dir, device=device))
        self.sr = int(self.tok.config.get("sample_rate", 16000))
        self.device = device

    def encode_semantic(self, wav):
        audio = wav.detach().cpu().reshape(-1).numpy()
        # SoundFile avoids requiring TorchCodec for a simple temporary WAV.
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            sf.write(handle.name, audio, self.sr, subtype="FLOAT")
            _, semantic = self.tok.tokenize(handle.name)
        return semantic.squeeze(0).to(self.device)
