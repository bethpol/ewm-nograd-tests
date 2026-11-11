"""Optional compression conditions for the grouped watermark experiment."""
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MP3_BITRATES = (16, 64, 128)
ENCODEC_BANDWIDTHS = (1.5, 6.0)
CODEC_NAMES = tuple(f"mp3_{rate}kbps" for rate in MP3_BITRATES) + tuple(
    f"encodec_{rate:g}kbps" for rate in ENCODEC_BANDWIDTHS
)


def mp3_transform(sample_rate):
    import torch

    bundled = ROOT / "checkpoints/ffmpeg/ffmpeg"
    ffmpeg = shutil.which("ffmpeg") or (str(bundled) if bundled.is_file() else None)
    if ffmpeg is None:
        raise FileNotFoundError("Install FFmpeg with libmp3lame or place it at checkpoints/ffmpeg/ffmpeg")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]

    def transform(audio, mask, bitrate):
        # A seekable MP3 file lets LAME write encoder delay and padding metadata.
        with TemporaryDirectory() as folder:
            encoded = Path(folder) / "audio.mp3"
            subprocess.run(command + [
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
                "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", str(encoded),
            ], input=audio.cpu().numpy().astype("<f4").tobytes(), check=True)
            decoded = subprocess.run(command + [
                "-i", str(encoded), "-ar", str(sample_rate), "-ac", "1",
                "-f", "f32le", "pipe:1",
            ], capture_output=True, check=True)
        samples = np.frombuffer(decoded.stdout, dtype="<f4").copy()
        return torch.from_numpy(samples).reshape(1, 1, -1), mask

    return transform


def encodec_transform(sample_rate):
    from torchaudio.functional import resample
    from transformers import EncodecModel

    path = ROOT / "checkpoints/encodec/encodec_24khz"
    if not path.is_dir():
        raise FileNotFoundError(f"Save the facebook/encodec_24khz checkpoint to {path}")
    model = EncodecModel.from_pretrained(path, local_files_only=True).eval()
    codec_rate = model.config.sampling_rate

    def transform(audio, mask, bandwidth):
        signal = resample(audio, sample_rate, codec_rate)
        decoded = model(signal, bandwidth=bandwidth).audio_values
        return resample(decoded, codec_rate, sample_rate), mask

    return transform


def codec_conditions(sample_rate, selected):
    for prefix, rates, factory in (
        ("mp3", MP3_BITRATES, mp3_transform),
        ("encodec", ENCODEC_BANDWIDTHS, encodec_transform),
    ):
        conditions = [(f"{prefix}_{rate:g}kbps", rate) for rate in rates
                      if f"{prefix}_{rate:g}kbps" in selected]
        if conditions:
            transform = factory(sample_rate)
            for name, rate in conditions:
                yield name, transform, rate, False
