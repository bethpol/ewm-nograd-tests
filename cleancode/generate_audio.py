#!/usr/bin/env python3
"""Read top to bottom: load a TTS model, synthesize speech, save a WAV.

Run --help without loading torch or either model. See README.md for examples.
This calls the local model APIs directly, without the evaluation pipelines.
"""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def generate_sparktts(model_dir, text, reference_audio, reference_text):
    import torch

    # These are local source checkouts, rather than installed Python packages.
    sys.path.insert(0, str(ROOT))  # Spark's source imports watermark.engine.
    sys.path.insert(0, str(ROOT / "models" / "Spark-TTS"))
    from cli.SparkTTS import SparkTTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. Loading Spark-TTS on {device}", flush=True)
    model = SparkTTS(model_dir=model_dir, device=device)
    model.model.eval()

    # Inside inference():
    #   BiCodec encodes the reference into global (voice) and semantic tokens.
    #   The text tokenizer packs reference transcript + target text + audio tokens.
    #   The causal LLM samples new semantic tokens, one token at a time.
    #   BiCodec decodes those new tokens with the reference voice into audio.
    print("2. Encoding the reference, generating speech tokens, decoding audio", flush=True)
    with torch.inference_mode():
        waveform = model.inference(
            text=text,
            prompt_speech_path=reference_audio,
            prompt_text=reference_text,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            max_new_tokens=3000,
            wm_enable=False,
        )
    # This API returns a NumPy waveform (despite its Tensor return annotation).
    return waveform, model.sample_rate


def generate_cosyvoice3(model_dir, text, reference_audio, reference_text):
    import torch

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "models" / "CosyVoice"))
    sys.path.insert(0, str(ROOT / "models" / "CosyVoice" / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice3

    # CosyVoice chooses CUDA when available, otherwise CPU, internally.
    print("1. Loading CosyVoice3 (automatic CUDA/CPU selection)", flush=True)
    model = CosyVoice3(model_dir=str(model_dir), fp16=False, load_trt=False, load_vllm=False)

    # CosyVoice3 requires this instruction/transcript separator in the reference
    # text. It is a model control token; it is not part of the spoken transcript.
    prompt_text = "You are a helpful assistant.<|endofprompt|>" + reference_text

    # Inside inference_zero_shot():
    #   The frontend extracts reference speech tokens, speaker embedding and mel
    #   features, and tokenizes the transcript and target text.
    #   The LLM generates speech tokens conditioned on that reference.
    #   The flow model turns tokens into mel features; HiFT produces a waveform.
    # The API is a generator. Even with stream=False, text splitting can yield
    # multiple outputs, so keep every piece in order.
    print("2. Encoding the reference, generating speech tokens, decoding audio", flush=True)
    chunks = []
    with torch.inference_mode():
        for output in model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=str(reference_audio),
            stream=False,
        ):
            chunks.append(output["tts_speech"].detach().cpu())  # [1, samples]
    if not chunks:
        raise RuntimeError("CosyVoice3 returned no audio.")
    return torch.cat(chunks, dim=-1).squeeze(0).numpy(), model.sample_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["sparktts", "cosyvoice3"])
    parser.add_argument("--text", required=True, help="The words to synthesize, not an instruction.")
    parser.add_argument("--reference-audio", required=True, type=Path, help="A real speech recording of the desired voice.")
    parser.add_argument("--reference-text", required=True, help="Exact transcript of the reference recording.")
    parser.add_argument("--model-dir", type=Path, help="Override the local checkpoint directory.")
    parser.add_argument("--output", required=True, type=Path, help="Destination WAV file.")
    args = parser.parse_args()

    if not args.text.strip() or not args.reference_text.strip():
        parser.error("Target text and reference transcript must not be empty.")
    reference_audio = args.reference_audio.expanduser().resolve()
    if not reference_audio.is_file():
        parser.error(f"Reference recording not found: {reference_audio}")
    defaults = {
        "sparktts": ROOT / "checkpoints/sparktts/Spark-TTS-0.5B",
        "cosyvoice3": ROOT / "checkpoints/cosyvoice/Fun-CosyVoice3-0.5B",
    }
    model_dir = (args.model_dir or defaults[args.model]).expanduser().resolve()
    config_name = "config.yaml" if args.model == "sparktts" else "cosyvoice3.yaml"
    if not (model_dir / config_name).is_file():
        parser.error(f"Missing {model_dir / config_name}; pass --model-dir with a local checkpoint.")

    import numpy as np
    import soundfile as sf

    generate = generate_sparktts if args.model == "sparktts" else generate_cosyvoice3
    waveform, sample_rate = generate(model_dir, args.text, reference_audio, args.reference_text)
    waveform = np.asarray(waveform).squeeze()
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError(f"Expected finite mono audio; received shape {waveform.shape}.")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the model's native sample rate; writing another rate changes pitch.
    sf.write(str(output), waveform, sample_rate, format="WAV", subtype="PCM_16")
    print(f"3. Saved {output} ({len(waveform) / sample_rate:.2f}s at {sample_rate} Hz)")


if __name__ == "__main__":
    main()
