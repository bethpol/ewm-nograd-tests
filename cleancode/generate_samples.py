#!/usr/bin/env python3
"""Generate the same sample sentence with both models into samples/."""

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Hello! These audio samples show how the same sentence sounds with two different speech models.")
    parser.add_argument("--reference-audio", type=Path, default=ROOT / "data/audio_prompts/prompt_00000.wav")
    parser.add_argument("--reference-text-file", type=Path, default=ROOT / "data/audio_prompts/prompt_00000.txt")
    parser.add_argument("--sparktts-model-dir", type=Path, default=ROOT / "checkpoints/sparktts/Spark-TTS-0.5B")
    parser.add_argument("--cosyvoice3-model-dir", type=Path, default=ROOT / "checkpoints/cosyvoice/Fun-CosyVoice3-0.5B")
    parser.add_argument("--output-dir", type=Path, default=HERE / "samples")
    args = parser.parse_args()

    for path in (args.reference_audio, args.reference_text_file):
        if not path.expanduser().is_file():
            parser.error(f"Missing reference file: {path}")
    reference_text = args.reference_text_file.expanduser().read_text(encoding="utf-8").strip()
    if not reference_text or not args.text.strip():
        parser.error("Reference transcript and target text must not be empty.")

    # Each model gets its own process, releasing all model memory before the next.
    # Try both independently so one missing checkpoint does not prevent the other
    # model from saving a sample. Report any failure with a nonzero exit status.
    failures = []
    for model, model_dir, config in (
        ("sparktts", args.sparktts_model_dir, "config.yaml"),
        ("cosyvoice3", args.cosyvoice3_model_dir, "cosyvoice3.yaml"),
    ):
        model_dir = model_dir.expanduser().resolve()
        if not (model_dir / config).is_file():
            print(f"{model}: missing {model_dir / config}", file=sys.stderr, flush=True)
            failures.append(model)
            continue
        output = args.output_dir.expanduser().resolve() / f"{model}.wav"
        print(f"Generating {output}", flush=True)
        result = subprocess.run([
            sys.executable, str(HERE / "generate_audio.py"),
            "--model", model,
            "--model-dir", str(model_dir),
            "--text", args.text,
            "--reference-audio", str(args.reference_audio.expanduser().resolve()),
            "--reference-text", reference_text,
            "--output", str(output),
        ], check=False)
        if result.returncode:
            failures.append(model)
    if failures:
        print(f"Samples could not be generated for: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"Both samples saved in {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
