"""Generate and detect position-keyed Spark-TTS samples (delta=0 diagnostics).

Run `python -m evals.keyed_sparktts --help`. Read docs/keyed_sparktts.md before
interpreting matches: this first detector uses saved generation distributions.
"""

import argparse
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets

import numpy as np
import soundfile as sf
import torch

from watermark.keyed_sampling import KeyedSemanticSampler, match_statistics, replay_tokens


def read_key(path, create=False):
    """Keep the key out of command lines, traces, and result JSON files."""
    value = os.environ.get("EWM_SECRET_KEY")
    if value:
        return value.encode("utf-8")
    path = Path(path)
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids accidentally replacing an existing key.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(secrets.token_bytes(32))
    key = path.read_bytes()
    if not key:
        raise ValueError("The secret key file is empty")
    return key


def sparktts_classes():
    # Reuse the repository's source-path setup and waveform re-encoder.
    from evals.wm_sparktts import SparkTTS, SparkTTSTokenizerWrapper
    return SparkTTS, SparkTTSTokenizerWrapper


def scores_for_observation(trace, observed, key):
    """A wrong-key comparison is a diagnostic, not a calibrated null test."""
    wrong_key = hmac.new(key, b"sparktts-wrong-key-control-v1", hashlib.sha256).digest()
    return {
        "same_key": match_statistics(observed, replay_tokens(trace, key)),
        "wrong_key": match_statistics(observed, replay_tokens(trace, wrong_key)),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def generate(args):
    key = read_key(args.key_file, create=True)
    torch.manual_seed(args.seed)  # controls ordinary draws, not the secret stream
    SparkTTS, Tokenizer = sparktts_classes()
    model = SparkTTS(Path(args.model_dir), device=torch.device(args.device))
    # Share the already-loaded codec; a second copy wastes GPU memory.
    tokenizer = Tokenizer(args.model_dir, torch.device(args.device),
                          audio_tokenizer=model.audio_tokenizer)
    from training import get_dummy_augs, get_validation_augs
    augs = (get_validation_augs(sample_rate=tokenizer.sr,
                               frame_size=int(0.08 * tokenizer.sr))
            if args.eval_aug else get_dummy_augs())
    for aug, _ in augs:
        aug.to(args.device)

    prompts = [p.strip() for p in Path(args.prompt_file).read_text().splitlines() if p.strip()]
    if args.nsamples > 0:
        prompts = prompts[:args.nsamples]
    if not prompts:
        raise ValueError("No text prompts found")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, text in enumerate(prompts):
        # A fresh public nonce prevents reusing the same keyed stream across clips.
        sample_id = secrets.token_hex(16)
        sample_dir = out / f"sample_{index:04d}_{sample_id}"
        sample_dir.mkdir()
        sampler = KeyedSemanticSampler(model.semantic_to_vocab, key, sample_id,
                                       args.temperature, args.top_k, args.top_p)
        with torch.inference_mode():
            wav, original = model.inference(
                text=text, prompt_speech_path=Path(args.prompt_audio),
                prompt_text=args.prompt_text, max_new_tokens=args.steps,
                wm_enable=False, return_semantic_ids=True, keyed_sampler=sampler,
            )
        trace = sampler.trace()
        if not len(trace["original_tokens"]):
            raise RuntimeError("No semantic tokens generated; increase --steps or check the prompt")
        if not torch.equal(original.cpu(), trace["original_tokens"]):
            raise RuntimeError("Generated semantic IDs differ from the sampler trace")
        if not torch.equal(replay_tokens(trace, key), original.cpu()):
            raise RuntimeError("Original keyed samples did not replay exactly")
        trace["sampling"] = {"temperature": args.temperature, "top_k": args.top_k,
                             "top_p": args.top_p, "seed": args.seed}
        torch.save(trace, sample_dir / "trace.pt")
        np.savetxt(sample_dir / "original_tokens.txt", original.cpu().numpy(), fmt="%d")
        write_json(sample_dir / "metadata.json", {
            "sample_id": sample_id, "text": text, "model_dir": args.model_dir,
            "prompt_audio": args.prompt_audio, "prompt_text": args.prompt_text,
            "sample_rate": tokenizer.sr, "max_new_tokens": args.steps,
            "sampling": trace["sampling"], "delta": 0,
            "detector": "saved-generation-distribution replay; aligned semantic indices",
        })
        result = {"original": scores_for_observation(trace, original, key), "audio": {}}
        audio = torch.as_tensor(np.asarray(wav), dtype=torch.float32,
                                device=args.device).reshape(1, 1, -1)
        # Save original audio separately from each attack's output.
        sf.write(sample_dir / "generated.wav", audio.cpu().numpy().reshape(-1),
                 tokenizer.sr, subtype="FLOAT")
        for aug_index, (aug, strengths) in enumerate(augs):
            for strength_index, strength in enumerate(strengths):
                tag = f"aug_{aug_index:02d}_{strength_index:02d}"
                with torch.inference_mode():
                    attacked, _ = aug(audio.clone(), None, strength)
                    recovered = tokenizer.encode_semantic(attacked)
                audio_path = sample_dir / f"{tag}.wav"
                sf.write(audio_path, attacked.detach().cpu().numpy().reshape(-1),
                         tokenizer.sr, subtype="FLOAT")
                np.savetxt(sample_dir / f"{tag}_tokens.txt", recovered.cpu().numpy(), fmt="%d")
                scores = scores_for_observation(trace, recovered, key)
                result["audio"][tag] = {"augmentation": str(aug), "strength": strength,
                                         "audio": str(audio_path), **scores}
                for key_type, values in scores.items():
                    rows.append({"sample_id": sample_id, "augmentation": str(aug),
                                 "strength": str(strength), "key_type": key_type, **values})
                print(f"{audio_path}: same-key {scores['same_key']['matches']}/"
                      f"{scores['same_key']['expected_tokens']}; wrong-key "
                      f"{scores['wrong_key']['matches']}/{scores['wrong_key']['expected_tokens']}",
                      flush=True)
        write_json(sample_dir / "results.json", result)
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detect(args):
    key = read_key(args.key_file)
    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    if args.tokens:
        observed = torch.tensor(np.loadtxt(args.tokens, dtype=np.int64, ndmin=1))
    else:
        _, Tokenizer = sparktts_classes()
        tokenizer = Tokenizer(args.model_dir, torch.device(args.device))
        audio, sr = sf.read(args.audio, always_2d=True)
        if sr != tokenizer.sr or audio.shape[1] != 1:
            raise ValueError(f"Expected mono audio at {tokenizer.sr} Hz")
        with torch.inference_mode():
            observed = tokenizer.encode_semantic(torch.tensor(audio.T, dtype=torch.float32))
    result = scores_for_observation(trace, observed, key)
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "detect"):
        sub = commands.add_parser(name)
        sub.add_argument("--key_file", default=".cache/keyed_sparktts/secret.key",
                         help="Raw key bytes; EWM_SECRET_KEY overrides this file")
        sub.add_argument("--model_dir", default="checkpoints/sparktts/Spark-TTS-0.5B")
        sub.add_argument("--device", default="cuda")
        if name == "generate":
            sub.add_argument("--output_dir", default="outputs/keyed_sparktts")
            sub.add_argument("--prompt_file", default="data/tts_prompts.txt")
            sub.add_argument("--prompt_audio", required=True)
            sub.add_argument("--prompt_text", required=True)
            sub.add_argument("--nsamples", type=int, default=1)
            sub.add_argument("--steps", type=int, default=800)
            sub.add_argument("--temperature", type=float, default=1.0)
            sub.add_argument("--top_k", type=int, default=50)
            sub.add_argument("--top_p", type=float, default=0.95)
            sub.add_argument("--seed", type=int, default=42424242)
            sub.add_argument("--eval_aug", action="store_true",
                             help="Enable the existing attack suite (requires its extra codecs)")
            sub.set_defaults(func=generate)
        else:
            sub.add_argument("--trace", required=True)
            inputs = sub.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--audio")
            inputs.add_argument("--tokens", help="Plain text semantic IDs; avoids loading models")
            sub.add_argument("--output")
            sub.set_defaults(func=detect)
    args = parser.parse_args()
    if args.command == "generate" and args.steps <= 0:
        parser.error("--steps must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
