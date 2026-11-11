#!/usr/bin/env python3
"""Grouped watermark baseline: generate -> fit -> detect -> plot.

Only position determines the keyed seed. No shifting or synchronization search.
Detection uses saved generation-time group probabilities, not audio alone.
Read cleancode/README.md for commands and the statistical assumptions.
"""

import argparse
import csv
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def position_uniform(key, position):
    """U_t = PRF(key, t); positions start at zero and count speech tokens only."""
    digest = hmac.new(key, int(position).to_bytes(8, "big"), hashlib.sha256).digest()
    return ((int.from_bytes(digest[:8], "big") >> 12) + 0.5) / 2**52


def choose_group(mass, uniform):
    cdf = np.cumsum(mass)
    cdf[-1] = 1.0
    return int(np.searchsorted(cdf, uniform, side="right"))


def make_partition(vocab_size, num_groups, seed, pattern='random'):
    if pattern == 'random':
        return np.random.default_rng(seed).permutation(vocab_size) % num_groups
    ids = np.arange(vocab_size)
    if pattern == 'contiguous':
        return ids * num_groups // vocab_size
    if pattern == 'modulo':
        return ids % num_groups
    raise ValueError(f'Unknown grouping pattern: {pattern}')


def make_sampler(semantic_to_vocab, groups, key, watermarked):
    """A small HF hook: select a group with the key, then sample within it."""
    import torch
    from transformers import LogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    class GroupSampler(LogitsProcessor):
        def __init__(self):
            mapping = semantic_to_vocab
            if sorted(mapping) != list(range(len(mapping))):
                raise ValueError("Expected contiguous semantic token IDs.")
            self.vocab_ids = torch.tensor([mapping[t] for t in range(len(mapping))])
            self.semantic_ids = {v: t for t, v in enumerate(self.vocab_ids.tolist())}
            self.tokens, self.masses = [], []
            self.warpers = [TemperatureLogitsWarper(0.8), TopKLogitsWarper(50), TopPLogitsWarper(0.95)]

        def __call__(self, input_ids, scores):
            if scores.shape[0] != 1:
                raise ValueError("This example supports one sample at a time.")
            for warper in self.warpers:
                scores = warper(input_ids, scores)
            probabilities = torch.softmax(scores[0].double(), dim=-1)
            # An ordinary draw preserves the probability of emitting EOS or other
            # non-speech tokens. Only speech-token draws get the grouped watermark.
            candidate = int(torch.multinomial(probabilities, 1).item())
            if candidate in self.semantic_ids:
                p = probabilities[self.vocab_ids.to(scores.device)].detach().cpu()
                p /= p.sum()
                mass = np.bincount(groups, weights=p.numpy(), minlength=int(groups.max()) + 1)
                mass /= mass.sum()
                token = self.semantic_ids[candidate]
                if watermarked:
                    group = choose_group(mass, position_uniform(key, len(self.tokens)))
                    weights = p * torch.from_numpy(groups == group)
                    token = int(torch.multinomial(weights, 1).item())
                self.tokens.append(token)
                self.masses.append(mass)
                candidate = int(self.vocab_ids[token])
            # Force HF to emit the sample chosen above. Spark disables its own
            # sampling filters when keyed_sampler is supplied, avoiding double filtering.
            selected = torch.full_like(scores, -float("inf"))
            selected[0, candidate] = 0.0
            return selected

    return GroupSampler()


def load_model(args):
    import torch

    sys.path.insert(0, str(ROOT))
    if args.model == "sparktts":
        sys.path.insert(0, str(ROOT / "models/Spark-TTS"))
        from cli.SparkTTS import SparkTTS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SparkTTS(args.model_dir.resolve(), device=device)
        model.model.eval()
        return model, model.semantic_to_vocab
    sys.path[:0] = [str(ROOT / "models/CosyVoice"), str(ROOT / "models/CosyVoice/third_party/Matcha-TTS")]
    from cosyvoice.cli.cosyvoice import CosyVoice3
    model = CosyVoice3(str(args.model_dir.resolve()), fp16=False, load_trt=False, load_vllm=False)
    return model, {i: i for i in range(model.model.llm.speech_token_size)}


def synthesize(model, sampler, args, reference_text):
    """Model-specific adapters; both return waveform and original speech tokens."""
    import torch

    if args.model == "sparktts":
        audio, tokens = model.inference(
            text=args.text, prompt_speech_path=args.reference_audio.resolve(),
            prompt_text=reference_text, max_new_tokens=args.max_tokens,
            wm_enable=False, keyed_sampler=sampler, return_semantic_ids=True,
        )
        return audio, tokens.detach().cpu().numpy().reshape(-1)

    llm = model.model.llm
    ordinary_sampling_ids = llm.sampling_ids

    def grouped_sampling_ids(weighted_scores, decoded_tokens, sampling, ignore_eos=True, wm_args=None):
        # Use the same explicit temperature/top-k/top-p categorical distribution
        # as the Spark experiment. This replaces Cosy's stock repetition-aware
        # sampler in ALL conditions, including calibration and null controls.
        scores = weighted_scores.clone()
        if len(sampler.tokens) >= args.max_tokens:
            return int(llm.stop_token_ids[0])
        if ignore_eos:
            scores[llm.stop_token_ids] = -float("inf")
        return int(sampler(None, scores.unsqueeze(0))[0].argmax().item())

    llm.sampling_ids = grouped_sampling_ids
    try:
        # One unsplit target gives one continuous position counter. Call the
        # frontend explicitly so that tts() also returns original AR tokens.
        prompt = "You are a helpful assistant.<|endofprompt|>" + reference_text
        inputs = model.frontend.frontend_zero_shot(
            args.text, prompt, str(args.reference_audio.resolve()), model.sample_rate, "")
        outputs = list(model.model.tts(**inputs, stream=False))
        audio = torch.cat([o["tts_speech"].detach().cpu() for o in outputs], dim=-1)
        tokens = torch.cat([o["ar_tokens"].detach().cpu().reshape(-1) for o in outputs])
        return audio.reshape(-1).numpy(), tokens.numpy()
    finally:
        llm.sampling_ids = ordinary_sampling_ids


def recover_tokens(model, wav_path, model_name):
    if model_name == "sparktts":
        _, tokens = model.audio_tokenizer.tokenize(str(wav_path))
    else:
        # CosyVoice's own frontend resamples to 16kHz and uses speech_tokenizer_v3.
        # Its current implementation requires audio clips no longer than 30s.
        tokens, _ = model.frontend._extract_speech_token(str(wav_path))
    return tokens.detach().cpu().numpy().reshape(-1)


def save_sample_index(out, manifest):
    """Human-readable inventory, updated after every saved/recovered clip."""
    fields = ["name", "model", "role", "used_for_channel_fit", "audio_path",
              "duration_seconds", "sample_rate", "original_tokens", "observed_tokens",
              "aligned_pairs", "hit_token_cap", "text", "reference_audio", "reference_text_file"]
    fields.append("exclusion_reason")
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in manifest["records"]:
            writer.writerow(dict(
                name=record["name"], model=manifest["model"], role=record["kind"],
                used_for_channel_fit=record["kind"] == "calibration" and not record.get("exclude_from_channel_fit", False),
                **{field: record[field] for field in fields[4:11]},
                text=manifest["text"], reference_audio=manifest["reference_audio"],
                reference_text_file=manifest["reference_text_file"],
                exclusion_reason=record.get("exclusion_reason", "")))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def generate(args, append_calibration=False):
    import soundfile as sf
    import torch

    out = args.output_dir
    # Refuse mixed runs: changing a partition would invalidate old traces/channels.
    if not append_calibration and ((out / "manifest.json").exists() or (out / "audio").exists()):
        raise ValueError("This output directory already contains a run. Choose a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)
    manifest = None
    if append_calibration:
        manifest = json.loads((out / "manifest.json").read_text())
        # Freeze partition, text, reference and generation cap to isolate the
        # effect of more calibration. Evaluation clips and keys stay untouched.
        args.text = manifest["text"]
        args.seed = manifest["partition_seed"]
        args.num_groups = manifest["num_groups"]
        args.max_tokens = manifest["max_tokens"]
        args.reference_audio = Path(manifest["reference_audio"])
        args.reference_text_file = Path(manifest["reference_text_file"])
    existing_calibration = sum(r["kind"] == "calibration" for r in manifest["records"]) if manifest else 0
    torch.manual_seed(args.seed + existing_calibration)
    model, semantic_to_vocab = load_model(args)
    # CosyVoice's YAML resets torch's RNG during loading. Apply our batch seed
    # afterward so an appended batch does not replay the initial calibration.
    torch.manual_seed(args.seed + existing_calibration)
    vocab_size = len(semantic_to_vocab)
    if not 1 <= args.num_groups <= vocab_size:
        raise ValueError(f"--num-groups must lie between 1 and {vocab_size}.")
    # Baseline: balanced by token count, not by model probability or codec stability.
    if append_calibration:
        groups = np.load(out / "groups.npy")
        if len(groups) != vocab_size:
            raise ValueError("Checkpoint and saved partition vocabularies disagree.")
    else:
        groups = make_partition(vocab_size, args.num_groups, args.seed, getattr(args, 'group_pattern', 'random'))
        np.save(out / "groups.npy", groups)
    reference_text = args.reference_text_file.read_text().strip()
    if not reference_text:
        raise ValueError("Reference transcript is empty.")
    records = manifest["records"] if manifest else []
    manifest = manifest or dict(format="simple-position-grouped-v1", model=args.model, num_groups=args.num_groups,
                    partition_seed=args.seed, group_pattern=getattr(args, 'group_pattern', 'random'), text=args.text, records=records,
                    reference_audio=str(args.reference_audio.resolve()),
                    reference_text_file=str(args.reference_text_file.resolve()),
                    reference_text=reference_text, max_tokens=args.max_tokens,
                    sampling=dict(temperature=0.8, top_k=50, top_p=0.95),
                    anchor="saved per-position generation group probabilities",
                    alignment="direct indices; compare shared prefix; no synchronization")
    conditions = (("calibration", args.calibration_samples),) if append_calibration else (
        ("calibration", args.calibration_samples), ("watermarked", args.samples), ("unwatermarked", args.samples))
    for kind, count in conditions:
        start = 1 + max((int(r["name"].rsplit("_", 1)[1]) for r in records if r["kind"] == kind), default=-1)
        for index in range(start, start + count):
            name = f"{kind}_{index:03d}"
            print(f"Generating {args.model}: {name}", flush=True)
            # Fresh independent experimental keys per clip prevent seed reuse.
            # Each key's PRF input is STILL ONLY position: no history or sample ID.
            key = secrets.token_bytes(32)
            key_path = out / f"{name}.key"
            key_path.touch(mode=0o600, exist_ok=False)
            key_path.write_bytes(key)
            sampler = make_sampler(semantic_to_vocab, groups, key, kind == "watermarked")
            with torch.inference_mode():
                audio, original = synthesize(model, sampler, args, reference_text)
            if not np.array_equal(original, sampler.tokens):
                raise RuntimeError("Sampler trace and generated semantic tokens disagree.")
            audio_dir = out / "audio" / kind
            audio_dir.mkdir(parents=True, exist_ok=True)
            wav_path = audio_dir / f"{name}.wav"
            sf.write(wav_path, audio, model.sample_rate, subtype="PCM_16")
            # Re-encode the actual saved WAV, not the hidden generation tokens.
            with torch.inference_mode():
                observed = recover_tokens(model, wav_path, args.model)
            if not len(observed):
                raise RuntimeError(f"No recovered tokens in {name}.")
            np.savez_compressed(out / f"{name}.npz", original=original,
                                observed=observed, group_mass=np.asarray(sampler.masses))
            records.append(dict(name=name, kind=kind, original_tokens=len(original),
                                observed_tokens=len(observed), aligned_pairs=min(len(original), len(observed)),
                                audio_path=str(wav_path.relative_to(out)), sample_rate=model.sample_rate,
                                duration_seconds=len(audio) / model.sample_rate,
                                hit_token_cap=len(original) >= args.max_tokens))
            save_sample_index(out, manifest)
            print(f"  Saved WAV; {len(original)} original / {len(observed)} recovered tokens", flush=True)


def fit_channel(out, beta):
    """Estimate C[a,b] = P(recovered group b | original group a)."""
    manifest = json.loads((out / "manifest.json").read_text())
    groups = np.load(out / "groups.npy")
    k = manifest["num_groups"]
    counts = np.zeros((k, k), dtype=np.int64)
    training_files = []
    for record in manifest["records"]:
        if record["kind"] != "calibration" or record.get("exclude_from_channel_fit", False):
            continue  # Never fit on the watermarked or null evaluation clips.
        with np.load(out / f'{record["name"]}.npz') as trace:
            n = min(len(trace["original"]), len(trace["observed"]))
            np.add.at(counts, (groups[trace["original"][:n]], groups[trace["observed"][:n]]), 1)
            training_files.append(dict(name=record["name"], aligned_pairs=n,
                                       audio_path=record.get("audio_path")))
    if not counts.sum():
        raise ValueError("No calibration token pairs.")
    channel = (counts + beta) / (counts.sum(axis=1, keepdims=True) + k * beta)
    np.savez(out / "channel.npz", counts=counts, channel=channel, beta=beta, groups=groups)
    (out / "channel_training.json").write_text(json.dumps(dict(
        model=manifest.get("model"), beta=beta, num_groups=k,
        source_group_counts=counts.sum(1).tolist(), total_pairs=int(counts.sum()),
        calibration_files=training_files,
        excluded_calibration_files=[dict(name=r["name"], reason=r.get("exclusion_reason", ""))
                                    for r in manifest["records"] if r.get("exclude_from_channel_fit", False)]), indent=2) + "\n")
    print(f"Channel fit on {counts.sum()} pairs; source counts per group: {counts.sum(1).tolist()}")
    if (counts.sum(1) < 20).any():
        print("Some groups have fewer than 20 calibration pairs; collect more before trusting channel power.")


def log_e_path(masses, observed_groups, channel, key):
    """Draft formula: e_t = C[A_t,B_t] / sum_a pi_t[a] C[a,B_t]."""
    n = min(len(masses), len(observed_groups))
    expected = np.array([choose_group(masses[t], position_uniform(key, t)) for t in range(n)])
    observed = observed_groups[:n]
    denominator = (masses[:n] * channel[:, observed].T).sum(axis=1)
    increments = np.log(channel[expected, observed]) - np.log(denominator)
    return np.r_[0.0, np.cumsum(increments)]  # E_0 = 1; log(E_0) = 0.


def detect(out, alpha):
    manifest = json.loads((out / "manifest.json").read_text())
    groups = np.load(out / "groups.npy")
    with np.load(out / "channel.npz") as archive:
        if not np.array_equal(groups, archive["groups"]):
            raise ValueError("Channel and generation partitions disagree.")
        channel = archive["channel"]
    summary = []
    for record in manifest["records"]:
        if record["kind"] == "calibration":
            continue
        name = record["name"]
        with np.load(out / f"{name}.npz") as trace:
            mass = trace["group_mass"]
            observed = groups[trace["observed"]]
            original = groups[trace["original"]]
            key = (out / f"{name}.key").read_bytes()
            path = log_e_path(mass, observed, channel, key)
            # Oracle clean-token control helps separate embedding from codec loss.
            clean_path = log_e_path(mass, original, channel, key)
        np.savez(out / f"{name}.evidence.npz", roundtrip=path, original=clean_path)
        hits = np.flatnonzero(path >= np.log(1 / alpha))
        summary.append(dict(**record, compared_tokens=len(path) - 1,
                            missing_tokens=max(0, len(original) - len(observed)),
                            extra_tokens=max(0, len(observed) - len(original)),
                            final_log_e=float(path[-1]),
                            first_detection=int(hits[0]) if len(hits) else None))
    (out / "detection.json").write_text(json.dumps(dict(alpha=alpha, samples=summary), indent=2) + "\n")
    print(f"Saved detection.json and per-sample evidence paths in {out}")


def average_paths(paths):
    # Compare a fixed cohort: truncate to the shortest path, never select only
    # longer survivors as t grows. Error bars describe variation across clips.
    length = min(map(len, paths))
    values = np.stack([p[:length] for p in paths])
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.full(length, np.nan)
    return mean, sem


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((out / "detection.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 5))
    csv_rows = []
    for kind in ("watermarked", "unwatermarked"):
        records = [r for r in result["samples"] if r["kind"] == kind]
        for source, style in (("roundtrip", "-"), ("original", "--")):
            paths = []
            for record in records:
                with np.load(out / f'{record["name"]}.evidence.npz') as evidence:
                    paths.append(evidence[source].copy())
            mean, sem = average_paths(paths)
            t = np.arange(len(mean))
            line, = ax.plot(t, mean, style, label=f"{kind}, {source} (n={len(paths)})")
            if len(paths) > 1:
                step = max(1, (len(t) - 1) // 15)
                ax.errorbar(t[::step], mean[::step], yerr=sem[::step], fmt="none",
                            ecolor=line.get_color(), capsize=2, alpha=0.6)
            csv_rows.extend((kind, source, int(i), len(paths), float(m), float(s))
                            for i, m, s in zip(t, mean, sem))
    ax.axhline(np.log(1 / result["alpha"]), color="black", linestyle=":", label="log(1/alpha)")
    ax.set(xlabel="Token count (direct positions)", ylabel="Mean cumulative log e (natural log)",
           title=f'{manifest["model"]}: position-only watermark, {manifest["num_groups"]} groups\n'
                 'Error bars = ±1 SEM across clips')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"log_e.{extension}", dpi=180)
    plt.close(fig)
    with (out / "log_e.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "source", "token_count", "samples", "mean_log_e", "sem"])
        writer.writerows(csv_rows)
    print(f"Saved {out / 'log_e.png'} (also PDF and CSV)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["sparktts", "cosyvoice3"], required=True)
    parser.add_argument("--stage", choices=["all", "generate", "calibrate", "fit", "detect", "plot"], default="all")
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--group-pattern", choices=['random', 'contiguous', 'modulo'], default='random')
    parser.add_argument("--calibration-samples", type=int, default=3, help="Calibration clips to generate; additional clips with --stage calibrate.")
    parser.add_argument("--samples", type=int, default=3, help="Clips per evaluation condition.")
    parser.add_argument("--text", default="A quiet morning is a good time to take a walk, listen to the birds, and plan the day ahead.")
    parser.add_argument("--reference-audio", type=Path, default=ROOT / "data/audio_prompts/prompt_00000.wav")
    parser.add_argument("--reference-text-file", type=Path, default=ROOT / "data/audio_prompts/prompt_00000.txt")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: cleancode/watermark_samples/<model>.")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42, help="Partition and ordinary sampling seed; NOT the secret key.")
    parser.add_argument("--beta", type=float, default=0.5, help="Positive pseudocount per group transition.")
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()
    defaults = {"sparktts": "sparktts/Spark-TTS-0.5B", "cosyvoice3": "cosyvoice/Fun-CosyVoice3-0.5B"}
    args.model_dir = args.model_dir or ROOT / "checkpoints" / defaults[args.model]
    args.output_dir = args.output_dir or ROOT / "cleancode/watermark_samples" / args.model
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.output_dir.is_relative_to(ROOT / "cleancode"):
        parser.error("Keep this experiment's --output-dir inside cleancode/.")
    if (args.num_groups < 1 or args.calibration_samples < 1 or args.samples < 1
            or args.max_tokens < 1 or not args.text.strip() or not np.isfinite(args.beta)
            or args.beta <= 0 or not 0 < args.alpha < 1):
        parser.error("Require positive counts/beta, nonempty text, and 0 < alpha < 1.")
    if (args.output_dir / "manifest.json").exists():
        manifest = json.loads((args.output_dir / "manifest.json").read_text())
        if manifest["model"] != args.model:
            parser.error("This output directory belongs to a different model; select its own directory.")
    if args.stage in ("all", "generate"):
        generate(args)
    if args.stage == "calibrate":
        generate(args, append_calibration=True)
    if args.stage in ("all", "fit"):
        fit_channel(args.output_dir, args.beta)
    if args.stage in ("all", "detect"):
        detect(args.output_dir, args.alpha)
    if args.stage in ("all", "plot"):
        plot(args.output_dir)


if __name__ == "__main__":
    main()
