"""Estimate and plot Spark-TTS's aligned semantic retokenization channel.

generate: ordinary sampling -> waveform -> semantic re-encoding -> counts.
build: rebuild matrices and plots from already-saved token pairs.
Rows are original IDs, columns are recovered IDs; diagonal matches are included.
"""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def heldout_score(matrix_path, heldout_counts):
    """Predictive log likelihood per held-out pair; higher is better."""
    matrix = np.load(matrix_path, mmap_mode="r")
    pairs = heldout_counts.tocoo()
    probabilities = np.asarray(matrix[pairs.row, pairs.col], dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("Held-out probabilities must be positive and finite")
    n = int(pairs.data.sum())
    if not n:
        raise ValueError("Held-out data is empty")
    return {"heldout_pairs": n,
            "heldout_mean_log_probability": float(pairs.data @ np.log(probabilities) / n)}


def checkpoint(args, samples, completed, heldout):
    """Publish a checkpoint only after all its artifacts are complete."""
    out = Path(args.output_dir)
    destination = out / "checkpoints" / f"clips_{completed:05d}"
    build(SimpleNamespace(output_dir=str(destination), samples_dir=str(samples),
                          vocab_size=args.vocab_size, beta=args.beta))
    summary = json.loads((destination / "transition_metadata.json").read_text())
    result = {"training_clips": completed, "paired_tokens": summary["paired_tokens"],
              "input_vocabulary_coverage": summary["input_vocabulary_coverage"],
              "checkpoint_dir": str(destination.resolve())}
    if heldout is not None:
        result.update(heldout_score(destination / "transition_matrix.npy", heldout))
    atomic_json(destination / "evaluation.json", result)
    history = out / "checkpoint_history.jsonl"
    with history.open("a") as handle:
        handle.write(json.dumps(result) + "\n")
    atomic_json(out / "latest_checkpoint.json", result)
    print("CHECKPOINT " + json.dumps(result), flush=True)
    return result


def generate_nonempty(operation, base_seed, on_retry, max_attempts=3):
    """Retry only Spark-TTS's explicit empty semantic output, with logged seeds."""
    import torch
    for attempt in range(max_attempts):
        # Reserve distant seed ranges for retries; ordinary clip seeds stay intact.
        seed = base_seed + attempt * 10_000_000
        torch.manual_seed(seed)
        try:
            return operation(), seed, attempt
        except RuntimeError as exc:
            if str(exc) != "Spark-TTS generated no semantic tokens":
                raise
            on_retry({"seed": seed, "attempt": attempt + 1, "error": str(exc)})
            if attempt + 1 == max_attempts:
                raise RuntimeError(f"Spark-TTS produced empty semantic output on all {max_attempts} attempts") from exc


def count_transitions(samples_dir, vocab_size):
    """Count the shared prefix of each pair; report unpaired tails explicitly."""
    transitions = Counter()
    records = []
    for original_path in sorted(Path(samples_dir).glob("sample_*/original_tokens.txt")):
        recovered_path = original_path.with_name("roundtrip_tokens.txt")
        if not recovered_path.exists():
            raise FileNotFoundError(f"Incomplete sample: {recovered_path}")
        original = np.loadtxt(original_path, dtype=np.int64, ndmin=1).reshape(-1)
        recovered = np.loadtxt(recovered_path, dtype=np.int64, ndmin=1).reshape(-1)
        for tokens in (original, recovered):
            if ((tokens < 0) | (tokens >= vocab_size)).any():
                raise ValueError(f"Token outside [0, {vocab_size}) in {original_path.parent}")
        n = min(len(original), len(recovered))
        transitions.update(zip(original[:n].tolist(), recovered[:n].tolist()))
        records.append({
            "sample": original_path.parent.name, "original_tokens": len(original),
            "recovered_tokens": len(recovered), "paired_tokens": n,
            "unpaired_original": len(original) - n, "unpaired_recovered": len(recovered) - n,
            "matching_tokens": int((original[:n] == recovered[:n]).sum()),
        })
    if not transitions:
        raise ValueError("No paired semantic tokens found")
    rows, columns = np.asarray(list(transitions), dtype=np.int64).T
    counts = sparse.csr_matrix((list(transitions.values()), (rows, columns)),
                               shape=(vocab_size, vocab_size), dtype=np.int64)
    return counts, records


def save_transition_matrix(counts, path, beta, block_rows=128):
    """Write P[y|v]=(N[v,y]+beta)/(N[v]+K*beta), in bounded memory.

    With beta>0, even previously unseen rows have a valid uniform distribution.
    Unseen edges remain positive, avoiding log(0) in a later channel detector.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    k = counts.shape[0]
    if counts.shape != (k, k):
        raise ValueError("Expected square transition counts")
    totals = np.asarray(counts.sum(axis=1)).reshape(-1)
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(k, k))
    for start in range(0, k, block_rows):
        stop = min(k, start + block_rows)
        block = counts[start:stop].toarray().astype(np.float64)
        block += beta
        block /= (totals[start:stop, None] + k * beta)
        matrix[start:stop] = block
    matrix.flush()
    del matrix
    return totals


def plot_transitions(counts, output_dir, sample_count):
    # A standalone scientific figure: every observed pair has a visible marker.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    points = counts.tocoo()
    k = counts.shape[0]
    norm = (LogNorm(vmin=1, vmax=points.data.max()) if points.data.max() > 1
            else Normalize(vmin=0, vmax=1))
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    ax.plot([0, k - 1], [0, k - 1], color="#b8bec5", linewidth=.6,
            linestyle="--", zorder=0)
    scatter = ax.scatter(points.col, points.row, c=points.data, norm=norm,
                         cmap="viridis", s=10, alpha=1, linewidths=0,
                         rasterized=False, zorder=2)
    ax.set(xlim=(-.5, k - .5), ylim=(-.5, k - .5), aspect="equal",
           xlabel="Recovered semantic token ID (output v′)",
           ylabel="Generated semantic token ID (input v)")
    ax.set_title(f"Spark-TTS retokenization transitions\n"
                 f"{sample_count:,} clips · {int(counts.sum()):,} aligned token pairs · "
                 f"{counts.nnz:,} observed transitions")
    ax.tick_params(labelsize=10)
    ax.set_facecolor("#fafbfc")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=.82, pad=.025)
    colorbar.set_label("Observed transition count (log color scale)" if points.data.max() > 1
                       else "Observed transition count")
    for extension in ("png", "pdf"):
        fig.savefig(Path(output_dir) / f"transition_scatter.{extension}", dpi=300)
    plt.close(fig)


def build(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts, records = count_transitions(args.samples_dir, args.vocab_size)
    sparse.save_npz(out / "transition_counts.npz", counts)
    totals = save_transition_matrix(counts, out / "transition_matrix.npy", args.beta)
    inverse = np.divide(1., totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    empirical = sparse.diags(inverse).dot(counts).tocsr()
    sparse.save_npz(out / "transition_empirical.npz", empirical)
    np.save(out / "input_token_counts.npy", totals)
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    n = int(counts.sum())
    summary = {
        "rows": "generated/original semantic token v",
        "columns": "recovered semantic token v_prime",
        "matrix_definition": "P[v, v_prime] = (N[v, v_prime] + beta) / (N[v] + K * beta)",
        "matrix_file": "transition_matrix.npy", "dtype": "float32",
        "shape": list(counts.shape), "beta": args.beta,
        "unseen_row_policy": "uniform over the full semantic vocabulary",
        "empirical_unseen_row_policy": "zero row; see input_token_counts.npy for support",
        "samples_dir": str(Path(args.samples_dir).resolve()),
        "clips": len(records), "paired_tokens": n,
        "unique_transitions": counts.nnz, "observed_input_tokens": int((totals > 0).sum()),
        "input_vocabulary_coverage": float((totals > 0).mean()),
        "matching_pairs": int(counts.diagonal().sum()),
        "token_recovery_rate": float(counts.diagonal().sum() / n),
        "unpaired_original_tokens": sum(r["unpaired_original"] for r in records),
        "unpaired_recovered_tokens": sum(r["unpaired_recovered"] for r in records),
        "alignment": "same index on shared prefix, no shifting or synchronization search",
        "channel": "direct waveform decode/re-encode, no audio attacks",
        "scope": "empirical estimate from generated clips; not full-vocabulary calibration",
    }
    (out / "transition_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_transitions(counts, out, len(records))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Plot: {(out / 'transition_scatter.png').resolve()}", flush=True)
    print(f"Transition matrix: {(out / 'transition_matrix.npy').resolve()}", flush=True)


def generate(args):
    import soundfile as sf
    import torch
    from evals.keyed_sparktts import sparktts_classes

    out = Path(args.output_dir)
    if (out / "collection_complete.json").exists():
        raise ValueError("This collection was finalized; use a new output directory for additional generation")
    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    if list(samples.glob("sample_*")) and not args.resume:
        raise ValueError("Samples already exist; use --resume or choose another output directory")
    prompts = [p.strip() for p in Path(args.prompt_file).read_text().splitlines() if p.strip()]
    references = [(p, p.with_suffix(".txt")) for p in sorted(Path(args.reference_dir).glob("*.wav"))
                  if p.with_suffix(".txt").exists()]
    if not prompts or not references:
        raise ValueError("Need nonempty prompts and reference WAV files with matching .txt transcripts")
    metadata = {k: v for k, v in vars(args).items() if k != "func"}
    metadata.update({"sampling": "ordinary, no logit bias or keyed watermark",
                     "prompts": prompts, "references": [str(p) for p, _ in references],
                     "reference_transcripts": [t.read_text().strip() for _, t in references]})
    config_path = out / "generation_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if args.num_samples < previous["num_samples"]:
            raise ValueError("Resume cannot reduce the requested number of clips")
        for field in ("model_dir", "vocab_size", "seed", "steps", "temperature", "top_k",
                      "top_p", "prompts", "references", "reference_transcripts", "beta",
                      "heldout_samples_dir"):
            if previous.get(field) != metadata.get(field):
                raise ValueError(f"Cannot resume with different {field}")
    atomic_json(config_path, metadata)
    heldout = None
    if args.heldout_samples_dir:
        if Path(args.heldout_samples_dir).resolve() == samples.resolve():
            raise ValueError("Held-out samples must be separate from training samples")
        heldout, _ = count_transitions(args.heldout_samples_dir, args.vocab_size)
    started = time.time()
    completed = 0
    latest_path = out / "latest_checkpoint.json"
    last_checkpoint = json.loads(latest_path.read_text()) if latest_path.exists() else None

    def progress(status, error=None):
        elapsed = time.time() - started
        atomic_json(out / "progress.json", {
            "status": status, "pid": os.getpid(), "completed_clips": completed,
            "target_clips": args.num_samples, "session_elapsed_seconds": elapsed,
            "updated_unix_time": time.time(), "latest_checkpoint": last_checkpoint,
            "error": error,
        })

    progress("loading_model")
    SparkTTS, Tokenizer = sparktts_classes()
    model = SparkTTS(Path(args.model_dir), device=torch.device(args.device))
    encoder = Tokenizer(args.model_dir, torch.device(args.device),
                        audio_tokenizer=model.audio_tokenizer)
    if model.semantic_vocab_size != args.vocab_size:
        raise ValueError("Requested vocabulary size differs from the Spark-TTS checkpoint")
    progress("generating")
    for i in range(args.num_samples):
        folder = samples / f"sample_{i:04d}"
        required = ("metadata.json", "generated.wav", "original_tokens.txt", "roundtrip_tokens.txt")
        if args.resume and all((folder / f).is_file() for f in required):
            completed = i + 1
            print(f"Resume: keeping completed clip {completed}/{args.num_samples}", flush=True)
            continue
        prompt = prompts[i % len(prompts)]
        reference, transcript = references[i % len(references)]
        folder = samples / f"sample_{i:04d}"
        folder.mkdir(exist_ok=args.resume)
        def operation():
            with torch.inference_mode():
                wav, original = model.inference(
                    text=prompt, prompt_speech_path=reference,
                    prompt_text=transcript.read_text().strip(), max_new_tokens=args.steps,
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                    wm_enable=False, return_semantic_ids=True,
                )
                audio = torch.as_tensor(np.asarray(wav), dtype=torch.float32,
                                        device=args.device).reshape(1, 1, -1)
                recovered = encoder.encode_semantic(audio)
            return wav, original, audio, recovered

        def record_retry(event):
            event.update(sample=i, text=prompt, reference=str(reference))
            with (out / "generation_retries.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print("RETRY " + json.dumps(event), flush=True)

        (wav, original, audio, recovered), actual_seed, retries = generate_nonempty(
            operation, args.seed + i, record_retry)
        if not original.numel() or not recovered.numel():
            raise RuntimeError(f"Empty semantic token stream for sample {i}")
        sf.write(folder / "generated.wav", np.asarray(wav).reshape(-1), encoder.sr, subtype="FLOAT")
        np.savetxt(folder / "original_tokens.txt", original.cpu().numpy(), fmt="%d")
        np.savetxt(folder / "roundtrip_tokens.txt", recovered.cpu().numpy(), fmt="%d")
        atomic_json(folder / "metadata.json", {
            "text": prompt, "reference": str(reference), "seed": actual_seed,
            "empty_generation_retries": retries,
            "sample_rate": encoder.sr, "seconds": audio.shape[-1] / encoder.sr,
            "original_tokens": original.numel(), "recovered_tokens": recovered.numel(),
        })
        completed = i + 1
        if args.checkpoint_every and (completed == 64 or completed % args.checkpoint_every == 0):
            progress("building_checkpoint")
            last_checkpoint = checkpoint(args, samples, completed, heldout)
        progress("generating")
        print(f"Clip {i + 1}/{args.num_samples}: {original.numel()} original, "
              f"{recovered.numel()} recovered tokens", flush=True)
    progress("building_final_matrix")
    args.samples_dir = str(samples)
    build(args)
    if heldout is not None:
        evaluation = heldout_score(out / "transition_matrix.npy", heldout)
        atomic_json(out / "heldout_evaluation.json", evaluation)
    if args.checkpoint_every and (last_checkpoint is None or last_checkpoint["training_clips"] != completed):
        last_checkpoint = checkpoint(args, samples, completed, heldout)
    progress("complete")


def finalize(args):
    """Finish a stopped collection using existing complete clips, without a GPU."""
    out = Path(args.output_dir)
    samples = out / "samples"
    progress_path = out / "progress.json"
    previous = json.loads(progress_path.read_text())
    if previous["status"] not in ("stopped", "failed", "complete"):
        raise ValueError("Stop the collector before finalizing")
    counts, records = count_transitions(samples, args.vocab_size)
    completed = len(records)
    if completed != previous["completed_clips"]:
        raise ValueError("Completed-file count differs from saved progress; inspect partial samples first")
    heldout, _ = count_transitions(args.heldout_samples_dir, args.vocab_size)
    result = checkpoint(args, samples, completed, heldout)
    # Publish the final artifacts at the run root as well as in the checkpoint.
    import shutil
    for source in Path(result["checkpoint_dir"]).iterdir():
        if source.is_file():
            shutil.copyfile(source, out / source.name)
    atomic_json(out / "heldout_evaluation.json", heldout_score(out / "transition_matrix.npy", heldout))
    previous.update(status="complete", completed_clips=completed, target_clips=completed,
                    original_target_clips=previous["target_clips"], error=None,
                    latest_checkpoint=result, updated_unix_time=time.time(),
                    stop_reason="Finalized existing clips at user request; no additional generation")
    atomic_json(progress_path, previous)
    atomic_json(out / "collection_complete.json", {**result, "finalized_unix_time": time.time()})
    (out / "exit_code").write_text("0\n")
    print(f"Finalized {completed} clips without further generation.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "build", "finalize"):
        sub = commands.add_parser(name)
        sub.add_argument("--output_dir", default="outputs/sparktts_transition")
        sub.add_argument("--vocab_size", type=int, default=8192)
        sub.add_argument("--beta", type=float, default=0.0001)
        if name == "build":
            sub.add_argument("--samples_dir", required=True)
            sub.set_defaults(func=build)
        elif name == "finalize":
            sub.add_argument("--heldout_samples_dir", default="outputs/sparktts_transition_64/samples")
            sub.set_defaults(func=finalize)
        else:
            sub.add_argument("--model_dir", default="checkpoints/sparktts/Spark-TTS-0.5B")
            sub.add_argument("--device", default="cuda")
            sub.add_argument("--prompt_file", default="data/sparktts_channel_prompts.txt")
            sub.add_argument("--reference_dir", default="data/audio_prompts")
            sub.add_argument("--num_samples", type=int, default=64)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--checkpoint_every", type=int, default=0,
                             help="Save complete matrix/plot snapshots periodically, plus at clip 64")
            sub.add_argument("--heldout_samples_dir", default=None,
                             help="Separate token pairs used only for predictive evaluation")
            sub.add_argument("--steps", type=int, default=800)
            sub.add_argument("--seed", type=int, default=20261023)
            sub.add_argument("--temperature", type=float, default=1.)
            sub.add_argument("--top_k", type=int, default=50)
            sub.add_argument("--top_p", type=float, default=.95)
            sub.set_defaults(func=generate)
    args = parser.parse_args()
    if not np.isfinite(args.beta) or args.beta <= 0 or args.vocab_size <= 0:
        parser.error("Require beta > 0 and vocab_size > 0")
    if args.command == "generate" and (args.num_samples <= 0 or args.steps <= 0):
        parser.error("Require num_samples > 0 and steps > 0")
    if args.command == "generate" and args.checkpoint_every < 0:
        parser.error("checkpoint_every must be nonnegative")
    try:
        args.func(args)
    except BaseException as exc:
        if args.command == "generate":
            progress_path = Path(args.output_dir) / "progress.json"
            if progress_path.exists():
                report = json.loads(progress_path.read_text())
                report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                              updated_unix_time=time.time())
                atomic_json(progress_path, report)
        raise


if __name__ == "__main__":
    main()
