"""Select K from the clean sweep and run the full attack suite for both models."""
import csv
import json
from pathlib import Path
import subprocess
import sys

from codec_attacks import CODEC_NAMES
from evals.sparktts_recovery import attack_suite

CLEAN = Path("cleancode/watermark_samples/random_groups_20261110")
OUTPUT = Path("cleancode/watermark_samples/attack_sweep_20261110")
LOG = Path("cleancode/experiment_log.csv")


def update_log(experiment, out, note):
    with LOG.open(newline="") as handle:
        rows = list(csv.reader(handle))
    row = next(row for row in rows if row[0] == experiment)
    row[5] = note
    row[6] = (
        f"{out}/attack_rates.png/.pdf/.csv; "
        f"{out}/<attack>/channel.png/.pdf; "
        f"{out}/<attack>/log_e.png/.pdf; "
        f"{out}/<attack>/channel_training.json"
    )
    temporary = LOG.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    temporary.replace(LOG)


def main():
    with (CLEAN / "detection_rates.csv").open(newline="") as handle:
        rates = list(csv.DictReader(handle))
    for model, label in (("sparktts", "Spark"), ("cosyvoice3", "Cosy")):
        candidates = [r for r in rates if r["model"] == model and
                      r["kind"] == "watermarked" and r["source"] == "roundtrip"]
        if {int(r["num_groups"]) for r in candidates} != {2, 4, 8, 16}:
            raise ValueError(f"Incomplete clean results for {model}")
        best = max(candidates, key=lambda r: (float(r["rate"]), -int(r["num_groups"])))
        k = int(best["num_groups"])
        source = CLEAN / f"{model}_groups{k}"
        manifest = json.loads((source / "manifest.json").read_text())
        sample_rate = manifest["records"][0]["sample_rate"]
        attacks = [name for name, *_ in attack_suite(sample_rate)] + list(CODEC_NAMES)
        out = OUTPUT / model
        command = [sys.executable, "cleancode/attack_sweep.py",
                   "--source", str(source), "--output-dir", str(out)]
        for attack in attacks:
            command.extend(["--attack", attack])
        subprocess.run(command, check=True)
        counts = {}
        for attack in attacks:
            fit = json.loads((out / attack / "channel_training.json").read_text())
            counts[attack] = len(fit["calibration_files"])
        calibration = (f"{next(iter(counts.values()))} calibration clips per P_c"
                       if len(set(counts.values())) == 1 else
                       "Calibration clips per P_c: " + ", ".join(f"{a}={n}" for a, n in counts.items()))
        with (out / "attack_rates.csv").open(newline="") as handle:
            trials = list(csv.DictReader(handle))
        if len(trials) != 2 * len(attacks):
            raise ValueError(f"Incomplete attack results for {model}")
        update_log(f"3-{label}", out,
                   f"K={k} (highest clean audio detection rate, smaller K breaks ties). "
                   f"{calibration}. {best['alpha']} alpha. "
                   f"{trials[0]['trials']} watermarked and {trials[1]['trials']} control clips per condition. "
                   "See channel_training.json for per-condition token pair counts.")


if __name__ == "__main__":
    main()
