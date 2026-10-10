"""Recompute historical timing criteria; no inference or third-party packages."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "code/configs/evaluation_v1.json"
EVIDENCE = ROOT / "research/evidence/legacy/metrics_and_telemetry"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def passes(reference, result, config, strict=False):
    """WER is in percentage points; similarity is on the recorded model scale."""
    ceiling = config["strict_wer_ceiling"] if strict else max(
        config["relative_wer_floor"], reference["wer"] + config["relative_wer_allowance"])
    return (result["wer"] <= ceiling and
            result["similarity"] >= reference["similarity"] - config["similarity_allowance"])


def index_unique(rows):
    indexed = {}
    for row in rows:
        key = (row["utt_id"], row["condition"])
        if key in indexed:
            raise ValueError(f"Duplicate utterance/condition: {key}")
        indexed[key] = row
    return indexed


def replay(fixed, adaptive, config):
    if not fixed or not adaptive:
        raise ValueError("Both telemetry tables must be nonempty")
    fixed_index, adaptive_index = index_unique(fixed), index_unique(adaptive)
    if fixed_index.keys() != adaptive_index.keys():
        raise ValueError("Telemetry tables must have exactly matching utterance/condition keys")
    labels = set(fixed[0]["policies"])
    if any(set(r["policies"]) != labels for r in fixed):
        raise ValueError("Fixed records have inconsistent policy sets")
    output = {
        "scope": "Offline replay of development records; not a scheduler runtime or held-out evaluation",
        "evaluator_version": config["version"],
        "wer_unit": config["wer_unit"],
        "instances": len(fixed),
        "policies": {},
    }
    for label in sorted(labels):
        output["policies"][label] = {
            "recorded_mean_nfe": mean(r["policies"][label]["nfe"] for r in fixed),
            "relative_failures": sum(not passes(r["policies"][config["baseline_label"]], r["policies"][label], config) for r in fixed),
            "strict_failures": sum(not passes(r["policies"][config["baseline_label"]], r["policies"][label], config, True) for r in fixed),
        }
    for name in ("snr_only", "calibrated_hybrid"):
        costs, relative_ids, strict_ids, matches = [], [], [], 0
        for row in adaptive:
            original = fixed_index[(row["utt_id"], row["condition"])]
            reference = original["policies"][config["baseline_label"]]
            if name == "snr_only":
                label = config["low_snr_label"] if row["prompt_snr_db"] < config["snr_threshold_db"] else config["high_snr_label"]
                result = original["policies"][label]
                cost = result["nfe"]
            else:
                result = row[name]
                cost = result["effective_nfe"]
                chosen = original["policies"][result["chosen_policy"]]
                matches += all(result[k] == chosen[k] for k in ("wer", "similarity", "latency_sec"))
            costs.append(cost)
            if not passes(reference, result, config):
                relative_ids.append(row["instance_id"])
            if not passes(reference, result, config, True):
                strict_ids.append(row["instance_id"])
        output["policies"][name] = {
            "recorded_mean_nfe": mean(costs),
            "relative_failures": len(relative_ids),
            "strict_failures": len(strict_ids),
            "relative_failure_ids": relative_ids,
            "strict_failure_ids": strict_ids,
            "exact_fixed_record_matches_wer_sim_latency": matches if name != "snr_only" else None,
        }
    output["actual_teacher_grid_times_by_step_count"] = {
        str(n): 1 - math.cos(math.pi * n / 32) for n in (1, 2, 3, 4, 6)}
    return output


def run():
    fixed_path = EVIDENCE / "fixed_handover_timing_proof.json"
    adaptive_path = EVIDENCE / "calibrated_hybrid_scheduler_telemetry.json"
    result = replay(read(fixed_path)["instances"], read(adaptive_path)["instances"], read(CONFIG))
    result["input_sha256"] = {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (fixed_path, adaptive_path, CONFIG)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional output JSON (relative to the current directory)")
    args = parser.parse_args()
    text = json.dumps(run(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
