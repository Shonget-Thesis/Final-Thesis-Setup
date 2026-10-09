"""Read-only telemetry audit; no model inference or third-party dependencies."""
import argparse
import json
import math
from pathlib import Path
from statistics import correlation, mean


def ranks(values):
    return [1 + sum(x < y for x in values) + (sum(x == y for x in values) - 1) / 2
            for y in values]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    baseline = read(args.downloads / "baseline_collision_validation_metrics.json")["trials"]
    stress = read(args.downloads / "stress_condition_pilot_metrics.json")["trials"]
    telemetry = repo / "FINAL_THESIS_DELIVERABLES/metrics_and_telemetry"
    fixed = read(telemetry / "fixed_handover_timing_proof.json")["instances"]
    hybrid = read(telemetry / "calibrated_hybrid_scheduler_telemetry.json")["instances"]
    trial_key = lambda r: (r["spk_id"], r["sent_index"], r["seed"])
    base_map = {trial_key(r): r for r in baseline}
    fixed_map = {(r["utt_id"], r["condition"]): r for r in fixed}
    assert len(base_map) == len(baseline)
    assert len({trial_key(r) for r in stress}) == len(stress)
    assert all(base_map[trial_key(r)]["target_text"] == r["target_text"] for r in stress)

    def passes(instance, result, strict=False):
        ref = instance["policies"]["t=0.25"]
        ceiling = 12.5 if strict else max(12.5, ref["wer"] + 1)
        return result["wer"] <= ceiling and result["similarity"] >= ref["similarity"] - .02

    out = {"scope": "Arithmetic audit of stored rounded records, not inference reproduction",
           "correlations": {}, "matched_stress": {}, "policies": {}}
    for name, rows in [("baseline", baseline), ("stress", stress)]:
        out["correlations"][name] = {
            metric: correlation(ranks([r["M_TS"] for r in rows]), ranks([r[metric] for r in rows]))
            for metric in ["WER", "SIM", "peak_flux"]}
    for metric in ["M_TS", "WER", "SIM", "peak_flux"]:
        b = mean(base_map[trial_key(r)][metric] for r in stress)
        s = mean(r[metric] for r in stress)
        out["matched_stress"][metric] = {"baseline": b, "stress": s, "delta": s-b}
    for label in fixed[0]["policies"]:
        out["policies"][label] = {
            "recorded_mean_nfe": mean(r["policies"][label]["nfe"] for r in fixed),
            "relative_failures": sum(not passes(r, r["policies"][label]) for r in fixed),
            "strict_failures": sum(not passes(r, r["policies"][label], True) for r in fixed)}
    for name in ["snr_only", "calibrated_hybrid", "pure_dsp_acoustic"]:
        costs, failures, strict, identical = [], [], [], 0
        for row in hybrid:
            fr = fixed_map[(row["utt_id"], row["condition"])]
            if name == "snr_only":
                label = "t=0.25" if row["prompt_snr_db"] < 18 else "t=0.15"
                result = fr["policies"][label]
                cost = result["nfe"]
            else:
                result = row[name]
                cost = result.get("effective_nfe", result.get("total_nfe"))
                original = fr["policies"][result["chosen_policy"]]
                identical += all(result[k] == original[k] for k in ["wer", "similarity", "latency_sec"])
            costs.append(cost)
            if not passes(fr, result):
                failures.append(row["instance_id"])
            if not passes(fr, result, True):
                strict.append(row["instance_id"])
        out["policies"][name] = {"recorded_mean_nfe": mean(costs),
            "relative_failure_ids": failures, "strict_failure_ids": strict,
            "exact_fixed_record_matches_wer_sim_latency": identical if name != "snr_only" else None}
    out["actual_teacher_grid_times"] = {
        str(n): 1-math.cos(math.pi*n/32) for n in [1, 2, 3, 4, 6]}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
