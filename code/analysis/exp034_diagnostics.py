"""Descriptive post-hoc diagnostics for EXP-034, outside its frozen preregistration.

Reads committed EXP-034 outputs and imports code/colab/run_invariant_residual.py
read-only so all paired speaker-cluster intervals use the exact frozen estimator.

Usage: python code/analysis/exp034_diagnostics.py [results dir]
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "colab"))
import run_invariant_residual as ir  # noqa: E402

EXP = ROOT / "research/experiments/EXP-034-invariant-residual"
RES = EXP / "results"
EXP031_DEMAND = ROOT / "research/experiments/EXP-031-snr-toh-method-comparison/results/demand_sha256.txt"


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify_choices(qrows):
    """Verify recorded per-row choices match ir.choose across all quality cases."""
    mismatches = 0
    for r in qrows:
        snr, rv, dv = ir._f(r, "prompt_snr"), ir._f(r, "r"), ir._f(r, "drift8")
        for p in ir.POLICIES:
            if r.get(f"choice_{p}") != ir.choose(p, snr, rv, dv):
                mismatches += 1
    return {"cases": len(qrows), "mismatches": mismatches}


def cohort_breakdown(qrows):
    """Compare positions 31-40 (seen in EXP-026) against fresh positions 41-60."""
    pos = {it["utt_id"]: i + 1 for i, it in enumerate(ir.cg.dev_items())}
    out = {}
    for label, lo, hi in (("pos_31_40", 31, 40), ("pos_41_60_fresh", 41, 60)):
        sub = [r for r in qrows if lo <= pos.get(r["utt_id"], 0) <= hi]
        out[label] = {
            "cases": len(sub),
            "speakers": len({r["utt_id"] for r in sub}),
            "SNR_R_minus_SNR_ONLY": ir.paired(sub, "SNR_R", "SNR_ONLY"),
            "SNR_R_minus_SNR_DRIFT": ir.paired(sub, "SNR_R", "SNR_DRIFT"),
            "SNR_R_minus_W8": ir.paired(sub, "SNR_R", "W8"),
            "means": {k: {m: v[m] for m in ("utmos", "wer_mean", "similarity", "runaway")}
                      for k, v in ir.policy_tables(sub)["all"].items() if isinstance(v, dict)},
        }
    return out


def clean_flagged_cases(qrows):
    """Inspect clean utterances sent to CPQH by SNR_R."""
    pos = {it["utt_id"]: i + 1 for i, it in enumerate(ir.cg.dev_items())}
    flagged = []
    for r in qrows:
        if r["condition"] == "clean" and r["choice_SNR_R"] == "CPQH":
            flagged.append({
                "pos": pos.get(r["utt_id"]),
                "utt_id": r["utt_id"],
                "prompt_snr": ir._f(r, "prompt_snr"),
                "r": ir._f(r, "r"),
                "needed": ir.needed(r),
                "utmos_CPH8": ir._f(r, "utmos_CPH8"),
                "utmos_CPQH": ir._f(r, "utmos_CPQH"),
                "wer_CPH8": ir._f(r, "wer_CPH8"),
                "wer_CPQH": ir._f(r, "wer_CPQH"),
                "similarity_CPH8": ir._f(r, "similarity_CPH8"),
                "similarity_CPQH": ir._f(r, "similarity_CPQH"),
            })
    return flagged


def similarity_and_runaways(qrows):
    """Extract condition-wise similarity means and WER > 1.0 runaway counts."""
    tables = ir.policy_tables(qrows)
    sim_table = {}
    runaway_table = {}
    cols = ir.POLICIES + ("SW_SNR", "ORACLE") + ir.QUAL_ARMS
    for cond, entry in tables.items():
        sim_table[cond] = {c: entry[c]["similarity"] for c in cols}
        runaway_table[cond] = {c: entry[c]["runaway"] for c in cols}
    return {"similarity_by_condition": sim_table, "runaways_by_condition": runaway_table}


def extra_contrasts(qrows):
    """Paired speaker-cluster contrasts against R_ONLY, CPQH, STU, and STUR2."""
    return {
        "SNR_R_minus_R_ONLY": ir.paired(qrows, "SNR_R", "R_ONLY"),
        "SNR_R_minus_CPQH": ir.paired(qrows, "SNR_R", "CPQH"),
        "SNR_R_minus_STU": ir.paired(qrows, "SNR_R", "STU"),
        "SNR_R_minus_STUR2": ir.paired(qrows, "SNR_R", "STUR2"),
        "STUR2_minus_CPQH": ir.paired(qrows, "STUR2", "CPQH"),
        "CPQH_minus_CPH8": ir.paired(qrows, "CPQH", "CPH8"),
    }


def main(res_dir=None):
    res_dir = Path(res_dir or RES)
    qrows = read_csv(res_dir / "quality.csv")
    demand_here = (res_dir / "demand_sha256.txt").read_text(encoding="utf-8").strip() if (res_dir / "demand_sha256.txt").exists() else ""
    demand_exp031 = EXP031_DEMAND.read_text(encoding="utf-8").strip() if EXP031_DEMAND.exists() else ""
    diag = {
        "experiment": "EXP-034",
        "choice_verification": verify_choices(qrows),
        "demand_sha256_here": demand_here,
        "demand_matches_exp031": bool(demand_here and demand_here == demand_exp031),
        "cohort_breakdown": cohort_breakdown(qrows),
        "clean_flagged_cases": clean_flagged_cases(qrows),
        **similarity_and_runaways(qrows),
        "extra_contrasts": extra_contrasts(qrows),
    }
    out_path = res_dir / "diagnostics.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(diag, indent=2) + "\n")
    print(f"[OK] Wrote {out_path}")
    return diag


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
