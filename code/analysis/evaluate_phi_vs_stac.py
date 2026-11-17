"""
Comparative Evaluation: Phi_k (g_k * d_k) vs STAC (A_k) vs STAC + Veto vs Stage-Wise SNR Cascade
Under 5-Fold Speaker-Clustered Cross-Validation on EXP-012/EXP-018 Data.
"""

import csv
import math
import random
import statistics
from pathlib import Path

# Load signals (EXP-012) and ground-truth metrics (EXP-018)
sig_path = Path("research/experiments/EXP-012-consistency-gate-prelim/results/signals.csv")
step_path = Path("research/experiments/EXP-018-swsr-handover-prelim/results/steps1to8.csv")

with open(sig_path, encoding="utf-8") as f:
    sig_rows = {(r["utt_id"], r["condition"]): r for r in csv.DictReader(f)}

with open(step_path, encoding="utf-8") as f:
    step_rows = list(csv.DictReader(f))

# Candidate steps: k in {2, 3, 4} (16-grid Sway)
# NFE map: teacher step k requires 2*(k+1) CFG passes, student requires 3 passes.
# Step 2 = 9 NFE, Step 3 = 11 NFE, Step 4 = 13 NFE, Step 5 = 15 NFE, Step 6 = 17 NFE, Step 8 = 21 NFE
NFE_MAP = {2: 9, 3: 11, 4: 13, 5: 15, 6: 17, 8: 21}

cases = []
spk_list = []
for r in step_rows:
    key = (r["utt_id"], r["condition"])
    if key not in sig_rows:
        continue
    s = sig_rows[key]
    w8, s8, u8 = float(r["wer_g16_8"]), float(r["similarity_g16_8"]), float(r["utmos_g16_8"])
    step_info = {}
    for k in range(1, 7):
        wk = float(r[f"wer_g16_{k}"])
        sk = float(r[f"similarity_g16_{k}"])
        uk = float(r[f"utmos_g16_{k}"])
        # Multi-metric bounded loss L in [0, 1]
        lw = max(0.0, (wk - w8 - 0.01) / 0.10)
        ls = max(0.0, (s8 - sk - 0.015) / 0.025)
        lu = max(0.0, (u8 - uk - 0.15) / 0.50) if r["condition"] == "clean" else 0.0
        is_fail = (wk - w8 > 0.05) or (s8 - sk > 0.05)
        if r["condition"] == "clean":
            is_fail = is_fail or (u8 - uk > 0.25)
        step_info[k] = {
            "loss": min(1.0, max(lw, ls, lu)),
            "is_fail": is_fail,
            "wer": wk, "sim": sk, "utmos": uk
        }

    g = [float(s[f"g_{i}"]) for i in range(7)]
    d = [float(s[f"d_{i}"]) for i in range(1, 7)]
    spk = r["speaker_id"]
    if spk not in spk_list:
        spk_list.append(spk)

    cases.append({
        "utt": r["utt_id"], "spk": spk, "cond": r["condition"],
        "snr": float(r["prompt_snr"]), "g": g, "d": d, "step_info": step_info
    })

# 5-fold split by speaker
folds = [spk_list[i::5] for i in range(5)]

def get_signal(c, k, name):
    if name == "stac":
        return sum(c["d"][:k])
    elif name == "phi":
        return c["d"][k-1] * c["g"][k]
    elif name == "snr":
        return c["snr"]
    elif name == "veto_ratio":
        return c["g"][k] / (c["g"][0] + 1e-8)
    raise ValueError(name)

def run_5fold_evaluation(policy_name, alpha=0.05):
    """
    Calibrate on training folds, evaluate on held-out test fold.
    Returns list of dicts with 'nfe' and 'loss' per case.
    """
    oof_results = [None] * len(cases)
    
    for f_idx, test_spks in enumerate(folds):
        train = [c for c in cases if c["spk"] not in test_spks]
        test_indices = [i for i, c in enumerate(cases) if c["spk"] in test_spks]
        test = [cases[i] for i in test_indices]
        
        # Policy optimization on train:
        # We consider early-exit stages k in {3, 4} (most active savings stages)
        best_cfg = None
        best_nfe = float("inf")
        
        if policy_name == "stac":
            # Grid of thresholds for A_3 and A_4
            t3_vals = sorted([get_signal(c, 3, "stac") for c in train])
            t4_vals = sorted([get_signal(c, 4, "stac") for c in train])
            for q3 in [0.95, 0.90, 0.85, 0.80, float("inf")]:
                tau3 = t3_vals[int(q3 * (len(t3_vals)-1))] if q3 != float("inf") else float("inf")
                for q4 in [0.70, 0.60, 0.50, 0.40, 0.30, 0.20]:
                    tau4 = t4_vals[int(q4 * (len(t4_vals)-1))]
                    # Evaluate on train
                    losses, nfes = [], []
                    for c in train:
                        step = 3 if get_signal(c, 3, "stac") >= tau3 else (4 if get_signal(c, 4, "stac") >= tau4 else 8)
                        losses.append(c["step_info"][step]["loss"] if step in c["step_info"] else 0.0)
                        nfes.append(NFE_MAP[step])
                    if sum(losses)/len(losses) <= alpha and sum(nfes)/len(nfes) < best_nfe:
                        best_nfe = sum(nfes)/len(nfes)
                        best_cfg = (tau3, tau4)
            if best_cfg is None:
                best_cfg = (float("inf"), float("inf"))
            tau3, tau4 = best_cfg
            for idx in test_indices:
                c = cases[idx]
                step = 3 if get_signal(c, 3, "stac") >= tau3 else (4 if get_signal(c, 4, "stac") >= tau4 else 8)
                oof_results[idx] = {
                    "step": step, "nfe": NFE_MAP[step],
                    "loss": c["step_info"][step]["loss"] if step in c["step_info"] else 0.0,
                    "is_fail": c["step_info"][step]["is_fail"] if step in c["step_info"] else False
                }

        elif policy_name == "phi":
            # Grid of thresholds for Phi_3 and Phi_4
            t3_vals = sorted([get_signal(c, 3, "phi") for c in train])
            t4_vals = sorted([get_signal(c, 4, "phi") for c in train])
            for q3 in [0.95, 0.90, 0.85, 0.80, float("inf")]:
                tau3 = t3_vals[int(q3 * (len(t3_vals)-1))] if q3 != float("inf") else float("inf")
                for q4 in [0.70, 0.60, 0.50, 0.40, 0.30, 0.20]:
                    tau4 = t4_vals[int(q4 * (len(t4_vals)-1))]
                    losses, nfes = [], []
                    for c in train:
                        step = 3 if get_signal(c, 3, "phi") >= tau3 else (4 if get_signal(c, 4, "phi") >= tau4 else 8)
                        losses.append(c["step_info"][step]["loss"] if step in c["step_info"] else 0.0)
                        nfes.append(NFE_MAP[step])
                    if sum(losses)/len(losses) <= alpha and sum(nfes)/len(nfes) < best_nfe:
                        best_nfe = sum(nfes)/len(nfes)
                        best_cfg = (tau3, tau4)
            if best_cfg is None:
                best_cfg = (float("inf"), float("inf"))
            tau3, tau4 = best_cfg
            for idx in test_indices:
                c = cases[idx]
                step = 3 if get_signal(c, 3, "phi") >= tau3 else (4 if get_signal(c, 4, "phi") >= tau4 else 8)
                oof_results[idx] = {
                    "step": step, "nfe": NFE_MAP[step],
                    "loss": c["step_info"][step]["loss"] if step in c["step_info"] else 0.0,
                    "is_fail": c["step_info"][step]["is_fail"] if step in c["step_info"] else False
                }

        elif policy_name == "stac_veto":
            # STAC + g_k/g0 veto: require g_k/g0 >= 0.25, else force step 8
            t3_vals = sorted([get_signal(c, 3, "stac") for c in train if get_signal(c, 3, "veto_ratio") >= 0.25])
            t4_vals = sorted([get_signal(c, 4, "stac") for c in train if get_signal(c, 4, "veto_ratio") >= 0.25])
            for q3 in [0.95, 0.90, 0.85, 0.80, float("inf")]:
                tau3 = t3_vals[int(q3 * (len(t3_vals)-1))] if (q3 != float("inf") and t3_vals) else float("inf")
                for q4 in [0.70, 0.60, 0.50, 0.40, 0.30, 0.20]:
                    tau4 = t4_vals[int(q4 * (len(t4_vals)-1))] if t4_vals else float("inf")
                    losses, nfes = [], []
                    for c in train:
                        v3 = get_signal(c, 3, "veto_ratio") >= 0.25 and get_signal(c, 3, "stac") >= tau3
                        v4 = get_signal(c, 4, "veto_ratio") >= 0.25 and get_signal(c, 4, "stac") >= tau4
                        step = 3 if v3 else (4 if v4 else 8)
                        losses.append(c["step_info"][step]["loss"] if step in c["step_info"] else 0.0)
                        nfes.append(NFE_MAP[step])
                    if sum(losses)/len(losses) <= alpha and sum(nfes)/len(nfes) < best_nfe:
                        best_nfe = sum(nfes)/len(nfes)
                        best_cfg = (tau3, tau4)
            if best_cfg is None:
                best_cfg = (float("inf"), float("inf"))
            tau3, tau4 = best_cfg
            for idx in test_indices:
                c = cases[idx]
                v3 = get_signal(c, 3, "veto_ratio") >= 0.25 and get_signal(c, 3, "stac") >= tau3
                v4 = get_signal(c, 4, "veto_ratio") >= 0.25 and get_signal(c, 4, "stac") >= tau4
                step = 3 if v3 else (4 if v4 else 8)
                oof_results[idx] = {
                    "step": step, "nfe": NFE_MAP[step],
                    "loss": c["step_info"][step]["loss"] if step in c["step_info"] else 0.0,
                    "is_fail": c["step_info"][step]["is_fail"] if step in c["step_info"] else False
                }

        elif policy_name == "snr_cascade":
            # Stage-wise SNR threshold: theta3, theta4
            snr_vals = sorted([c["snr"] for c in train])
            for q3 in [0.95, 0.90, 0.85, 0.80, float("inf")]:
                th3 = snr_vals[int(q3 * (len(snr_vals)-1))] if q3 != float("inf") else float("inf")
                for q4 in [0.70, 0.60, 0.50, 0.40, 0.30, 0.20]:
                    th4 = snr_vals[int(q4 * (len(snr_vals)-1))]
                    losses, nfes = [], []
                    for c in train:
                        step = 3 if c["snr"] >= th3 else (4 if c["snr"] >= th4 else 8)
                        losses.append(c["step_info"][step]["loss"] if step in c["step_info"] else 0.0)
                        nfes.append(NFE_MAP[step])
                    if sum(losses)/len(losses) <= alpha and sum(nfes)/len(nfes) < best_nfe:
                        best_nfe = sum(nfes)/len(nfes)
                        best_cfg = (th3, th4)
            if best_cfg is None:
                best_cfg = (float("inf"), float("inf"))
            th3, th4 = best_cfg
            for idx in test_indices:
                c = cases[idx]
                step = 3 if c["snr"] >= th3 else (4 if c["snr"] >= th4 else 8)
                oof_results[idx] = {
                    "step": step, "nfe": NFE_MAP[step],
                    "loss": c["step_info"][step]["loss"] if step in c["step_info"] else 0.0,
                    "is_fail": c["step_info"][step]["is_fail"] if step in c["step_info"] else False
                }
    return oof_results

# Speaker-cluster bootstrap function
def speaker_bootstrap_eval(cases, oof_dict, B=2000, seed=42):
    spk_groups = {}
    for idx, c in enumerate(cases):
        spk_groups.setdefault(c["spk"], []).append(idx)
    groups = list(spk_groups.values())
    
    rng = random.Random(seed)
    n = len(groups)
    
    results = {}
    for name in oof_dict:
        results[name] = {"nfe": [], "risk": []}
    paired = {}
    for name in oof_dict:
        if name != "stac":
            paired[name] = {"diff_nfe": [], "diff_risk": []}
            
    for _ in range(B):
        sample_indices = []
        for _ in range(n):
            sample_indices.extend(rng.choice(groups))
            
        stac_nfe = sum(oof_dict["stac"][i]["nfe"] for i in sample_indices) / len(sample_indices)
        stac_risk = sum(oof_dict["stac"][i]["loss"] for i in sample_indices) / len(sample_indices)
        
        for name in oof_dict:
            m_nfe = sum(oof_dict[name][i]["nfe"] for i in sample_indices) / len(sample_indices)
            m_risk = sum(oof_dict[name][i]["loss"] for i in sample_indices) / len(sample_indices)
            results[name]["nfe"].append(m_nfe)
            results[name]["risk"].append(m_risk)
            if name != "stac":
                paired[name]["diff_nfe"].append(m_nfe - stac_nfe)
                paired[name]["diff_risk"].append(m_risk - stac_risk)
                
    summary = {}
    for name in oof_dict:
        nfe_vals = sorted(results[name]["nfe"])
        risk_vals = sorted(results[name]["risk"])
        pt_nfe = sum(oof_dict[name][i]["nfe"] for i in range(len(cases))) / len(cases)
        pt_risk = sum(oof_dict[name][i]["loss"] for i in range(len(cases))) / len(cases)
        summary[name] = {
            "nfe": (pt_nfe, nfe_vals[int(0.025 * B)], nfe_vals[int(0.975 * B)]),
            "risk": (pt_risk, risk_vals[int(0.025 * B)], risk_vals[int(0.975 * B)])
        }
        if name != "stac":
            dn_vals = sorted(paired[name]["diff_nfe"])
            dr_vals = sorted(paired[name]["diff_risk"])
            pt_dn = pt_nfe - summary["stac"]["nfe"][0]
            pt_dr = pt_risk - summary["stac"]["risk"][0]
            summary[name]["paired_diff_nfe"] = (pt_dn, dn_vals[int(0.025 * B)], dn_vals[int(0.975 * B)])
            summary[name]["paired_diff_risk"] = (pt_dr, dr_vals[int(0.025 * B)], dr_vals[int(0.975 * B)])
            
    return summary

# Within-condition AUC calculation
def compute_auc(scores, labels):
    n1 = sum(1 for y in labels if y)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j+1]] == scores[order[i]]:
            j += 1
        for q in range(i, j + 1):
            ranks[order[q]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = sum(ranks[i] for i, y in enumerate(labels) if y)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def auc_within_condition(cond_name, B=2000, seed=42):
    cond_cases = [c for c in cases if c["cond"] == cond_name]
    scores = [c["d"][3] * c["g"][4] for c in cond_cases] # Phi_4 (higher is safer, so -scores for failure)
    # Failure label at step 4:
    labels = [c["step_info"][4]["is_fail"] for c in cond_cases]
    # Predict failure: lower Phi_4 => higher failure probability, so score = -Phi_4
    pred_scores = [-s for s in scores]
    base_auc = compute_auc(pred_scores, labels)
    
    # Speaker cluster bootstrap
    spk_groups = {}
    for idx, c in enumerate(cond_cases):
        spk_groups.setdefault(c["spk"], []).append(idx)
    groups = list(spk_groups.values())
    rng = random.Random(seed)
    boot_aucs = []
    for _ in range(B):
        samp = []
        for _ in range(len(groups)):
            samp.extend(rng.choice(groups))
        s_scores = [pred_scores[i] for i in samp]
        s_labels = [labels[i] for i in samp]
        a = compute_auc(s_scores, s_labels)
        if a is not None:
            boot_aucs.append(a)
    boot_aucs.sort()
    if not boot_aucs or base_auc is None:
        return base_auc, None, None
    return base_auc, boot_aucs[int(0.025 * len(boot_aucs))], boot_aucs[int(0.975 * len(boot_aucs))]

if __name__ == "__main__":
    print("Running 5-fold evaluation for alpha = 0.05...")
    policies = ["stac", "phi", "stac_veto", "snr_cascade"]
    oof = {p: run_5fold_evaluation(p, alpha=0.05) for p in policies}
    stats = speaker_bootstrap_eval(cases, oof)
    
    print("\n" + "="*80)
    print("ITEM 1: POOLED OUT-OF-FOLD NFE AND RISK (alpha = 0.05, 5-Fold Speaker CV)")
    print("="*80)
    for p in policies:
        nfe_pt, nfe_lo, nfe_hi = stats[p]["nfe"]
        r_pt, r_lo, r_hi = stats[p]["risk"]
        print(f"Policy: {p:<15s} | NFE: {nfe_pt:5.2f} [{nfe_lo:5.2f}, {nfe_hi:5.2f}] | Risk: {r_pt*100:4.2f}% [{r_lo*100:4.2f}%, {r_hi*100:4.2f}%]")
        if p != "stac":
            dn_pt, dn_lo, dn_hi = stats[p]["paired_diff_nfe"]
            dr_pt, dr_lo, dr_hi = stats[p]["paired_diff_risk"]
            print(f"   vs STAC Paired Diff: Delta NFE = {dn_pt:+5.2f} [{dn_lo:+5.2f}, {dn_hi:+5.2f}] | Delta Risk = {dr_pt*100:+4.2f}% [{dr_lo*100:+4.2f}%, {dr_hi*100:+4.2f}%]")

    print("\n" + "="*80)
    print("ITEM 2: NFE-VERSUS-RISK CURVES (alpha from 0.01 to 0.10)")
    print("="*80)
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    print(f"{'Alpha':<6s} | {'STAC NFE':<9s} {'STAC Risk':<9s} | {'Phi NFE':<9s} {'Phi Risk':<9s} | {'Veto NFE':<9s} {'Veto Risk':<9s} | {'SNR NFE':<9s} {'SNR Risk':<9s}")
    for a in alphas:
        row = f"{a:<6.2f} | "
        for p in policies:
            oof_p = run_5fold_evaluation(p, alpha=a)
            m_nfe = sum(r["nfe"] for r in oof_p) / len(oof_p)
            m_risk = sum(r["loss"] for r in oof_p) / len(oof_p)
            row += f"{m_nfe:5.2f} ({m_risk*100:4.1f}%) | "
        print(row)

    print("\n" + "="*80)
    print("ITEM 3: WITHIN-CONDITION AUC OF Phi_4 FOR STEP-4 FAILURE PREDICTION")
    print("="*80)
    for c in ["clean", "gauss25", "gauss22", "gauss20", "gauss18", "noisy15"]:
        auc_val, lo, hi = auc_within_condition(c)
        if auc_val is None:
            print(f"Condition: {c:<10s} | No discordant pairs (0 or 100% failure rate)")
        else:
            lo_str = f"{lo:.3f}" if lo is not None else "N/A"
            hi_str = f"{hi:.3f}" if hi is not None else "N/A"
            print(f"Condition: {c:<10s} | AUC = {auc_val:.3f} [{lo_str}, {hi_str}]")

