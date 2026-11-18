"""Shared inference-time handover mechanism operators for Path A (EXP-035) and Path B (EXP-036).

All operators accept both 2D `(frames, bands)` arrays (NumPy or PyTorch) for unit-tested math and
3D `(1, plen + n_gen, bands)` trajectories with `draft_prompt_payload()` fixing the prompt frames
to the one-pass student draft `s0`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_consistency_gate_prelim as cg  # noqa: E402
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as gl  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_selective_handover_prelim as sh  # noqa: E402

# Frozen prior calibration anchors from EXP-033 CPU screen (dev positions 1-30).
CLEAN_R_MED = 0.23394784403208624
LAMBDA_R = 0.34004843191300677
FLOOR_Q = 10.0
CLAMP_Q = 1.0
MACRO_WINDOW = 9
EPS = 1e-12


def unseen_cal_slice(start, end):
    """Return calibration items disjoint from all 100 development speakers, sliced [start:end]."""
    dev_spks = {it["speaker_id"] for it in cg.dev_items()}
    cal = sorted(rf.split_items("calibration"), key=lambda it: it["utt_id"])
    unseen = [it for it in cal if it["speaker_id"] not in dev_spks]
    picked = unseen[start:end]
    if len(picked) != (end - start):
        raise RuntimeError(f"Expected {end - start} unseen calibration speakers at [{start}:{end}], got {len(picked)}")
    return picked


def _is_torch(x):
    return hasattr(x, "clone")


def _clone(x):
    return x.clone() if _is_torch(x) else np.array(x, copy=True)


def _to_numpy(x):
    return x.detach().cpu().numpy() if _is_torch(x) else np.asarray(x)


def _from_numpy(arr, ref):
    if _is_torch(ref):
        import torch

        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(dtype=ref.dtype, device=ref.device)
    return np.asarray(arr, dtype=ref.dtype)


def draft_prompt_payload(s0, gen_frames, plen):
    """Build a (1, plen + n_gen, B) payload keeping s0's prompt frames [:plen] and setting [plen:] = gen_frames."""
    out = _clone(s0)
    out[:, plen:, :] = gen_frames if getattr(gen_frames, "ndim", 0) == 3 else gen_frames[None, :, :]
    return out


# --------------------------------------------------------------------------- Path A: Condition-Invariance Ladder
def floor_canonical_gen(student_gen, teacher_gen, floor_q=FLOOR_Q, clamp_q=CLAMP_Q):
    """Scalar power-floor canonicalization (FCH) on `(frames, bands)` natural log-mel values."""
    t_np = np.asarray(_to_numpy(teacher_gen), dtype=np.float64)
    s_np = np.asarray(_to_numpy(student_gen), dtype=np.float64)
    pt = np.exp(np.clip(2.0 * t_np, -40.0, 40.0))
    ps = np.exp(np.clip(2.0 * s_np, -40.0, 40.0))
    shifted = pt - np.percentile(pt, floor_q, axis=0) + np.percentile(ps, floor_q, axis=0)
    out = 0.5 * np.log(np.maximum(shifted, np.percentile(ps, clamp_q, axis=0)))
    return _from_numpy(out, teacher_gen)


def voice_swap_gen(student_gen, teacher_gen):
    """G1 shift-invariant payload: teacher time variation around each band's student-draft mean."""
    if _is_torch(teacher_gen):
        return (teacher_gen - teacher_gen.mean(dim=0, keepdim=True)) + student_gen.mean(dim=0, keepdim=True)
    t = np.asarray(teacher_gen, dtype=np.float64)
    s = np.asarray(student_gen, dtype=np.float64)
    return ((t - t.mean(axis=0, keepdims=True)) + s.mean(axis=0, keepdims=True)).astype(t.dtype)


def affine_gen(student_gen, teacher_gen, eps=1e-8):
    """G2 positive-affine-invariant payload: match per-band mean and standard deviation to student_gen."""
    if _is_torch(teacher_gen):
        import torch

        t = teacher_gen.to(dtype=torch.float64)
        s = student_gen.to(dtype=torch.float64)
        mu_t, mu_s = t.mean(dim=0, keepdim=True), s.mean(dim=0, keepdim=True)
        sd_t = t.std(dim=0, unbiased=False, keepdim=True)
        sd_s = s.std(dim=0, unbiased=False, keepdim=True)
        z = torch.where(sd_t > eps, (t - mu_t) / torch.clamp(sd_t, min=eps), torch.zeros_like(t))
        return (z * sd_s + mu_s).to(dtype=teacher_gen.dtype)
    t = np.asarray(teacher_gen, dtype=np.float64)
    s = np.asarray(student_gen, dtype=np.float64)
    mu_t, mu_s = t.mean(axis=0, keepdims=True), s.mean(axis=0, keepdims=True)
    sd_t, sd_s = t.std(axis=0, ddof=0, keepdims=True), s.std(axis=0, ddof=0, keepdims=True)
    z = np.where(sd_t > eps, (t - mu_t) / np.maximum(sd_t, eps), 0.0)
    return (z * sd_s + mu_s).astype(np.asarray(teacher_gen).dtype)


def cph8_dp_payload(s0, t_k, plen):
    """Base CPH8 control retaining raw teacher generated frames paired with clean student draft prompt frames."""
    return draft_prompt_payload(s0, t_k[0, plen:, :], plen)


def fch_payload(s0, t_k, plen):
    """FCH generated frames paired with draft prompt frames (resolves teacher prompt leakage confound)."""
    return draft_prompt_payload(s0, floor_canonical_gen(s0[0, plen:, :], t_k[0, plen:, :]), plen)


def g1_vswap_payload(s0, t_k, plen):
    """G1 Voice-Swap generated frames paired with draft prompt frames."""
    return draft_prompt_payload(s0, voice_swap_gen(s0[0, plen:, :], t_k[0, plen:, :]), plen)


def g2_affine_payload(s0, t_k, plen):
    """G2 Affine z-score generated frames paired with draft prompt frames."""
    return draft_prompt_payload(s0, affine_gen(s0[0, plen:, :], t_k[0, plen:, :]), plen)


# --------------------------------------------------------------------------- Path B: Dual-Space & Closed-Loop Ops
def _sorted_pair(student_gen, teacher_gen):
    """Return sorted values and stable inverse rank indices along frame axis 0."""
    if _is_torch(teacher_gen):
        import torch

        t_vals, order = torch.sort(teacher_gen, dim=0, stable=True)
        s_vals = torch.sort(student_gen, dim=0, stable=True).values
        ranks = torch.argsort(order, dim=0)
        return s_vals, t_vals, ranks
    t = np.asarray(teacher_gen, dtype=np.float64)
    s = np.asarray(student_gen, dtype=np.float64)
    order = np.argsort(t, axis=0, kind="stable")
    ranks = np.argsort(order, axis=0, kind="stable")
    return np.sort(s, axis=0, kind="stable"), np.sort(t, axis=0, kind="stable"), ranks


def _unit_ranks(x):
    """Normalized rank coordinates in [0, 1] along axis 0 per band."""
    n = x.shape[0]
    denom = float(max(n - 1, 1))
    if _is_torch(x):
        import torch

        order = torch.argsort(x, dim=0, stable=True)
        ranks = torch.argsort(order, dim=0).to(dtype=torch.float64)
        return ranks / denom
    order = np.argsort(np.asarray(x, dtype=np.float64), axis=0, kind="stable")
    ranks = np.argsort(order, axis=0, kind="stable").astype(np.float64)
    return ranks / denom


def _band_roughness(x):
    """Von Neumann ratio per band: mean squared frame delta / band variance (NumPy or PyTorch)."""
    if _is_torch(x):
        import torch

        xf = x.to(dtype=torch.float64)
        if xf.shape[0] < 2:
            return torch.zeros(xf.shape[1], dtype=torch.float64, device=x.device)
        d = xf[1:] - xf[:-1]
        return (d * d).mean(dim=0) / torch.clamp(xf.var(dim=0, unbiased=False), min=EPS)
    xf = np.asarray(x, dtype=np.float64)
    if xf.shape[0] < 2:
        return np.zeros(xf.shape[1], dtype=np.float64)
    d = np.diff(xf, axis=0)
    return np.mean(d * d, axis=0) / np.maximum(np.var(xf, axis=0), EPS)


def geodesic_shrink_weights(student_gen, teacher_gen, r_clean=CLEAN_R_MED, r_hi=LAMBDA_R, global_mix=0.5):
    """Continuous per-band and global W2 transport weights alpha_b in [0, 1] without a binary gate."""
    s_vals, t_vals, ranks = _sorted_pair(student_gen, teacher_gen)
    span = max(float(r_hi) - float(r_clean), 1e-6)
    if _is_torch(teacher_gen):
        import torch

        diff = (t_vals - s_vals).to(dtype=torch.float64)
        t64 = teacher_gen.to(dtype=torch.float64)
        r_band = torch.linalg.norm(diff, dim=0) / (torch.linalg.norm(t64, dim=0) + EPS)
        r_glob = torch.linalg.norm(diff) / (torch.linalg.norm(t64) + EPS)
        a_band = torch.clamp((r_band - r_clean) / span, 0.0, 1.0)
        a_glob = torch.clamp((r_glob - r_clean) / span, 0.0, 1.0)
        alpha = torch.maximum(a_band, float(global_mix) * a_glob)
        return s_vals, t_vals, ranks, alpha, r_band, r_glob
    diff = np.asarray(t_vals, dtype=np.float64) - np.asarray(s_vals, dtype=np.float64)
    t64 = np.asarray(teacher_gen, dtype=np.float64)
    r_band = np.linalg.norm(diff, axis=0) / (np.linalg.norm(t64, axis=0) + EPS)
    r_glob = float(np.linalg.norm(diff) / (np.linalg.norm(t64) + EPS))
    a_band = np.clip((r_band - r_clean) / span, 0.0, 1.0)
    a_glob = float(np.clip((r_glob - r_clean) / span, 0.0, 1.0))
    alpha = np.maximum(a_band, float(global_mix) * a_glob)
    return s_vals, t_vals, ranks, alpha, r_band, r_glob


def multiscale_rank_blend(teacher_gen, student_gen, alpha, window=MACRO_WINDOW, macro_weight=0.0):
    """Damp high-frequency noise jitter in rank order above student roughness while keeping macro contour."""
    n = teacher_gen.shape[0]
    w = min(int(window), n if n % 2 == 1 else max(1, n - 1))
    w = max(1, w)
    u_t, u_s = _unit_ranks(teacher_gen), _unit_ranks(student_gen)
    macro_t = sh.smooth_time(u_t, w)
    macro_s = sh.smooth_time(u_s, w)
    micro_t, micro_s = u_t - macro_t, u_s - macro_s
    rough_t, rough_s = _band_roughness(teacher_gen), _band_roughness(student_gen)
    if _is_torch(teacher_gen):
        import torch

        eta = torch.clamp((rough_t - rough_s) / (rough_t + EPS), 0.0, 1.0)
        m_mix = (float(macro_weight) * alpha * eta).unsqueeze(0)
        f_mix = eta.unsqueeze(0)
        u_blend = ((1.0 - m_mix) * macro_t + m_mix * macro_s) + ((1.0 - f_mix) * micro_t + f_mix * micro_s)
        # Preserve exact ties with original teacher when stabilization weight vanishes everywhere.
        order = torch.argsort(u_blend, dim=0, stable=True)
        return torch.argsort(order, dim=0), eta
    eta = np.clip((rough_t - rough_s) / (rough_t + EPS), 0.0, 1.0)
    m_mix = float(macro_weight) * alpha[None, :] * eta[None, :]
    f_mix = eta[None, :]
    u_blend = ((1.0 - m_mix) * macro_t + m_mix * macro_s) + ((1.0 - f_mix) * micro_t + f_mix * micro_s)
    order = np.argsort(u_blend, axis=0, kind="stable")
    return np.argsort(order, axis=0, kind="stable"), eta


def dual_space_geodesic_gen(student_gen, teacher_gen, macro_weight=0.0, window=MACRO_WINDOW,
                            r_clean=CLEAN_R_MED, r_hi=LAMBDA_R, global_mix=0.5):
    """Continuous 1D W2 soft-shrinkage in values + excess-roughness rank stabilization in time order."""
    s_vals, t_vals, orig_ranks, alpha, _, _ = geodesic_shrink_weights(
        student_gen, teacher_gen, r_clean=r_clean, r_hi=r_hi, global_mix=global_mix,
    )
    blended_ranks, eta = multiscale_rank_blend(teacher_gen, student_gen, alpha, window=window, macro_weight=macro_weight)
    if _is_torch(teacher_gen):
        import torch

        a = alpha.to(dtype=teacher_gen.dtype).unsqueeze(0)
        q_geo = (1.0 - a) * t_vals.to(dtype=teacher_gen.dtype) + a * s_vals.to(dtype=teacher_gen.dtype)
        # Exact identity preservation on clean smooth bands where eta == 0 and alpha == 0.
        exact_t = (alpha == 0) & (eta == 0)
        use_ranks = torch.where(exact_t.unsqueeze(0), orig_ranks, blended_ranks)
        return torch.gather(q_geo, 0, use_ranks)
    a = alpha[None, :]
    q_geo = (1.0 - a) * np.asarray(t_vals, dtype=np.float64) + a * np.asarray(s_vals, dtype=np.float64)
    exact_t = (alpha == 0) & (eta == 0)
    use_ranks = np.where(exact_t[None, :], orig_ranks, blended_ranks)
    return np.take_along_axis(q_geo, use_ranks, axis=0).astype(np.asarray(teacher_gen).dtype)


def dsgh_payload(s0, t_k, plen, macro_weight=0.0, **kw):
    """Wrap dual_space_geodesic_gen onto draft prompt frames for handover at t = 0.2929."""
    return draft_prompt_payload(
        s0, dual_space_geodesic_gen(s0[0, plen:, :], t_k[0, plen:, :], macro_weight=macro_weight, **kw), plen
    )


def proximal_endpoint_gen(student_gen, ep_gen, step_scale=1.0, r_clean=CLEAN_R_MED, r_hi=LAMBDA_R, global_mix=0.5):
    """Closed-loop 1D Wasserstein proximal projection applied inside intermediate Euler teacher steps."""
    s_vals, t_vals, orig_ranks, alpha, _, _ = geodesic_shrink_weights(
        student_gen, ep_gen, r_clean=r_clean, r_hi=r_hi, global_mix=global_mix,
    )
    eff = float(step_scale) * alpha
    if _is_torch(ep_gen):
        import torch

        a = eff.to(dtype=ep_gen.dtype).unsqueeze(0)
        q_prox = (1.0 - a) * t_vals.to(dtype=ep_gen.dtype) + a * s_vals.to(dtype=ep_gen.dtype)
        return torch.gather(q_prox, 0, orig_ranks)
    a = eff[None, :]
    q_prox = (1.0 - a) * np.asarray(t_vals, dtype=np.float64) + a * np.asarray(s_vals, dtype=np.float64)
    return np.take_along_axis(q_prox, orig_ranks, axis=0).astype(np.asarray(ep_gen).dtype)


def cwps_teacher(pipe, ctx, s0, grid, k_end=4, macro_weight=0.5):
    """Closed-Loop Wasserstein-Proximal Teacher Sampling (CWPS, steps 0..k_end = 10 teacher passes).

    Applies proximal quantile contraction toward student draft s0 starting from i >= 1 (keeping t_0's
    initial symmetry untouched) before velocity update v_tilde = (ep_tilde - x) / (1 - t_i), then returns
    the dual-space readout payload with draft prompt frames.
    """
    torch, sampler = pipe.torch, pipe.sampler
    plen = ctx["prompt_len"]
    s_gen = s0[0, plen:, :]
    x = ctx["y0"].clone()
    t_commit = float(grid[k_end])
    ep = None
    with torch.inference_mode():
        for i in range(k_end + 1):
            t_val = float(grid[i])
            t = torch.tensor(t_val, device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ep = x + (1.0 - t_val) * v
            if 1 <= i < k_end:
                scale = min(1.0, float(grid[i + 1]) / max(t_commit, 1e-6))
                ep_gen = proximal_endpoint_gen(s_gen, ep[0, plen:, :], step_scale=scale)
                v_gen = (ep_gen - x[0, plen:, :]) / max(1.0 - t_val, 1e-6)
                v = v.clone()
                v[0, plen:, :] = v_gen
            if i < k_end:
                x = x + (float(grid[i + 1]) - t_val) * v
    return dsgh_payload(s0, ep, plen, macro_weight=macro_weight)


# --------------------------------------------------------------------------- Shared Prereg, Manifest, Timing & Stats
def write_prereg_doc(prereg_path, payload, force=False):
    """Hash and persist a preregistration JSON document cleanly."""
    import json
    import run_transmissibility_bridge as tb

    prereg_path = Path(prereg_path)
    if prereg_path.exists() and not force:
        raise RuntimeError(f"Preregistration already exists at {prereg_path}; pass --force to overwrite.")
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    sha = tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    doc = {**payload, "sha256": sha, "created_utc": tb._utc()}
    with open(prereg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=2) + "\n")
    return doc


def verify_prereg_doc(prereg_path, code_dir, thesis_root, item_checks):
    """Verify preregistration integrity, code SHA-256 hashes, input SHA-256 hashes, and exact speaker lists."""
    import json
    import run_transmissibility_bridge as tb

    prereg_path = Path(prereg_path)
    if not prereg_path.exists():
        raise RuntimeError(f"Missing preregistration: {prereg_path}. Run `freeze` first.")
    doc = json.loads(prereg_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in doc.items() if k not in ("sha256", "created_utc")}
    if doc.get("sha256") != tb._sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8")):
        raise RuntimeError("Preregistration SHA-256 mismatch.")
    for name, rec in doc.get("code_hashes", {}).items():
        actual = tb._sha256_file(Path(code_dir) / name)
        if actual != rec:
            raise RuntimeError(f"Code hash mismatch for {name}: expected {rec[:12]}, got {actual[:12]}.")
    for rel, rec in doc.get("input_hashes", {}).items():
        actual = tb._sha256_file(Path(thesis_root) / rel)
        if actual != rec:
            raise RuntimeError(f"Input hash mismatch for {rel}: expected {rec[:12]}, got {actual[:12]}.")
    for key, items in item_checks:
        if doc.get(key) != [{"utt_id": it["utt_id"], "speaker_id": it["speaker_id"]} for it in items]:
            raise RuntimeError(f"Speaker list {key} does not match preregistration.")
    return doc


def write_results_manifest(out_dir, exp_id, prereg_path, file_paths):
    """Write SHA-256 MANIFEST.json covering preregistration reference and result files."""
    import json
    import run_transmissibility_bridge as tb

    entries = {}
    if prereg_path is not None and Path(prereg_path).exists():
        entries["preregistration"] = {"path": str(prereg_path), "sha256": tb._sha256_file(prereg_path)}
    for p in file_paths:
        p = Path(p)
        if p.exists():
            entries[p.name] = {"sha256": tb._sha256_file(p), "bytes": p.stat().st_size}
    out_dir = Path(out_dir)
    with open(out_dir / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"experiment": exp_id, "utc": tb._utc(), "files": entries}, indent=2) + "\n")


def run_timing_arms(pipe, items, out_dir, time_arms, standalone_fn, time_fields, cond="clean", csv_name="timing.csv"):
    """Synthesize each arm independently between GPU sync barriers with forward-pass counting."""
    import run_cheaper_teacher_path as ctp
    import run_noise_guidance_prelim as ng
    import run_transmissibility_bridge as tb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    todo = [it for it in items if (it["utt_id"], cond) not in done]
    print(f"[timing] {len(todo)} cases to run ({len(done)} already in {path.name})", flush=True)

    def one(ctx, arm):
        return ng.decode(pipe, ctx, standalone_fn(pipe, ctx, arm))

    with ctp.PassCounter(pipe) as counter:
        for n_run, item in enumerate(todo, 1):
            ctx = pipe.prepare(item, cond, int(item["input_seed"]), out_dir)
            if n_run == 1:
                one(ctx, time_arms[0])
            row = {
                "utt_id": item["utt_id"],
                "speaker_id": item["speaker_id"],
                "condition": cond,
                "prompt_snr": float(ctx["prompt_snr"]),
                "gpu": pipe.gpu,
                "frames": int(ctx["y0"].shape[1] - ctx["prompt_len"]),
                "seq_len": int(ctx["y0"].shape[1]),
            }
            for arm in (time_arms if n_run % 2 else time_arms[::-1]):
                counter.reset()
                _, sec = gl.timed(pipe, one, ctx, arm)
                row.update({
                    f"seconds_{arm}": round(sec, 4),
                    f"teacher_passes_{arm}": counter.n["teacher"],
                    f"student_passes_{arm}": counter.n["student"],
                })
            row["utc"] = tb._utc()
            tb._append(path, time_fields, row)
            print(f"[{n_run}/{len(todo)}] timing: {item['utt_id']}", flush=True)
    print(f"[OK] timing complete: {path}", flush=True)
    return path


def paired_speaker_contrast(rows, val_fn, a, b, metrics=("wer", "similarity", "utmos"),
                            boot_b=2000, seed=2026, wer_bar=0.01, sim_bar=-0.01):
    """Speaker-level paired difference (a - b) with bootstrap 95% intervals and standard pass flags."""
    import run_cheaper_teacher_path as ctp
    import run_transmissibility_confirm as tc

    out = {}
    for m in metrics:
        per = {}
        for r in rows:
            per.setdefault(r["utt_id"], []).append(val_fn(r, a, m) - val_fn(r, b, m))
        out[m] = tc.boot_ci([ctp._mean(v) for v in per.values()], B=boot_b, seed=seed)
    out["better"] = bool(
        out["utmos"]["mean"] > 0
        and out["utmos"]["lo"] > 0
        and out["wer"]["hi"] <= wer_bar
        and out["similarity"]["lo"] >= sim_bar
    )
    out["non_inferior_utmos"] = bool(out["utmos"]["lo"] >= -0.05 and out["wer"]["hi"] <= wer_bar and out["similarity"]["lo"] >= sim_bar)
    return out
