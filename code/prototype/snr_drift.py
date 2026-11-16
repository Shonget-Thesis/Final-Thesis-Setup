"""SNR + drift: the thesis's input-adaptive handover rule (Chapter 3), kept apart from the model and the interface.

The rule decides when DMOSpeech 2's teacher hands over to the student.

1. Gate. The prompt's SNR is estimated from frame energies (Definition 3.3): 25 ms frames with a 10 ms hop, the mean
   energy of the loudest half against the quietest 10%. Below 20 dB the prompt is treated as noisy, and the teacher runs
   to the reference step: Step 8 of the 16-interval sway grid, the thesis's version of the paper's t_switch = 0.25.
2. Drift. Otherwise, after each teacher step k the endpoint estimate x1_hat(t_k) = x_k + (1 - t_k) v_k is compared with
   the previous one, over generated frames only:
       d_k = ||x1_hat(t_k) - x1_hat(t_{k-1})|| / (||x1_hat(t_k)|| + eps).
   The teacher hands over at the first candidate step k in {4, 5, 6} with d_k <= lambda. If none qualifies, it continues
   to Step 8.

The arithmetic is the same as code/scheduling/sampler.py (estimate_prompt_snr, compute_stabilization_drift,
evaluate_handover_policy); code/tests/test_prototype_snr_drift.py checks that they agree. Nothing here loads a model or
draws an interface, and NumPy is the only dependency (PyTorch tensors are accepted where noted).

Thresholds (Appendix D, development evidence):
    lambda = 0.35  chosen by speaker-grouped cross-validation in the development pilot (the drift check is active);
    lambda = 1.20  certified by fixed-sequence Learn-Then-Test on the 200 calibration speakers. At this value the drift
                   check never blocks a handover in practice, so the rule acts as SNR only.
Neither is the final calibrated threshold; that follows the full run after the proposal.
"""
import math

import numpy as np

SNR_GATE_DB = 20.0
CANDIDATE_STEPS = (4, 5, 6)
REFERENCE_STEP = 8
CFG_PASSES_PER_STEP = 2
LAMBDA_PRESETS = {
    "cv": {"value": 0.35, "label": "0.35 (cross-validated, development pilot)"},
    "ltt": {"value": 1.20, "label": "1.20 (LTT-certified, acts as SNR only)"},
}
DEFAULT_LAMBDA = LAMBDA_PRESETS["cv"]["value"]

TEACHER_GRID = [1.0 - math.cos(math.pi / 2.0 * i / 16.0) for i in range(17)]    # t_0 .. t_16, sway s = -1
STUDENT_TAUS = [1.0 - math.cos(math.pi / 2.0 * s) for s in (0.25, 0.50, 0.75)]  # 0.0761, 0.2929, 0.6173


def passes(step):
    """Forward passes when the teacher hands over at `step`: 2 per guided teacher step (steps 0..step), then 3 student
    steps. Step 4 costs 13, Step 8 (the reference) 21."""
    return CFG_PASSES_PER_STEP * (int(step) + 1) + len(STUDENT_TAUS)


REFERENCE_PASSES = passes(REFERENCE_STEP)


def estimate_prompt_snr(waveform, sample_rate=24000, eps=1e-8):
    """Prompt SNR in dB, clipped to [0, 50]; 0.0 if the prompt has fewer than two frames."""
    x = np.asarray(waveform)
    if x.ndim > 1:
        x = x.squeeze()
    frame_length = round(0.025 * sample_rate)
    hop_length = round(0.010 * sample_rate)
    num_frames = 1 + (len(x) - frame_length) // hop_length
    if num_frames < 2:
        return 0.0
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(num_frames, frame_length), strides=(x.strides[0] * hop_length, x.strides[0]))
    energy = np.sort(np.mean(frames ** 2, axis=-1))
    noise = np.mean(energy[:max(1, round(0.10 * len(energy)))])
    signal = np.mean(energy[len(energy) // 2:])
    return float(np.clip(10.0 * np.log10((signal + eps) / (noise + eps)), 0.0, 50.0))


def stabilization_drift(endpoint_curr, endpoint_prev, prompt_len, eps=1e-8):
    """d_k over generated frames [prompt_len:] of (batch, frames, mels) arrays; PyTorch tensors or NumPy arrays."""
    if hasattr(endpoint_curr, "detach"):
        import torch

        diff = (endpoint_curr - endpoint_prev)[:, prompt_len:, :]
        return float((torch.norm(diff) / (torch.norm(endpoint_curr[:, prompt_len:, :]) + eps)).item())
    curr, prev = np.asarray(endpoint_curr, dtype=np.float64), np.asarray(endpoint_prev, dtype=np.float64)
    diff = (curr - prev)[:, prompt_len:, :]
    return float(np.linalg.norm(diff) / (np.linalg.norm(curr[:, prompt_len:, :]) + eps))


def decide(drift_trace, prompt_snr, lam=DEFAULT_LAMBDA, gate_db=SNR_GATE_DB, candidates=CANDIDATE_STEPS,
           reference=REFERENCE_STEP):
    """The handover step from a whole drift trace {k: d_k}: the reference if the gate is closed, else the first
    candidate with d_k <= lam, else the reference."""
    if prompt_snr < gate_db:
        return reference
    for k in candidates:
        d = drift_trace.get(k)
        if d is not None and d <= lam:
            return k
    return reference


class HandoverMonitor:
    """The same rule, applied while the teacher runs. Call observe(step, drift) after each teacher step (drift is None
    at step 0); it returns True once the teacher should stop, and from then on `step` and `reason` hold the decision.
    Because each test needs only d_k up to the current step, stopping early gives the same step as decide() on the full
    trace."""

    def __init__(self, prompt_snr, lam=DEFAULT_LAMBDA, gate_db=SNR_GATE_DB, candidates=CANDIDATE_STEPS,
                 reference=REFERENCE_STEP):
        self.prompt_snr, self.lam, self.gate_db = float(prompt_snr), float(lam), float(gate_db)
        self.candidates, self.reference = tuple(candidates), int(reference)
        self.gate_open = self.prompt_snr >= self.gate_db
        self.drifts = {}
        self.checks = []
        self.step = None
        self.reason = None

    def observe(self, step, drift):
        step = int(step)
        if drift is not None:
            self.drifts[step] = float(drift)
        if self.step is not None:
            return True
        if self.gate_open and step in self.candidates and drift is not None:
            ok = drift <= self.lam
            self.checks.append({"step": step, "drift": float(drift), "passes": bool(ok)})
            if ok:
                self.step = step
                self.reason = f"d_{step} = {drift:.3f} <= lambda = {self.lam:.2f}: the trajectory has stabilized"
                return True
        if step >= self.reference:
            self.step = self.reference
            self.reason = (f"prompt SNR {self.prompt_snr:.1f} dB < {self.gate_db:.0f} dB: gate closed, run to the reference"
                           if not self.gate_open else
                           f"no candidate step had d_k <= lambda = {self.lam:.2f}: run to the reference")
            return True
        return False

    def summary(self):
        step = self.step
        return {"prompt_snr_db": self.prompt_snr, "gate_db": self.gate_db, "gate_open": self.gate_open,
                "lambda": self.lam, "candidates": list(self.candidates), "reference_step": self.reference,
                "drifts": {str(k): v for k, v in sorted(self.drifts.items())}, "checks": self.checks,
                "handover_step": step, "handover_time": None if step is None else TEACHER_GRID[step],
                "reason": self.reason, "passes": None if step is None else passes(step),
                "reference_passes": REFERENCE_PASSES,
                "saving": None if step is None else 1.0 - passes(step) / REFERENCE_PASSES}
