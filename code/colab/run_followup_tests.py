"""Follow-up Colab T4 tests for the adaptive handover study (audit of 2026-10-25).

Subcommands, run in this order (DMOSpeech2_Followup_Tests.ipynb drives them):

  freeze        Build the fresh-speaker manifest and write preregistration.json. Nothing is
                synthesized. Every later subcommand checks the preregistration hash first.
  smoke         Run every synthesis path on 2 development speakers into OUT/smoke/.
                Never touches fresh speakers, so it can be run before or after freezing.
  pilot         Optional development pilot (default 20 development speakers) into OUT/pilot/:
                checks that the pipeline reproduces the T4 grid, whether the SNR gate lets
                degraded prompts through, a small seed-floor sample, and the time per case.
                Writes PILOT_REPORT.md. Never touches fresh or test speakers; needs no freeze.
  seed-floor    Test split (200 speakers, clean + noisy15): Step 8 with the manifest seed
                (seed A) and with a second seed (seed B); seed B is scored as a candidate
                against seed A with the frozen failure rule.
  fresh-grid    The 166 prompt speakers in no split, clean + noisy15, Steps 4/5/6/8. Note: these
                speakers were already used in the team's Phase 12 run (EXP-006), so they are not unseen.
  degradations  The same fresh speakers under Gaussian 18-25 dB, DEMAND cafeteria babble,
                synthetic reverberation and telephone-band prompts, Steps 4/5/6/8.
  analyze       Compute every reported number from the CSVs written above and from the
                committed T4 grid. No GPU. Missing inputs raise; nothing is filled in.

Synthesis and scoring reuse core/sampler.py and core/metrics.py from a Shonget-DMOSpeech2
clone pinned to the commit that produced research/evidence/benchmarks/gpu_grid_all1000.csv,
and repeat the per-case steps of colab/run_gpu_expanded_benchmark.py.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = THESIS_ROOT / "research/protocols/splits"
T4_GRID = THESIS_ROOT / "research/evidence/benchmarks/gpu_grid_all1000.csv"

STEPS = (4, 5, 6, 8)
REFERENCE_STEP = 8
COST = {4: 13, 5: 15, 6: 17, 8: 21}
SEED_B_BASE = 1337
LAMBDA_GRID = [round(0.20 + 0.05 * i, 2) for i in range(21)]
ALPHA, DELTA, THETA = 0.05, 0.10, 20.0
DEMAND_FILE = "PCAFETER/ch01.wav"
DEMAND_URL = "https://zenodo.org/records/1227121/files/PCAFETER_16k.zip"

# name -> (kind, parameter); noisy15 is the same white-noise condition as the T4 grid.
CONDITIONS = {
    "clean": ("clean", None),
    "noisy15": ("gauss", 15.0),
    "gauss18": ("gauss", 18.0),
    "gauss20": ("gauss", 20.0),
    "gauss22": ("gauss", 22.0),
    "gauss25": ("gauss", 25.0),
    "babble15": ("babble", 15.0),
    "babble20": ("babble", 20.0),
    "reverb03": ("reverb", 0.3),
    "reverb09": ("reverb", 0.9),
    "telephone": ("telephone", None),
}
GRID_CONDITIONS = ("clean", "noisy15")
DEGRADATION_CONDITIONS = tuple(c for c in CONDITIONS if c not in GRID_CONDITIONS)
# Pilot: development speakers only, every condition of the full run. clean + noisy15 check
# reproduction of the T4 grid; the rest are the degradations planned for Chapter 3.
PILOT_CONDITIONS = tuple(CONDITIONS)
ROW_FIELDS = ["test", "utt_id", "speaker_id", "condition", "base_seed", "input_seed", "teacher_step", "nfe",
              "prompt_snr", "d_4", "d_5", "d_6", "wer", "similarity", "utmos", "wer_ref", "similarity_ref",
              "utmos_ref", "is_failure", "fail_wer", "fail_sim", "hypothesis", "audio_duration", "gt_sr", "utc"]


# ----------------------------------------------------------------------------- data
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_meta_lst(path):
    """Seed-TTS meta.lst: utt|prompt_text|prompt_wav[|gt_text[|gt_wav]] (as data/make_splits.py)."""
    base = Path(path).parent
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4:
            utt, prompt_text, prompt_rel, gt_text = parts
            gt_rel = f"wavs/{utt}.wav"
        elif len(parts) >= 5:
            utt, prompt_text, prompt_rel, gt_text, gt_rel = parts[:5]
        else:
            continue
        entries.append({"utt_id": utt, "speaker_id": Path(prompt_rel).stem, "prompt_text": prompt_text,
                        "prompt_wav_rel": prompt_rel, "gt_text": gt_text, "gt_wav_rel": gt_rel})
    return entries


def used_speakers():
    used = set()
    for name in ("development", "calibration", "test"):
        used |= {r["speaker_id"] for r in json.loads((SPLITS_DIR / f"{name}_manifest.json").read_text(encoding="utf-8"))}
    return used


def select_fresh(entries, used):
    """One utterance per speaker outside every split; the utterance with the smallest SHA-256 of its ID."""
    by_spk = {}
    for e in entries:
        if e["speaker_id"] in used:
            continue
        key = hashlib.sha256(e["utt_id"].encode("utf-8")).hexdigest()
        if e["speaker_id"] not in by_spk or key < by_spk[e["speaker_id"]][0]:
            by_spk[e["speaker_id"]] = (key, e)
    return sorted((e for _, e in by_spk.values()), key=lambda e: e["utt_id"])


def derive_input_seed(base_seed, utt_id):
    """Same as core.sampler.derive_input_seed (repeated here so freeze and tests need no torch)."""
    return (int(base_seed) + int(hashlib.sha256(utt_id.encode("utf-8")).hexdigest()[:8], 16)) % (2**31 - 1)


def split_items(name):
    rows = json.loads((SPLITS_DIR / f"{name}_manifest.json").read_text(encoding="utf-8"))
    items, seen = [], set()
    for r in rows:
        if r["utt_id"] in seen:
            continue
        seen.add(r["utt_id"])
        rel = lambda p: p.replace("\\", "/").split("seed-tts-eval/en/")[-1]
        items.append({"utt_id": r["utt_id"], "speaker_id": r["speaker_id"], "prompt_text": r["prompt_text"],
                      "prompt_wav_rel": rel(r["prompt_wav_path"]), "gt_text": r["gt_text"],
                      "gt_wav_rel": rel(r["gt_wav_path"]), "input_seed": int(r["input_seed"])})
    return items


# ----------------------------------------------------------------------- degradations
def _rms(x):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def add_gaussian(clean, snr_db, seed, eps=1e-12):
    """Identical arithmetic to data/make_splits.py add_gaussian_noise_15db (checked at run time)."""
    import numpy as np
    rng = np.random.RandomState(seed % (2**31 - 1))
    sig = float(np.mean(clean**2))
    if sig < eps:
        return clean.copy()
    raw = rng.randn(*clean.shape).astype(np.float32)
    noise = raw * np.sqrt((sig / (10.0 ** (snr_db / 10.0))) / (float(np.mean(raw**2)) + eps))
    return np.clip(clean + noise, -1.0, 1.0)


def add_babble(clean, track, snr_db, seed, eps=1e-12):
    import numpy as np
    rng = np.random.RandomState(seed % (2**31 - 1))
    n = len(clean)
    if len(track) > n:
        start = rng.randint(0, len(track) - n)
        seg = track[start:start + n].astype(np.float64)
    else:
        seg = np.tile(track, int(np.ceil(n / len(track))))[:n].astype(np.float64)
    sig = float(np.mean(clean.astype(np.float64) ** 2))
    if sig < eps:
        return clean.copy()
    seg *= np.sqrt((sig / (10.0 ** (snr_db / 10.0))) / (float(np.mean(seg**2)) + eps))
    return np.clip(clean + seg, -1.0, 1.0).astype(np.float32)


def add_reverb(clean, sr, rt60, seed):
    """Synthetic room: exponentially decaying Gaussian impulse response (60 dB decay at rt60 s)."""
    import numpy as np
    from scipy.signal import fftconvolve
    rng = np.random.RandomState(seed % (2**31 - 1))
    t = np.arange(int(rt60 * sr)) / sr
    h = rng.randn(len(t)) * np.exp(-6.908 * t / rt60)
    h[0] = 1.0
    h /= np.sqrt(np.sum(h**2))
    wet = fftconvolve(clean.astype(np.float64), h)[: len(clean)]
    wet *= _rms(clean) / max(_rms(wet), 1e-12)
    return np.clip(wet, -1.0, 1.0).astype(np.float32)


def telephone_band(clean, sr):
    """Resample to 8 kHz, band-pass 300-3400 Hz, resample back; loudness matched."""
    import numpy as np
    from math import gcd
    from scipy.signal import butter, resample_poly, sosfiltfilt
    g = gcd(8000, sr)
    low = resample_poly(clean.astype(np.float64), 8000 // g, sr // g)
    low = sosfiltfilt(butter(4, [300, 3400], btype="band", fs=8000, output="sos"), low)
    back = resample_poly(low, sr // g, 8000 // g)[: len(clean)]
    back = np.pad(back, (0, len(clean) - len(back)))
    back *= _rms(clean) / max(_rms(back), 1e-12)
    return np.clip(back, -1.0, 1.0).astype(np.float32)


def degrade(clean, sr, condition, seed, babble_track=None):
    kind, param = CONDITIONS[condition]
    if kind == "clean":
        return clean
    if kind == "gauss":
        return add_gaussian(clean, param, seed)
    if kind == "babble":
        if babble_track is None:
            raise RuntimeError("DEMAND babble track not loaded")
        return add_babble(clean, babble_track, param, seed)
    if kind == "reverb":
        return add_reverb(clean, sr, param, seed)
    return telephone_band(clean, sr)


# --------------------------------------------------------------------------- policy
def choose_step(row8, lam, theta):
    """Definition 3.4 over candidates 4, 5, 6 with the reference at Step 8."""
    if float(row8["prompt_snr"]) < theta:
        return REFERENCE_STEP
    for k in (4, 5, 6):
        if float(row8[f"d_{k}"]) <= lam:
            return k
    return REFERENCE_STEP


POLICIES = {
    "reference": lambda r, lam: REFERENCE_STEP,
    "snr_only": lambda r, lam: choose_step(r, math.inf, THETA),
    "snr_drift_ltt": lambda r, lam: choose_step(r, lam, THETA),
    "snr_drift_050": lambda r, lam: choose_step(r, 0.50, THETA),
    "snr_step5": lambda r, lam: 5 if float(r["prompt_snr"]) >= THETA else REFERENCE_STEP,
    # SNR-only with the gate itself as the calibrated parameter (lam carries theta).
    "snr_theta": lambda r, lam: 4 if float(r["prompt_snr"]) >= lam else REFERENCE_STEP,
}
# Gate grid for calibrating the SNR-only rule, conservative (high theta) to aggressive.
THETA_GRID = [round(30.0 - 0.5 * i, 1) for i in range(33)]   # 30.0, 29.5, ..., 14.0


def is_true(v):
    return str(v).strip().lower() in ("true", "1")


def index_cases(rows):
    """(utt_id, condition) -> {step: row}; only complete cases (all four steps)."""
    cases = {}
    for r in rows:
        cases.setdefault((r["utt_id"], r["condition"]), {})[int(r["teacher_step"])] = r
    return {k: v for k, v in cases.items() if set(v) == set(STEPS)}


def evaluate(cases, utts, conditions, policy, lam):
    """Per speaker: fails if any listed condition fails. Returns (n, failures, per-speaker list, mean NFE)."""
    fails, nfe = [], []
    for u in utts:
        f = False
        for c in conditions:
            steps = cases[(u, c)]
            k = POLICIES[policy](steps[REFERENCE_STEP], lam)
            f = f or is_true(steps[k]["is_failure"])
            nfe.append(COST[k])
        fails.append(int(f))
    return len(utts), sum(fails), fails, sum(nfe) / len(nfe)


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def ltt(cases, utts, conditions, grid=LAMBDA_GRID, alpha=ALPHA, delta=DELTA, policy="snr_drift_ltt"):
    """Fixed-sequence Learn-Then-Test (Algorithm 3.4) over an ordered grid. Returns (parameter or None, table)."""
    lam_hat, table = None, []
    for lam in grid:
        n, k, _, nfe = evaluate(cases, utts, conditions, policy, lam)
        p = binom_cdf(k, n, alpha)
        table.append({"lambda": lam, "failures": k, "n": n, "p": p, "mean_nfe": nfe})
        if p > delta:
            break
        lam_hat = lam
    return lam_hat, table


def clopper_pearson(k, n, conf=0.95):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired 0/1 lists."""
    from scipy.stats import binomtest
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    p = 1.0 if only_a + only_b == 0 else float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return only_a, only_b, p


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------- preregistration
def git_head(path):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prereg_payload(prereg):
    return json.dumps({k: v for k, v in prereg.items() if k not in ("created_utc", "sha256")}, sort_keys=True)


def load_prereg(out):
    path = Path(out) / "preregistration.json"
    if not path.exists():
        raise SystemExit("preregistration.json not found: run `freeze` first.")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    if hashlib.sha256(prereg_payload(prereg).encode("utf-8")).hexdigest() != prereg["sha256"]:
        raise SystemExit("preregistration.json was edited after freezing; refusing to run.")
    for name, digest in prereg["input_hashes"].items():
        if sha256_file(THESIS_ROOT / name) != digest:
            raise SystemExit(f"{name} changed since freezing; refusing to run.")
    return prereg


def cmd_freeze(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "preregistration.json").exists():
        raise SystemExit(f"{out / 'preregistration.json'} already exists; a protocol is frozen once.")
    meta = Path(args.dmo_root) / "data/seed-tts-eval/en/meta.lst"
    entries = parse_meta_lst(meta)
    fresh = select_fresh(entries, used_speakers())
    for e in fresh:
        e["input_seed"] = derive_input_seed(42, e["utt_id"])
    (out / "fresh_manifest.json").write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    inputs = ["research/protocols/splits/development_manifest.json", "research/protocols/splits/calibration_manifest.json",
              "research/protocols/splits/test_manifest.json", "research/evidence/benchmarks/gpu_grid_all1000.csv",
              "code/colab/run_followup_tests.py"]
    prereg = {
        "purpose": "Follow-up T4 tests recommended by research/audits/2026-10-25/FINDINGS.md",
        "thesis_commit": git_head(THESIS_ROOT),
        "dmo_commit": git_head(args.dmo_root),
        "input_hashes": {p: sha256_file(THESIS_ROOT / p) for p in inputs},
        "meta_lst_sha256": sha256_file(meta),
        "fresh_manifest_sha256": sha256_file(out / "fresh_manifest.json"),
        "n_source_speakers": len({e["speaker_id"] for e in entries}),
        "n_fresh_speakers": len(fresh),
        "failure_rule": "fail = WER(k) > max(0.125, WER(8) + 0.01) or SIM(k) < SIM(8) - 0.02 (core/failure_config.json)",
        "steps": list(STEPS), "theta_db": THETA, "alpha": ALPHA, "delta": DELTA, "lambda_grid": LAMBDA_GRID,
        "seed_b_base": SEED_B_BASE, "conditions": {k: list(v) for k, v in CONDITIONS.items()},
        "demand": {"file": DEMAND_FILE, "source": DEMAND_URL},
        "policies": {
            "reference": "Step 8 for every input",
            "snr_only": "Step 4 if prompt SNR >= 20 dB, else Step 8",
            "snr_drift_ltt": "Definition 3.4 with lambda_hat from fixed-sequence LTT on the T4 calibration split (clean + noisy15, speaker fails if either fails)",
            "snr_drift_050": "Definition 3.4 with lambda = 0.50 (the reports' CPU-label threshold)",
            "snr_step5": "Step 5 if prompt SNR >= 20 dB, else Step 8",
        },
        "hypotheses": {
            "H1_heldout": "On fresh speakers (clean + noisy15, speaker-level), snr_drift_ltt and snr_only each have realized risk with a 95% Clopper-Pearson interval reported against alpha = 0.05.",
            "H2_drift_value": "Per degradation condition, exact two-sided McNemar of snr_only vs snr_drift_050 failures on fresh speakers, Holm-adjusted across the nine conditions; drift adds value only if it has fewer failures with adjusted p < 0.05.",
            "H3_seed_floor": "Speaker-level rate at which Step 8 with seed B fails against Step 8 with seed A (test split, clean or noisy15), with a 95% Clopper-Pearson interval.",
        },
        "secondary": ["seed-A Step-8 reproducibility against the committed T4 grid (max abs WER and SIM difference)",
                      "development-evidence analyses on the committed T4 grid: calibrate on clean, test on noisy15; calibration size n = 50, 100, 200 (200 subsamples each, seed 2026)"],
    }
    prereg["sha256"] = hashlib.sha256(prereg_payload(prereg).encode("utf-8")).hexdigest()
    prereg["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen: {len(fresh)} fresh speakers of {prereg['n_source_speakers']}; preregistration sha256 {prereg['sha256']}")
    print("Commit preregistration.json and fresh_manifest.json to the thesis repository before running the tests.")


# --------------------------------------------------------------------- synthesis
class Pipeline:
    """Loads DMOSpeech 2, the metric suite and helpers from the pinned Shonget-DMOSpeech2 clone."""

    def __init__(self, dmo_root, device):
        root = Path(dmo_root).resolve()
        for p in (root, root / "src", root / "Handover Scheduling"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from torch.nn.utils.rnn import pad_sequence
        from infer import DMOInference
        from f5_tts.infer.utils_infer import convert_char_to_pinyin, preprocess_ref_audio_text
        from core import sampler
        from core.metrics import MetricSuite, check_failure, load_failure_config
        from data.make_splits import add_gaussian_noise_15db
        self.np, self.sf, self.torch, self.torchaudio, self.pad_sequence = np, sf, torch, torchaudio, pad_sequence
        self.sampler, self.check_failure, self.cfg = sampler, check_failure, load_failure_config()
        self.convert_char_to_pinyin, self.preprocess_ref_audio_text = convert_char_to_pinyin, preprocess_ref_audio_text
        self.data_dir = root / "data/seed-tts-eval/en"
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise SystemExit("These tests are defined for the Colab GPU; no CUDA device found.")
        self.gpu = torch.cuda.get_device_name(0)
        torch.backends.cudnn.benchmark = True
        self.tts = DMOInference(student_checkpoint_path=str(root / "ckpts/model_85000.pt"),
                                duration_predictor_path=str(root / "ckpts/model_1500.pt"), device=self.device)
        self.metrics = MetricSuite(device=device)
        # The white-noise path must match the T4 grid's noisy15 exactly.
        probe = np.random.RandomState(0).randn(24000).astype(np.float32) * 0.1
        if not np.array_equal(add_gaussian(probe, 15.0, 7), add_gaussian_noise_15db(probe, seed=7, target_snr_db=15.0)[0]):
            raise SystemExit("add_gaussian differs from data/make_splits.add_gaussian_noise_15db")
        self.babble_raw = None

    def load_babble(self, demand_dir):
        path = Path(demand_dir) / DEMAND_FILE
        if not path.exists():
            raise SystemExit(f"DEMAND file missing: {path} (download {DEMAND_URL})")
        track, sr = self.sf.read(str(path), dtype="float32")
        self.babble_sr, self.babble_raw, self.babble_sha = sr, track, sha256_file(path)

    def babble_for(self, sr):
        from math import gcd
        from scipy.signal import resample_poly
        if self.babble_raw is None:
            raise SystemExit("Babble condition requested but no DEMAND track loaded (--demand_dir).")
        g = gcd(sr, self.babble_sr)
        return resample_poly(self.babble_raw, sr // g, self.babble_sr // g).astype(self.np.float32)

    def synthesize(self, item, condition, seed, steps, out_dir, noise_seed=None):
        """Steps of run_gpu_expanded_benchmark.py for one (utterance, condition, seed).

        The prompt degradation uses noise_seed (default: the synthesis seed, as the T4 grid did).
        """
        ctx = self.prepare(item, condition, seed, out_dir, noise_seed)
        endpoints, drift = self.teacher(ctx, steps)
        return {k: self.student(ctx, endpoints[k]) for k in steps}, ctx["prompt_snr"], drift

    def teacher(self, ctx, steps):
        endpoints, drift, _ = self.sampler.run_teacher_single_pass(
            tts=self.tts, step_cond=ctx["step_cond"], text=ctx["text_ids"], y0=ctx["y0"], prompt_len=ctx["prompt_len"],
            target_steps=tuple(steps), cfg_strength=self.sampler.CFG_STRENGTH)
        return endpoints, drift

    def student(self, ctx, endpoint):
        return self.sampler.run_student_completion(tts=self.tts, cond=ctx["cond"], text=ctx["text_ids"], teacher_endpoint=endpoint,
                                                   prompt_len=ctx["prompt_len"], input_seed=ctx["seed"])[1]

    def prepare(self, item, condition, seed, out_dir, noise_seed=None, prompt_transform=None):
        """Prompt degradation, duration prediction, conditioning and initial noise for one case.

        prompt_transform(prompt, sr), if given (for example a speech enhancer), is applied after the prompt SNR is
        measured, so ctx["prompt_snr"] stays the SNR of the degraded prompt the scheduler would see."""
        torch, sampler = self.torch, self.sampler
        prompt_path = self.data_dir / item["prompt_wav_rel"]
        if not prompt_path.exists() or prompt_path.stat().st_size < 1000:
            raise SystemExit(f"Prompt audio missing or an LFS pointer: {prompt_path}")
        raw, sr = self.torchaudio.load(str(prompt_path))
        prompt = raw.squeeze().numpy()
        track = self.babble_for(sr) if CONDITIONS[condition][0] == "babble" else None
        prompt = degrade(prompt, sr, condition, seed if noise_seed is None else noise_seed, track)
        prompt_snr = sampler.estimate_prompt_snr(prompt, sample_rate=sr)
        if prompt_transform is not None:
            prompt = prompt_transform(prompt, sr)
        tmp = Path(out_dir) / f"tmp_{item['utt_id']}_{condition}.wav"
        self.sf.write(str(tmp), prompt, sr)
        ref_text = self.preprocess_ref_audio_text(str(tmp), item["prompt_text"])[1]
        audio = torch.from_numpy(prompt).float().unsqueeze(0)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < 0.3:
            audio = audio * 0.3 / rms
        if sr != self.tts.target_sample_rate:
            audio = self.torchaudio.transforms.Resample(sr, self.tts.target_sample_rate)(audio)
        audio = audio.to(self.device)
        torch.manual_seed(seed)
        dur = max(10, int(self.tts.predict_duration(str(tmp), item["gt_text"], item["prompt_text"], dp_softmax_range=0.7, temperature=0)))
        tmp.unlink()
        text = self.convert_char_to_pinyin([ref_text + item["gt_text"]])
        cond, text_ids, step_cond, cond_mask, _, dur_tensor = self.tts._prepare_inputs(audio, text, audio.shape[-1] // self.tts.hop_length + dur)
        prompt_len = cond_mask.sum().item()
        gen = sampler.make_local_generator(seed)
        y0 = self.pad_sequence([torch.randn(d, 100, dtype=torch.float32, generator=gen) for d in dur_tensor],
                               padding_value=0, batch_first=True).to(self.device)
        return {"cond": cond, "text_ids": text_ids, "step_cond": step_cond, "prompt_len": prompt_len, "y0": y0,
                "prompt_snr": prompt_snr, "seed": seed, "prompt": prompt, "sr": sr}

    def worked_example(self, item, condition, out_dir):
        """Real intermediate values for Appendix A: prompt frame energies, endpoint norms, one endpoint entry.

        Repeats the arithmetic of estimate_prompt_snr and run_teacher_single_pass step by step and checks
        that it gives the same SNR and the same d_i as those functions, so the appendix numbers are exact.
        """
        torch, np, sampler = self.torch, self.np, self.sampler
        ctx = self.prepare(item, condition, item["input_seed"], out_dir)
        # Prompt SNR: 25 ms frames, 10 ms hop, top half vs bottom 10% (Definition 3.3).
        x, sr = ctx["prompt"].squeeze(), ctx["sr"]
        flen, hop = round(0.025 * sr), round(0.010 * sr)
        n = 1 + (len(x) - flen) // hop
        frames = np.lib.stride_tricks.as_strided(x, shape=(n, flen), strides=(x.strides[0] * hop, x.strides[0]))
        energy = np.mean(frames**2, axis=-1)                  # same view and reduction as estimate_prompt_snr
        e_sorted = np.sort(energy)
        bottom, top = max(1, round(0.10 * n)), n // 2
        e_bot, e_top = float(np.mean(e_sorted[:bottom])), float(np.mean(e_sorted[top:]))
        snr = float(np.clip(10 * np.log10((e_top + 1e-8) / (e_bot + 1e-8)), 0.0, 50.0))
        res = {"utt_id": item["utt_id"], "condition": condition, "snr": {
            "frames": int(n), "frame_ms": 25, "hop_ms": 10, "bottom_frames": int(bottom), "top_frames": int(n - top),
            "mean_energy_bottom": e_bot, "mean_energy_top": e_top, "snr_db": snr,
            "estimate_prompt_snr_db": float(ctx["prompt_snr"]), "matches": abs(snr - ctx["prompt_snr"]) < 1e-4,
            "first_10_frame_energies": [float(e) for e in energy[:10]]}}
        # Teacher steps 0..6 with the guided velocity, endpoint estimate and stabilization score.
        grid, plen, state, prev = sampler.SWAY_TEACHER_GRID, ctx["prompt_len"], ctx["y0"].clone(), None
        steps, f, m = {}, plen + 10, 50          # one entry: generated frame 10, mel bin 50
        with torch.inference_mode():
            for i in range(7):
                t = torch.tensor(grid[i], device=self.device, dtype=torch.float32)
                kw = dict(x=state, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
                pred = self.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
                null = self.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
                v = pred + (pred - null) * sampler.CFG_STRENGTH
                ep = state + (1.0 - grid[i]) * v
                rec = {"t": grid[i], "endpoint_norm_M": float(torch.norm(ep[:, plen:, :]))}
                if prev is not None:
                    rec["change_norm_M"] = float(torch.norm((ep - prev)[:, plen:, :]))
                    rec["d"] = rec["change_norm_M"] / (rec["endpoint_norm_M"] + 1e-8)
                if i == 5:
                    rec["entry"] = {"frame": int(f - plen), "mel_bin": m, "x_t": float(state[0, f, m]),
                                    "v": float(v[0, f, m]), "endpoint": float(ep[0, f, m])}
                steps[i] = rec
                prev = ep
                state = state + (grid[i + 1] - grid[i]) * v
        _, drift = self.teacher(ctx, (6,))
        res["teacher_steps"] = steps
        res["max_abs_d_diff_vs_sampler"] = max(abs(steps[i]["d"] - drift[i]) for i in range(1, 7))
        res["drift_matches_sampler"] = res["max_abs_d_diff_vs_sampler"] < 1e-5   # float32 vs float64 rounding only
        return res

    def verify(self, item, out_dir):
        """The three checks of Section 3.3.5 on one clean case, plus single-pass equivalence.

        1. Determinism: two runs with the same seed give bitwise-identical audio and drift.
        2. Forced fallback: the online early-stopping teacher of the T4 timing benchmark, with
           lambda = -1, stops at Step 8 and reproduces the reference audio exactly; with the gate
           open and lambda = +inf it stops at Step 4 and reproduces the Step-4 audio.
        3. Forward passes: counters around the teacher and student networks record 2(k+1) + 3.
        Also: the single teacher pass gives the same endpoint as a separate run to each step.
        """
        torch, np = self.torch, self.np
        sys.path.insert(0, str(Path(self.data_dir).parents[2] / "Handover Scheduling" / "colab"))
        from run_gpu_full_pipeline_timing_benchmark import run_early_stop_teacher
        res = {"utt_id": item["utt_id"]}
        seed = item["input_seed"]
        a1 = self.synthesize(item, "clean", seed, STEPS, out_dir)
        a2 = self.synthesize(item, "clean", seed, STEPS, out_dir)
        res["determinism"] = all(np.array_equal(a1[0][k], a2[0][k]) for k in STEPS) and a1[2] == a2[2]
        # If not bitwise identical, the size tells GPU kernel jitter (~1e-6) apart from a real difference.
        res["determinism_max_abs_audio_diff"] = max(float(np.max(np.abs(a1[0][k] - a2[0][k]))) for k in STEPS)
        res["determinism_max_abs_drift_diff"] = max(abs(a1[2][i] - a2[2][i]) for i in a1[2])
        ctx = self.prepare(item, "clean", seed, out_dir)
        endpoints, _ = self.teacher(ctx, STEPS)
        res["single_pass_equals_separate_runs"] = all(torch.equal(self.teacher(ctx, (k,))[0][k], endpoints[k]) for k in STEPS)
        online = {}
        for label, lam, expect in (("forced_fallback", -1.0, 8), ("gate_open_lambda_inf", 1e9, 4)):
            ep, step, *_ = run_early_stop_teacher(self.tts, ctx["step_cond"], ctx["text_ids"], ctx["y0"], ctx["prompt_len"],
                                                  50.0, "proposed", lambda_thresh=lam, snr_thresh=THETA)
            online[label] = {"exit_step": int(step), "expected_step": expect,
                             "endpoint_identical": bool(torch.equal(ep, endpoints[expect])),
                             "audio_identical": bool(np.array_equal(self.student(ctx, ep), a1[0][expect]))}
        res["online_policy"] = online
        counts = {"teacher": 0, "student": 0}
        restore = []

        def count_calls(attr, key):
            """Count calls of tts.<attr>: wrap the module's forward, or the attribute if it is a plain callable."""
            target = getattr(self.tts, attr)
            holder, name = (target, "forward") if isinstance(target, torch.nn.Module) else (self.tts, attr)
            original = getattr(holder, name)

            def wrapped(*a, **kw):
                counts[key] += 1
                return original(*a, **kw)
            setattr(holder, name, wrapped)
            restore.append((holder, name, original, holder is target))

        count_calls("teacher", "teacher")
        count_calls("generator", "student")
        try:
            nfe = {}
            for k in STEPS:
                counts.update(teacher=0, student=0)
                self.student(ctx, self.teacher(ctx, (k,))[0][k])
                nfe[k] = {"teacher": counts["teacher"], "student": counts["student"], "total": counts["teacher"] + counts["student"],
                          "expected": COST[k]}
        finally:
            for holder, name, original, instance_forward in restore:
                if instance_forward:
                    del holder.forward          # back to the class's forward
                else:
                    setattr(holder, name, original)
        res["forward_passes"] = nfe
        res["all_passed"] = (res["determinism"] and res["single_pass_equals_separate_runs"]
                             and all(v["exit_step"] == v["expected_step"] and v["endpoint_identical"] and v["audio_identical"] for v in online.values())
                             and all(v["total"] == v["expected"] for v in nfe.values()))
        return res

    def ground_truth(self, item):
        path = self.data_dir / item["gt_wav_rel"]
        if not path.exists() or path.stat().st_size < 1000:
            raise SystemExit(f"Ground-truth audio missing: {path}")
        raw, sr = self.torchaudio.load(str(path))
        return raw.squeeze().numpy(), sr

    def score(self, audio, item, gt):
        hyp = self.metrics.asr.transcribe(audio, sample_rate=24000)
        from core.metrics import compute_wer_details
        wer = compute_wer_details(item["gt_text"], hyp)["wer"]
        sim = self.metrics.compute_similarity(audio, gt[0], sample_rate=24000, gt_sr=gt[1])
        return {"wer": float(wer), "similarity": float(sim), "utmos": float(self.metrics.compute_utmos(audio, sample_rate=24000)), "hypothesis": hyp}


# ---------------------------------------------------------------------- runners
def append_rows(path, rows):
    """Append one case's rows and flush them to disk (Google Drive) before the next case starts."""
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def repair_csv(path):
    """Drop a half-written last line left by a disconnect, so the file parses and the run resumes."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])
        print(f"[resume] removed a partial last line from {path.name}")


def done_keys(path, rows_per_case):
    """Cases with all their rows written; a case cut off mid-write is run again."""
    repair_csv(path)
    if not Path(path).exists():
        return set()
    counts = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["utt_id"], r["condition"])
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n >= rows_per_case}


def save_failure_audio(pipe, out_dir, item, condition, step, audio, ref_audio):
    d = Path(out_dir) / "listening"
    d.mkdir(exist_ok=True)
    stem = f"{item['utt_id']}_{condition}"
    pipe.sf.write(str(d / f"{stem}_step{step}.wav"), audio, 24000)
    pipe.sf.write(str(d / f"{stem}_step8_reference.wav"), ref_audio, 24000)


def run_grid(pipe, items, conditions, test, out_dir, csv_name):
    path = Path(out_dir) / csv_name
    done = done_keys(path, len(STEPS))
    todo = [(it, c) for it in items for c in conditions if (it["utt_id"], c) not in done]
    print(f"[{test}] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    for i, (item, cond) in enumerate(todo, 1):
        seed = item["input_seed"]
        audios, snr, drift = pipe.synthesize(item, cond, seed, STEPS, out_dir)
        gt = pipe.ground_truth(item)
        ref = pipe.score(audios[REFERENCE_STEP], item, gt)
        rows = []
        for k in STEPS:
            m = ref if k == REFERENCE_STEP else pipe.score(audios[k], item, gt)
            fail, diag = (False, {"wer_failed": False, "sim_failed": False}) if k == REFERENCE_STEP else \
                pipe.check_failure(m["wer"], ref["wer"], m["similarity"], ref["similarity"], config=pipe.cfg)
            if fail:
                save_failure_audio(pipe, out_dir, item, cond, k, audios[k], audios[REFERENCE_STEP])
            rows.append({"test": test, "utt_id": item["utt_id"], "speaker_id": item["speaker_id"], "condition": cond,
                         "base_seed": 42, "input_seed": seed, "teacher_step": k, "nfe": COST[k], "prompt_snr": snr,
                         "d_4": drift.get(4, -1.0), "d_5": drift.get(5, -1.0), "d_6": drift.get(6, -1.0),
                         "wer": m["wer"], "similarity": m["similarity"], "utmos": m["utmos"], "wer_ref": ref["wer"],
                         "similarity_ref": ref["similarity"], "utmos_ref": ref["utmos"], "is_failure": bool(fail),
                         "fail_wer": bool(diag["wer_failed"]), "fail_sim": bool(diag["sim_failed"]),
                         "hypothesis": m["hypothesis"], "audio_duration": len(audios[k]) / 24000.0,
                         "gt_sr": gt[1], "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        append_rows(path, rows)
        print(f"[{i}/{len(todo)}] {test}: {item['utt_id']} {cond}", flush=True)
    print(f"[OK] {test} complete: {path}", flush=True)


def run_seed_floor(pipe, items, out_dir, csv_name="seed_floor.csv"):
    path = Path(out_dir) / csv_name
    done = done_keys(path, 2)
    todo = [(it, c) for it in items for c in GRID_CONDITIONS if (it["utt_id"], c) not in done]
    print(f"[seed-floor] {len(todo)} cases to run ({len(done)} already in {path.name}) on {pipe.gpu}", flush=True)
    for i, (item, cond) in enumerate(todo, 1):
        gt = pipe.ground_truth(item)
        seeds = {"A": item["input_seed"], "B": derive_input_seed(SEED_B_BASE, item["utt_id"])}
        out = {}
        for tag, seed in seeds.items():
            # Both runs degrade the prompt with the manifest seed, so only the synthesis seed differs.
            audios, snr, drift = pipe.synthesize(item, cond, seed, (REFERENCE_STEP,), out_dir, noise_seed=item["input_seed"])
            out[tag] = (audios[REFERENCE_STEP], pipe.score(audios[REFERENCE_STEP], item, gt), snr, drift)
        a, b = out["A"][1], out["B"][1]
        fail, diag = pipe.check_failure(b["wer"], a["wer"], b["similarity"], a["similarity"], config=pipe.cfg)
        if fail:
            save_failure_audio(pipe, out_dir, item, cond + "_seedB", REFERENCE_STEP, out["B"][0], out["A"][0])
        rows = []
        for tag, k in (("A", REFERENCE_STEP), ("B", REFERENCE_STEP)):
            m = out[tag][1]
            rows.append({"test": f"seed_floor_{tag}", "utt_id": item["utt_id"], "speaker_id": item["speaker_id"],
                         "condition": cond, "base_seed": 42 if tag == "A" else SEED_B_BASE, "input_seed": seeds[tag],
                         "teacher_step": k, "nfe": COST[k], "prompt_snr": out[tag][2], "d_4": out[tag][3].get(4, -1.0),
                         "d_5": out[tag][3].get(5, -1.0), "d_6": out[tag][3].get(6, -1.0), "wer": m["wer"],
                         "similarity": m["similarity"], "utmos": m["utmos"], "wer_ref": a["wer"],
                         "similarity_ref": a["similarity"], "utmos_ref": a["utmos"],
                         "is_failure": bool(fail) if tag == "B" else False,
                         "fail_wer": bool(diag["wer_failed"]) if tag == "B" else False,
                         "fail_sim": bool(diag["sim_failed"]) if tag == "B" else False, "hypothesis": m["hypothesis"],
                         "audio_duration": len(out[tag][0]) / 24000.0, "gt_sr": gt[1],
                         "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        append_rows(path, rows)
        print(f"[{i}/{len(todo)}] seed-floor: {item['utt_id']} {cond}", flush=True)
    print(f"[OK] seed-floor complete: {path}", flush=True)


def run_verification(pipe, items, out_dir):
    """Run Pipeline.verify on each item and write verification.json; an exception is recorded, not raised."""
    import traceback
    checks = []
    for it in items:
        try:
            checks.append(pipe.verify(it, out_dir))
        except Exception:
            checks.append({"utt_id": it["utt_id"], "all_passed": False, "error": traceback.format_exc()})
    res = {"gpu": pipe.gpu, "checks": checks, "all_passed": all(c["all_passed"] for c in checks)}
    (Path(out_dir) / "verification.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] verification checks {'passed' if res['all_passed'] else 'FAILED'}: {Path(out_dir) / 'verification.json'}", flush=True)
    return res


# Appendix A's utterance: the first clean development utterance (by ID) that the policy of Definition 3.4
# at lambda = 0.50, theta = 20 dB keeps past Step 4 and stops at Step 5 (selected from the committed T4 grid).
WORKED_EXAMPLE_UTT = "common_voice_en_13900-common_voice_en_13901"


def run_worked_example(pipe, out_dir):
    """Write worked_example.json (clean and noisy15); an exception is recorded, not raised."""
    import traceback
    item = next(it for it in split_items("development") if it["utt_id"] == WORKED_EXAMPLE_UTT)
    res = {}
    for cond in GRID_CONDITIONS:
        try:
            res[cond] = pipe.worked_example(item, cond, out_dir)
        except Exception:
            res[cond] = {"error": traceback.format_exc()}
    (Path(out_dir) / "worked_example.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    ok = all("error" not in r and r["snr"]["matches"] and r["drift_matches_sampler"] for r in res.values())
    print(f"[OK] worked example {'recorded' if ok else 'FAILED or mismatched'}: {Path(out_dir) / 'worked_example.json'}", flush=True)
    return res


def pilot_items(n):
    """The first n development speakers by utterance ID. Never fresh or test speakers."""
    return sorted(split_items("development"), key=lambda it: it["utt_id"])[:n]


def fresh_items(out):
    path = Path(out) / "fresh_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_gpu(args):
    out = Path(args.out)
    if args.cmd == "smoke":
        out = out / "smoke"
        out.mkdir(parents=True, exist_ok=True)
        pipe = Pipeline(args.dmo_root, args.device)
        if args.demand_dir:
            pipe.load_babble(args.demand_dir)
        dev = split_items("development")[:2]
        conds = ("clean", "noisy15", "reverb09", "telephone") + (("babble15",) if args.demand_dir else ())
        ver = run_verification(pipe, dev[:1], out)
        run_grid(pipe, dev, conds, "smoke", out, "smoke_grid.csv")
        run_seed_floor(pipe, dev[:1], out, "smoke_seed_floor.csv")
        print(f"Smoke run complete: {out}; verification checks {'passed' if ver['all_passed'] else 'FAILED, see verification.json'}")
        return
    if args.cmd == "pilot":
        out = out / "pilot"
        out.mkdir(parents=True, exist_ok=True)
        pipe = Pipeline(args.dmo_root, args.device)
        conds = args.conditions.split(",") if args.conditions else list(PILOT_CONDITIONS)
        unknown = [c for c in conds if c not in CONDITIONS]
        if unknown:
            raise SystemExit(f"Unknown conditions: {unknown}")
        if any(CONDITIONS[c][0] == "babble" for c in conds):
            pipe.load_babble(args.demand_dir or "")
        dev = pilot_items(args.n_speakers)
        if not (out / "verification.json").exists():
            run_verification(pipe, dev[:2], out)
        if not (out / "worked_example.json").exists():
            run_worked_example(pipe, out)
        run_grid(pipe, dev, conds, "pilot", out, "pilot_grid.csv")
        run_seed_floor(pipe, dev[: max(1, args.n_speakers // 2)], out, "pilot_seed_floor.csv")
        pilot_report(Path(args.out))
        return
    prereg = load_prereg(out)
    pipe = Pipeline(args.dmo_root, args.device)
    if args.cmd == "seed-floor":
        run_seed_floor(pipe, split_items("test"), out)
    elif args.cmd == "fresh-grid":
        run_grid(pipe, fresh_items(out), GRID_CONDITIONS, "fresh_grid", out, "fresh_grid.csv")
    else:
        conds = args.conditions.split(",") if args.conditions else list(DEGRADATION_CONDITIONS)
        unknown = [c for c in conds if c not in DEGRADATION_CONDITIONS]
        if unknown:
            raise SystemExit(f"Unknown degradation conditions: {unknown}")
        if any(CONDITIONS[c][0] == "babble" for c in conds):
            pipe.load_babble(args.demand_dir or "")
            (out / "demand_sha256.txt").write_text(f"{DEMAND_FILE} {pipe.babble_sha}\n", encoding="utf-8")
        run_grid(pipe, fresh_items(out), conds, "degradations", out, "degradations.csv")
    (out / f"run_info_{args.cmd}.json").write_text(json.dumps({
        "gpu": pipe.gpu, "torch": pipe.torch.__version__, "cuda": pipe.torch.version.cuda,
        "dmo_commit": git_head(args.dmo_root), "thesis_commit": git_head(THESIS_ROOT),
        "preregistration_sha256": prereg["sha256"], "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ analysis
def read_csv(path, repair=False):
    """repair=True only for this run's own output files, never for committed evidence."""
    if not Path(path).exists():
        raise SystemExit(f"Missing input: {path}")
    if repair:
        repair_csv(path)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def analyze(out, allow_partial=False):
    """Every number in the report is computed here from the CSV files; nothing is typed in."""
    out = Path(out)
    prereg = load_prereg(out)
    res = {"preregistration_sha256": prereg["sha256"], "partial": False}

    # Calibration on the committed T4 grid (as frozen).
    split_of = {}
    for name in ("development", "calibration", "test"):
        for r in json.loads((SPLITS_DIR / f"{name}_manifest.json").read_text(encoding="utf-8")):
            split_of[r["utt_id"]] = name
    grid = index_cases(read_csv(T4_GRID))
    calib = sorted({u for (u, _) in grid if split_of.get(u) == "calibration"})
    test = sorted({u for (u, _) in grid if split_of.get(u) == "test"})
    lam_hat, table = ltt(grid, calib, GRID_CONDITIONS)
    res["calibration"] = {"lambda_hat": lam_hat, "n": len(calib), "table": table}
    lam = lam_hat if lam_hat is not None else -1.0

    # H1: fresh speakers, clean + noisy15.
    fresh = index_cases(read_csv(out / "fresh_grid.csv", repair=True))
    expected = [e["utt_id"] for e in fresh_items(out)]
    have = [u for u in expected if all((u, c) in fresh for c in GRID_CONDITIONS)]
    if len(have) < len(expected):
        if not allow_partial:
            raise SystemExit(f"fresh_grid.csv covers {len(have)}/{len(expected)} speakers; finish the run or pass --allow_partial.")
        res["partial"] = True
    h1 = {}
    for pol in ("reference", "snr_only", "snr_drift_ltt", "snr_drift_050", "snr_step5"):
        n, k, fails, nfe = evaluate(fresh, have, GRID_CONDITIONS, pol, lam)
        lo, hi = clopper_pearson(k, n)
        h1[pol] = {"n": n, "failures": k, "risk": k / n, "ci95": [lo, hi], "mean_nfe": nfe,
                   "p_risk_gt_alpha": binom_cdf(k, n, ALPHA), "fails": fails}
    res["H1_heldout"] = {pol: {k: v for k, v in d.items() if k != "fails"} for pol, d in h1.items()}
    # The committed T4 grid scored SIM with core/metrics.py before 656d86f, which read the ground-truth
    # array as 24 kHz whatever its rate. New runs honor the rate; record which rates occurred.
    res["ground_truth_sample_rates"] = sorted({int(r[REFERENCE_STEP]["gt_sr"]) for r in fresh.values() if r[REFERENCE_STEP].get("gt_sr")})
    res["H1_heldout"]["mcnemar_snr_only_vs_snr_drift_ltt"] = dict(zip(("only_snr_only", "only_snr_drift", "p"), mcnemar_exact(h1["snr_only"]["fails"], h1["snr_drift_ltt"]["fails"])))

    # H2: degradations, per condition.
    deg_path = out / "degradations.csv"
    if deg_path.exists():
        deg = index_cases(read_csv(deg_path, repair=True))
        h2, pvals = {}, []
        for c in DEGRADATION_CONDITIONS:
            utts = [u for u in expected if (u, c) in deg]
            if not utts:
                continue
            if len(utts) < len(expected):
                if not allow_partial:
                    raise SystemExit(f"degradations.csv covers {len(utts)}/{len(expected)} speakers for {c}.")
                res["partial"] = True
            row = {"n": len(utts)}
            fl = {}
            for pol in ("snr_only", "snr_drift_050", "snr_drift_ltt", "snr_step5"):
                n, k, fl[pol], nfe = evaluate(deg, utts, (c,), pol, lam)
                row[pol] = {"failures": k, "risk": k / n, "ci95": list(clopper_pearson(k, n)), "mean_nfe": nfe}
            snrs = sorted(float(deg[(u, c)][REFERENCE_STEP]["prompt_snr"]) for u in utts)
            row["prompt_snr_min_median_max"] = [snrs[0], snrs[len(snrs) // 2], snrs[-1]]
            row["gate_fallback_rate"] = sum(s < THETA for s in snrs) / len(snrs)
            a, b, p = mcnemar_exact(fl["snr_only"], fl["snr_drift_050"])
            row["mcnemar_snr_only_vs_snr_drift_050"] = {"only_snr_only": a, "only_snr_drift": b, "p": p}
            pvals.append(p)
            h2[c] = row
        for c, p_adj in zip(h2, holm(pvals)):
            h2[c]["mcnemar_snr_only_vs_snr_drift_050"]["p_holm"] = p_adj
        res["H2_drift_value"] = h2
    else:
        res["H2_drift_value"] = "not run"

    # H3: seed floor on the test split.
    sf_path = out / "seed_floor.csv"
    if sf_path.exists():
        rows = read_csv(sf_path, repair=True)
        by = {}
        for r in rows:
            by.setdefault(r["utt_id"], {})[(r["test"], r["condition"])] = r
        complete = [u for u, v in by.items() if all(("seed_floor_B", c) in v for c in GRID_CONDITIONS)]
        if len(complete) < len(test) and not allow_partial:
            raise SystemExit(f"seed_floor.csv covers {len(complete)}/{len(test)} test speakers.")
        k = sum(any(is_true(by[u][("seed_floor_B", c)]["is_failure"]) for c in GRID_CONDITIONS) for u in complete)
        clean_k = sum(is_true(by[u][("seed_floor_B", "clean")]["is_failure"]) for u in complete)
        res["H3_seed_floor"] = {"n": len(complete), "failures": k, "rate": k / len(complete),
                                "ci95": list(clopper_pearson(k, len(complete))), "clean_failures": clean_k}
        # Reproducibility of seed A against the committed grid.
        dw, ds = [], []
        for u in complete:
            for c in GRID_CONDITIONS:
                g = grid.get((u, c))
                if g:
                    dw.append(abs(float(by[u][("seed_floor_A", c)]["wer"]) - float(g[REFERENCE_STEP]["wer"])))
                    ds.append(abs(float(by[u][("seed_floor_A", c)]["similarity"]) - float(g[REFERENCE_STEP]["similarity"])))
        res["seed_A_vs_t4_grid"] = {"cases": len(dw), "max_abs_wer_diff": max(dw), "max_abs_sim_diff": max(ds),
                                    "identical_wer_cases": sum(d == 0 for d in dw)}
    else:
        res["H3_seed_floor"] = "not run"

    # Secondary, development evidence on the committed grid.
    res.update(t4_grid_dev_analyses(grid, calib, test))

    (out / "analysis_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "FOLLOWUP_REPORT.md").write_text(render_report(res), encoding="utf-8")
    print(f"Wrote {out / 'analysis_results.json'} and {out / 'FOLLOWUP_REPORT.md'}")
    return res


def pilot_report(out):
    """Development pilot summary, computed from OUT/pilot/*.csv and the committed T4 grid."""
    import statistics
    pdir = Path(out) / "pilot"
    rows = read_csv(pdir / "pilot_grid.csv", repair=True)
    cases = index_cases(rows)
    grid = index_cases(read_csv(T4_GRID))
    res = {"note": "Development pilot (development-split speakers only). Assurance and planning, not evidence for the method."}

    # 1. Reproduction of the committed T4 grid on clean + noisy15.
    rep = {"cases": 0, "same_failure_label": 0, "rows": 0}
    diffs = {"prompt_snr": [], "d_4": [], "wer": [], "similarity": []}
    for (u, c), steps in cases.items():
        if c not in GRID_CONDITIONS or (u, c) not in grid:
            continue
        rep["cases"] += 1
        for k in STEPS:
            new, old = steps[k], grid[(u, c)][k]
            rep["rows"] += 1
            rep["same_failure_label"] += is_true(new["is_failure"]) == is_true(old["is_failure"])
            for col in diffs:
                diffs[col].append(abs(float(new[col]) - float(old[col])))
    rep.update({f"max_abs_diff_{k}": (max(v) if v else None) for k, v in diffs.items()})
    res["reproduction_vs_t4_grid"] = rep

    # 2. Per condition: does the gate let prompts through, and does drift ever matter?
    per = {}
    for c in dict.fromkeys(c for (_, c) in cases):
        utts = sorted(u for (u, cc) in cases if cc == c)
        snrs = [float(cases[(u, c)][REFERENCE_STEP]["prompt_snr"]) for u in utts]
        row = {"n": len(utts), "prompt_snr_min_max": [min(snrs), max(snrs)],
               "gate_fallback_rate": sum(s < THETA for s in snrs) / len(snrs),
               "step4_failures_any_prompt": sum(is_true(cases[(u, c)][4]["is_failure"]) for u in utts)}
        fl = {}
        for pol in ("snr_only", "snr_drift_050", "snr_step5"):
            n, k, fl[pol], nfe = evaluate(cases, utts, (c,), pol, 0.0)
            row[pol] = {"failures": k, "mean_nfe": nfe}
        a, b, _ = mcnemar_exact(fl["snr_only"], fl["snr_drift_050"])
        row["discordant_only_snr_only_fails"], row["discordant_only_snr_drift_fails"] = a, b
        per[c] = row
    res["conditions"] = per

    # 3. Seed-noise floor (small sample).
    sf = read_csv(pdir / "pilot_seed_floor.csv", repair=True)
    b_rows = [r for r in sf if r["test"] == "seed_floor_B"]
    spk = {}
    for r in b_rows:
        spk[r["utt_id"]] = spk.get(r["utt_id"], False) or is_true(r["is_failure"])
    res["seed_floor"] = {"speakers": len(spk), "failures": sum(spk.values())}

    # 4. Time per case from completion timestamps.
    done = {}
    for r in rows:
        done[(r["utt_id"], r["condition"])] = r["utc"]
    stamps = sorted(datetime.datetime.fromisoformat(t) for t in done.values())
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    sec = statistics.median(gaps) if gaps else None
    res["seconds_per_case_median"] = sec
    if sec:
        res["estimated_full_run_hours"] = {"fresh_grid": 332 * sec / 3600, "degradations": 1494 * sec / 3600,
                                           "seed_floor_upper_bound": 400 * sec / 3600}
    res["ground_truth_sample_rates"] = sorted({int(r["gt_sr"]) for r in rows if r.get("gt_sr")})

    # 5. The Section 3.3.5 checks, if the pilot ran them.
    ver_path = pdir / "verification.json"
    res["verification"] = json.loads(ver_path.read_text(encoding="utf-8")) if ver_path.exists() else None

    # 5b. Appendix A worked example, if recorded.
    we_path = pdir / "worked_example.json"
    res["worked_example"] = json.loads(we_path.read_text(encoding="utf-8")) if we_path.exists() else None

    # 6. Offline analyses on the committed T4 grid (no GPU).
    split_of = {}
    for name in ("calibration", "test"):
        for r in json.loads((SPLITS_DIR / f"{name}_manifest.json").read_text(encoding="utf-8")):
            split_of[r["utt_id"]] = name
    calib = sorted({u for (u, _) in grid if split_of.get(u) == "calibration"})
    test = sorted({u for (u, _) in grid if split_of.get(u) == "test"})
    lam_hat, lam_table = ltt(grid, calib, GRID_CONDITIONS)
    res["t4_calibration"] = {"lambda_hat": lam_hat, "n": len(calib), "table": lam_table}
    res.update(t4_grid_dev_analyses(grid, calib, test))

    (pdir / "pilot_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    pct = lambda x: f"{100 * x:.0f}%"
    L = ["# Development pilot", "", res["note"], "", "## Pre-experiment checks (Section 3.3.5)", ""]
    v = res["verification"]
    if v:
        L.append(f"Overall: **{'PASSED' if v['all_passed'] else 'FAILED'}** on {len(v['checks'])} clean development cases ({v['gpu']}).")
        L += ["", "| Case | Determinism | Single pass = separate runs | Forced fallback = Step 8 | Gate open, λ = ∞ → Step 4 | Forward passes = 2(k+1)+3 |",
              "|---|---|---|---|---|---|"]
        ok = lambda b: "yes" if b else "**no**"
        for c in v["checks"]:
            if "error" in c:
                L.append(f"| {c['utt_id'][-24:]} | **error**: {c['error'].strip().splitlines()[-1][:120]} | | | | |")
                continue
            o = c["online_policy"]
            fb, op = o["forced_fallback"], o["gate_open_lambda_inf"]
            nfe = ", ".join(f"k={k}: {d['total']}/{d['expected']}" for k, d in c["forward_passes"].items())
            det = ok(c["determinism"]) + ("" if c["determinism"] else f" (max |Δaudio| {c['determinism_max_abs_audio_diff']:.2e})")
            L.append(f"| {c['utt_id'][-24:]} | {det} | {ok(c['single_pass_equals_separate_runs'])} | "
                     f"{ok(fb['exit_step'] == 8 and fb['endpoint_identical'] and fb['audio_identical'])} | "
                     f"{ok(op['exit_step'] == 4 and op['endpoint_identical'] and op['audio_identical'])} | {nfe} |")
    else:
        L.append("Not run.")
    L += ["", "## Appendix A worked example (real intermediate values)", ""]
    we = res["worked_example"]
    if we:
        for cond, w in we.items():
            if "error" in w:
                L.append(f"- {cond}: **error**: {w['error'].strip().splitlines()[-1][:160]}")
                continue
            s = w["snr"]
            L.append(f"- {cond}: {s['frames']} frames; quietest {s['bottom_frames']} mean energy {s['mean_energy_bottom']:.3e}, "
                     f"loudest {s['top_frames']} mean energy {s['mean_energy_top']:.3e} → Ŝ = {s['snr_db']:.2f} dB "
                     f"(estimate_prompt_snr {s['estimate_prompt_snr_db']:.2f}; {'matches' if s['matches'] else '**mismatch**'}).")
            d = ", ".join(f"d{i} = {w['teacher_steps'][str(i)]['change_norm_M']:.2f}/{w['teacher_steps'][str(i)]['endpoint_norm_M']:.2f} = "
                          f"{w['teacher_steps'][str(i)]['d']:.3f}" for i in (4, 5, 6)) if isinstance(next(iter(w["teacher_steps"])), str) else \
                ", ".join(f"d{i} = {w['teacher_steps'][i]['change_norm_M']:.2f}/{w['teacher_steps'][i]['endpoint_norm_M']:.2f} = {w['teacher_steps'][i]['d']:.3f}" for i in (4, 5, 6))
            L.append(f"  {d} ({'matches the sampler' if w['drift_matches_sampler'] else '**differs from the sampler**'}).")
    else:
        L.append("Not recorded.")
    L += ["", "## Reproduction of the committed T4 grid (clean + noisy15)", ""]
    if rep["cases"]:
        L += [f"{rep['cases']} cases, {rep['rows']} rows; same failure label in {rep['same_failure_label']}/{rep['rows']} rows. "
              f"Max |Δ|: prompt SNR {rep['max_abs_diff_prompt_snr']:.4f} dB, d₄ {rep['max_abs_diff_d_4']:.6f}, "
              f"WER {rep['max_abs_diff_wer']:.4f}, SIM {rep['max_abs_diff_similarity']:.4f}."]
    else:
        L.append("No clean or noisy15 cases in this pilot.")
    L += ["", "## Conditions", "",
          "| Condition | n | Prompt SNR range (dB) | Gate fallback | Step-4 failures | SNR-only fails / NFE | SNR + drift 0.50 fails / NFE | Only SNR-only fails | Only SNR + drift fails |",
          "|---|---:|---|---:|---:|---|---|---:|---:|"]
    for c, d in per.items():
        L.append(f"| {c} | {d['n']} | {d['prompt_snr_min_max'][0]:.1f}–{d['prompt_snr_min_max'][1]:.1f} | {pct(d['gate_fallback_rate'])} | "
                 f"{d['step4_failures_any_prompt']} | {d['snr_only']['failures']} / {d['snr_only']['mean_nfe']:.2f} | "
                 f"{d['snr_drift_050']['failures']} / {d['snr_drift_050']['mean_nfe']:.2f} | "
                 f"{d['discordant_only_snr_only_fails']} | {d['discordant_only_snr_drift_fails']} |")
    s = res["seed_floor"]
    L += ["", f"Seed-noise floor: Step 8 with seed B fails against seed A for {s['failures']}/{s['speakers']} speakers.", "",
          f"Ground-truth sample rates: {res['ground_truth_sample_rates']}."]
    if sec:
        e = res["estimated_full_run_hours"]
        L += ["", f"Median {sec:.1f} s per case. Estimated full run: fresh grid {e['fresh_grid']:.1f} h, "
                  f"degradations {e['degradations']:.1f} h, seed floor at most {e['seed_floor_upper_bound']:.1f} h."]
    L += ["", "Small development sample: use it to check the pipeline and plan, not to estimate risk or test drift."]
    t = res["dev_snr_only_theta"]
    cn = res["dev_clean_to_noisy"]
    L += ["", "## Offline analyses on the committed T4 grid (development evidence; no GPU)", "",
          f"- Learn-Then-Test for SNR + drift on the calibration split (n = {res['t4_calibration']['n']}, λ grid 0.20–1.20): λ̂ = {res['t4_calibration']['lambda_hat']}.",
          f"- SNR-only rule with θ calibrated by Learn-Then-Test (θ from 30.0 down to 14.0 dB in 0.5 dB steps): θ̂ = {t['theta_hat']}; "
          f"on the test split {t['test_failures']}/{t['n']} speaker failures, mean NFE {t['test_mean_nfe']:.2f}.",
          f"- Calibrate on clean prompts only: λ̂ = {cn['lambda_hat_clean_only']}; on test noisy15 prompts {cn['test_noisy15_failures']}/{cn['n']} failures, mean NFE {cn['mean_nfe']:.2f}.",
          "", "| Calibration size | No threshold certified | Median λ̂ | Mean test risk | Mean test NFE |", "|---:|---:|---:|---:|---:|"]
    for size, s_ in res["dev_calibration_size"].items():
        L.append(f"| {size} | {pct(s_['reference_only_share'])} | {s_['median_lambda_hat']} | {100 * s_['mean_test_risk']:.1f}% | {s_['mean_test_nfe']:.2f} |")
    L += ["", "The 200-speaker test split was inspected before any protocol was frozen, so these are development evidence."]
    (pdir / "PILOT_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {pdir / 'PILOT_REPORT.md'}")
    return res


def t4_grid_dev_analyses(grid, calib, test):
    """Development-evidence analyses on the committed T4 grid (the test split was inspected earlier)."""
    import random
    res = {}
    lam_c, _ = ltt(grid, calib, ("clean",))
    n, k, _, nfe = evaluate(grid, test, ("noisy15",), "snr_drift_ltt", lam_c if lam_c is not None else -1.0)
    res["dev_clean_to_noisy"] = {"lambda_hat_clean_only": lam_c, "test_noisy15_failures": k, "n": n, "mean_nfe": nfe}
    th, table = ltt(grid, calib, GRID_CONDITIONS, grid=THETA_GRID, policy="snr_theta")
    n, k, _, nfe = evaluate(grid, test, GRID_CONDITIONS, "snr_theta", th if th is not None else math.inf)
    res["dev_snr_only_theta"] = {"theta_grid": [THETA_GRID[0], THETA_GRID[-1], 0.5], "theta_hat": th,
                                 "calibration_table": table, "test_failures": k, "n": n, "test_mean_nfe": nfe}
    rng = random.Random(2026)
    sizes = {}
    for size in (50, 100, 200):
        picks, risks, nfes = [], [], []
        for _ in range(200):
            sub = rng.sample(calib, size)
            lh, _ = ltt(grid, sub, GRID_CONDITIONS)
            n, k, _, nfe = evaluate(grid, test, GRID_CONDITIONS, "snr_drift_ltt", lh if lh is not None else -1.0)
            picks.append(lh)
            risks.append(k / n)
            nfes.append(nfe)
        certified = sorted(p for p in picks if p is not None)
        sizes[size] = {"reference_only_share": 1 - len(certified) / len(picks),
                       "median_lambda_hat": certified[len(certified) // 2] if certified else None,
                       "mean_test_risk": sum(risks) / len(risks), "mean_test_nfe": sum(nfes) / len(nfes)}
    res["dev_calibration_size"] = sizes
    return res


def render_report(res):
    pct = lambda x: f"{100 * x:.1f}%"
    L = ["# Follow-up T4 tests: results", "",
         f"Preregistration sha256 `{res['preregistration_sha256']}`. Generated by `code/colab/run_followup_tests.py analyze` "
         "from the CSV files in this folder and the committed T4 grid; every number below is computed, none is typed in."
         + (" **PARTIAL: some runs are incomplete.**" if res["partial"] else ""), "",
         f"Calibration (T4 calibration split, n = {res['calibration']['n']}): λ̂ = {res['calibration']['lambda_hat']}.", "",
         f"Ground-truth sample rates in the new runs: {res['ground_truth_sample_rates']}."
         + ("" if res["ground_truth_sample_rates"] in ([], [24000]) else
            " **Not all 24 kHz: the committed T4 grid scored SIM against ground truth read at the wrong rate, so its calibration labels are not comparable with these runs; recalibrate before using H1.**"), "",
         "## H1. Fresh speakers, clean + noisy15 (speaker fails if either fails)", "",
         "| Policy | Failures | Risk | 95% CI | Mean NFE |", "|---|---:|---:|---|---:|"]
    for pol, d in res["H1_heldout"].items():
        if pol.startswith("mcnemar"):
            continue
        L.append(f"| {pol} | {d['failures']}/{d['n']} | {pct(d['risk'])} | {pct(d['ci95'][0])}–{pct(d['ci95'][1])} | {d['mean_nfe']:.2f} |")
    m = res["H1_heldout"]["mcnemar_snr_only_vs_snr_drift_ltt"]
    L += ["", f"McNemar SNR-only vs SNR + drift (λ̂): {m['only_snr_only']} vs {m['only_snr_drift']} discordant, p = {m['p']:.4f}.", ""]
    L += ["## H2. Degraded prompts (fresh speakers)", ""]
    if isinstance(res["H2_drift_value"], dict):
        L += ["| Condition | Gate fallback | SNR-only fails / NFE | SNR + drift 0.50 fails / NFE | McNemar p (Holm) |", "|---|---:|---|---|---:|"]
        for c, d in res["H2_drift_value"].items():
            mm = d["mcnemar_snr_only_vs_snr_drift_050"]
            L.append(f"| {c} | {pct(d['gate_fallback_rate'])} | {d['snr_only']['failures']}/{d['n']} / {d['snr_only']['mean_nfe']:.2f} | "
                     f"{d['snr_drift_050']['failures']}/{d['n']} / {d['snr_drift_050']['mean_nfe']:.2f} | {mm['p']:.4f} ({mm['p_holm']:.4f}) |")
    else:
        L.append("Not run.")
    L += ["", "## H3. Seed-noise floor (test split)", ""]
    h3 = res["H3_seed_floor"]
    if isinstance(h3, dict):
        r = res["seed_A_vs_t4_grid"]
        L += [f"Step 8 with seed B fails against seed A for {h3['failures']}/{h3['n']} speakers ({pct(h3['rate'])}, 95% CI "
              f"{pct(h3['ci95'][0])}–{pct(h3['ci95'][1])}); clean prompts: {h3['clean_failures']}.", "",
              f"Seed A against the committed T4 grid: {r['identical_wer_cases']}/{r['cases']} cases with identical WER; "
              f"max |ΔWER| = {r['max_abs_wer_diff']:.4f}, max |ΔSIM| = {r['max_abs_sim_diff']:.4f}."]
    else:
        L.append("Not run.")
    d = res["dev_clean_to_noisy"]
    L += ["", "## Development evidence on the committed T4 grid", "",
          f"Calibrate on clean only: λ̂ = {d['lambda_hat_clean_only']}; on test noisy15 prompts {d['test_noisy15_failures']}/{d['n']} failures, mean NFE {d['mean_nfe']:.2f}.", "",
          "| Calibration size | No threshold certified | Median λ̂ | Mean test risk | Mean test NFE |", "|---:|---:|---:|---:|---:|"]
    for size, s in res["dev_calibration_size"].items():
        L.append(f"| {size} | {pct(s['reference_only_share'])} | {s['median_lambda_hat']} | {pct(s['mean_test_risk'])} | {s['mean_test_nfe']:.2f} |")
    L += ["", "The test split was inspected before this protocol was frozen, so the last two analyses are development evidence."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["freeze", "smoke", "pilot", "pilot-report", "seed-floor", "fresh-grid", "degradations", "analyze"])
    ap.add_argument("--out", required=True, help="results folder, e.g. on Google Drive")
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2", help="Shonget-DMOSpeech2 clone")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--demand_dir", default=None, help="folder holding PCAFETER/ch01.wav")
    ap.add_argument("--conditions", default=None, help="comma-separated subset of degradation conditions")
    ap.add_argument("--allow_partial", action="store_true", help="analyze: report incomplete runs, marked PARTIAL")
    ap.add_argument("--n_speakers", type=int, default=20, help="pilot: number of development speakers")
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        cmd_freeze(args)
    elif args.cmd == "pilot-report":
        pilot_report(args.out)
    elif args.cmd == "analyze":
        analyze(args.out, args.allow_partial)
    else:
        cmd_gpu(args)


if __name__ == "__main__":
    main()
