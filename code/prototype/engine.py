"""Local DMOSpeech 2 engine for the SNR + Invariant Residual (SNR_R) prototype (CPU demo, not an evaluation tool).

It reproduces the EXP-034 / EXP-035 T4 pipeline: prompt SNR on the prompt as given, RMS normalization to 0.3, duration
predictor, one-pass student draft s_0 at tau = 0.0, guided Euler teacher on the 8-interval sway grid (Steps 0..4,
ending at t = 0.2929), invariant residual r between teacher endpoint x_hat_1 and student draft s_0 over generated
frames, SNR + r routing between CPH8 (teacher endpoint) and CPQH (quantile handover with draft prompt frames), and the
two-step student restart completion at tau in {0.2929, 0.6173}.

The vendored DMOSpeech 2 code (code/vendor/dmospeech2/src) is imported read-only. Checkpoints are read from ckpts/
(model_85000.pt and model_1500.pt) and Seed-TTS examples from research/data/seed-tts-eval/en.
"""
import gc
import io
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR_SRC = ROOT / "code/vendor/dmospeech2/src"
COLAB_DIR = ROOT / "code/colab"
CKPT_DIR = Path(os.environ.get("DMO_CKPT_DIR", ROOT / "ckpts"))
DATA_DIR = Path(os.environ.get("SEED_TTS_EN_DIR", ROOT / "research/data/seed-tts-eval/en"))
VOCAB_FILE = VENDOR_SRC / "f5_tts/infer/examples/vocab.txt"   # the 2545-entry Emilia ZH-EN pinyin vocabulary
CHECKPOINTS = {"model_85000.pt": "e143188f7c46b69abb60786937d79b89dac9777d3c4c1e91a97634bcd39ebf2c",
               "model_1500.pt": "6bd50428f3b5f0f25c1a60a0053f40ee47c817bda03c8ee120100f22680ae4c2"}
OFFLINE_CONDITIONS = ("clean", "gauss25", "pink25", "hp25", "noisy15", "reverb03", "reverb09", "telephone")   # babble needs DEMAND
SAMPLE_RATE = 24000

for p in (HERE, VENDOR_SRC, COLAB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import snr_drift as sd  # noqa: E402
import snr_invariant as si  # noqa: E402


def ensure_sentence_end(ref_text):
    """The prompt-text rule of f5_tts.infer.utils_infer.preprocess_ref_audio_text (its audio processing is discarded by
    the thesis pipeline, so only this part is repeated)."""
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        ref_text += " " if ref_text.endswith(".") else ". "
    return ref_text


def development_examples():
    """Development-split speakers only (never calibration or test), with the experiments' per-utterance seeds."""
    import run_followup_tests as rf
    return rf.split_items("development")


def degrade_prompt(prompt, sr, condition, seed):
    """The experiments' prompt degradations (identical arithmetic including pink25/hp25 from EXP-034)."""
    import run_invariant_residual as ir
    if condition not in OFFLINE_CONDITIONS:
        raise ValueError(f"Unknown or unavailable condition {condition!r}; choose one of {OFFLINE_CONDITIONS}")
    # Kept in the exact dtype returned by ir.degrade, matching the T4 experiments.
    return ir.degrade(np.asarray(prompt, dtype=np.float32), sr, condition, seed)


def wav_bytes(audio, sr=SAMPLE_RATE, peak=0.99):
    """16-bit WAV for listening. The pipeline normalizes the prompt to RMS 0.3, so generated speech can exceed full
    scale; it is scaled down only if it would clip. Playback only: the experiments score the unscaled float audio."""
    import soundfile as sf
    x = np.asarray(audio, dtype=np.float32)
    top = float(np.max(np.abs(x))) if x.size else 0.0
    if top > peak:
        x = x * (peak / top)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _load_vocabulary():
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = {line[:-1]: i for i, line in enumerate(f)}
    assert vocab[" "] == 0 and len(vocab) == 2545, "unexpected vocabulary"
    return vocab, len(vocab)


def _use_bundled_vocabulary():
    """The vendored snapshot lacks data/Emilia_ZH_EN_pinyin/vocab.txt, which DMOInference and the guidance model load by
    dataset name. The same 2545-entry vocabulary ships in f5_tts/infer/examples/vocab.txt; the loader is redirected to
    it in memory, so no vendored file changes."""
    import f5_tts.model.utils as f5u
    import infer as dmo_infer
    if getattr(f5u.get_tokenizer, "_bundled", False):
        return
    original = f5u.get_tokenizer

    def get_tokenizer(dataset_name, tokenizer="pinyin"):
        if dataset_name == "Emilia_ZH_EN" and tokenizer == "pinyin":
            return _load_vocabulary()
        return original(dataset_name, tokenizer)

    get_tokenizer._bundled = True
    f5u.get_tokenizer = get_tokenizer          # guidance_model imports it from here when the model is built
    dmo_infer.get_tokenizer = get_tokenizer    # infer.py bound the name at import


def missing_files():
    out = [f"ckpts/{n}" for n in CHECKPOINTS if not (CKPT_DIR / n).exists()]
    if not (DATA_DIR / "meta.lst").exists():
        out.append("research/data/seed-tts-eval/en")
    return out


class Engine:
    def __init__(self, threads=None, log=print):
        import torch
        self.torch = torch
        self.log = log
        missing = [n for n in CHECKPOINTS if not (CKPT_DIR / n).exists()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints in {CKPT_DIR}: {missing}")
        torch.set_num_threads(int(threads or max(1, (os.cpu_count() or 2) - 1)))
        self.device = torch.device("cpu")
        t0 = time.perf_counter()
        _use_bundled_vocabulary()
        from infer import DMOInference

        log("[engine] loading DMOSpeech 2 (teacher, student, duration predictor, vocoder) on CPU ...")
        self.tts = DMOInference(student_checkpoint_path=str(CKPT_DIR / "model_85000.pt"),
                                duration_predictor_path=str(CKPT_DIR / "model_1500.pt"), device=self.device)
        g = self.tts.model.guidance_model
        for name in ("fake_unet", "cls_pred_branch", "ctc_model", "sv_model_en", "sv_model_zh"):
            if hasattr(g, name):
                delattr(g, name)        # training-only networks; inference uses real_unet (teacher) and the generator
        gc.collect()
        from f5_tts.infer.utils_infer import convert_char_to_pinyin
        self.convert_char_to_pinyin = convert_char_to_pinyin
        self.load_seconds = time.perf_counter() - t0
        self.threads = torch.get_num_threads()
        log(f"[engine] ready in {self.load_seconds:.0f} s ({self.threads} CPU threads)")

    # ------------------------------------------------------------------ preparation (as the experiments do it)
    def prepare(self, prompt, sr, prompt_text, text, seed):
        torch = self.torch
        prompt = np.asarray(prompt)
        if prompt.ndim > 1:
            prompt = prompt.mean(axis=0) if prompt.shape[0] < prompt.shape[-1] else prompt.mean(axis=1)
        prompt = np.ascontiguousarray(prompt)
        prompt_snr = sd.estimate_prompt_snr(prompt, sample_rate=sr)      # on the prompt as given, as in the experiments
        ref_text = ensure_sentence_end(prompt_text.strip())
        audio = torch.from_numpy(prompt).float().unsqueeze(0)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < 0.3:
            audio = audio * 0.3 / rms
        if sr != self.tts.target_sample_rate:
            import torchaudio
            audio = torchaudio.transforms.Resample(sr, self.tts.target_sample_rate)(audio)
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.wav"
            sf.write(str(path), prompt, sr)
            torch.manual_seed(seed)
            dur = max(10, int(self.tts.predict_duration(str(path), text, prompt_text, dp_softmax_range=0.7, temperature=0)))
        text_list = self.convert_char_to_pinyin([ref_text + text])
        cond, text_ids, step_cond, cond_mask, _, dur_tensor = self.tts._prepare_inputs(
            audio, text_list, audio.shape[-1] // self.tts.hop_length + dur)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        from torch.nn.utils.rnn import pad_sequence
        y0 = pad_sequence([torch.randn(d, 100, dtype=torch.float32, generator=gen) for d in dur_tensor],
                          padding_value=0, batch_first=True)
        prompt_len = int(cond_mask.sum().item())
        return {"cond": cond, "text_ids": text_ids, "step_cond": step_cond, "prompt_len": prompt_len,
                "y0": y0, "prompt_snr": prompt_snr, "seed": int(seed),
                "frames": int(y0.shape[1] - prompt_len)}   # generated frames, as the experiments' CSVs report them

    # ------------------------------------------------------------------ teacher, student, vocoder
    def _teacher_step(self, ctx, state, t_val):
        torch = self.torch
        t = torch.tensor(float(t_val), dtype=torch.float32)
        kw = dict(x=state, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
        pred = self.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
        null = self.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
        v = pred + (pred - null) * 2.0
        return v, state + (1.0 - float(t_val)) * v

    def _student(self, ctx, endpoint, taus=sd.STUDENT_TAUS):
        torch = self.torch
        x = endpoint.clone()
        for tau in taus:
            gen = torch.Generator(device="cpu")
            gen.manual_seed((int(ctx["seed"]) + int(tau * 100)) % (2 ** 31 - 1))
            eps = torch.randn(x.shape, dtype=torch.float32, generator=gen)
            tt = torch.tensor([tau], dtype=torch.float32)
            ts = tt.unsqueeze(-1).unsqueeze(-1)
            phi = (1.0 - ts) * eps + ts * x
            pred = self.tts.generator(x=phi, cond=ctx["cond"], text=ctx["text_ids"], time=tt,
                                      drop_audio_cond=False, drop_text=False)
            x = phi + (1.0 - ts) * pred
        return x

    def _decode(self, ctx, x):
        mel = x.permute(0, 2, 1) * self.tts.scale
        return self.tts.vocos.decode(mel[..., ctx["prompt_len"]:]).cpu().numpy().squeeze()

    def _run_teacher_8(self, ctx, s0_gen, emit):
        """Run Steps 0..4 of the 8-interval sway grid (t = 0 -> 0.2929) while tracking r against s_0."""
        state, r_trace, ep = ctx["y0"].clone(), {}, None
        plen = ctx["prompt_len"]
        t_teacher = 0.0
        for i in si.TEACHER_STEPS:
            t0 = time.perf_counter()
            v, ep = self._teacher_step(ctx, state, si.TEACHER_GRID[i])
            r_val = si.invariant_residual(ep[0, plen:, :], s0_gen)
            dt = time.perf_counter() - t0
            t_teacher += dt
            r_trace[i] = r_val
            emit({"phase": "teacher", "step": i, "t": si.TEACHER_GRID[i], "r": r_val,
                  "seconds": dt, "commit": i == si.COMMIT_STEP})
            if i < si.COMMIT_STEP:
                state = state + (si.TEACHER_GRID[i + 1] - si.TEACHER_GRID[i]) * v
        return ep, r_trace, t_teacher

    def _run_w8_reference(self, ctx):
        """Run the 21-pass Chapter 3 reference (16-interval Step 8 + 3 student steps) for comparison."""
        t0 = time.perf_counter()
        state, ep = ctx["y0"].clone(), None
        for i in range(si.W8_REFERENCE_STEP + 1):
            v, ep = self._teacher_step(ctx, state, sd.TEACHER_GRID[i])
            if i < si.W8_REFERENCE_STEP:
                state = state + (sd.TEACHER_GRID[i + 1] - sd.TEACHER_GRID[i]) * v
        ref_audio = self._decode(ctx, self._student(ctx, ep, sd.STUDENT_TAUS))
        return ref_audio, time.perf_counter() - t0

    def synthesize(self, prompt, sr, prompt_text, text, seed=42, lam_r=si.DEFAULT_LAMBDA_R,
                   gate_db=si.SNR_GATE_DB, compare_reference=False, progress=None):
        """Run SNR + Invariant Residual (SNR_R) on one prompt and text.

        Computes the 1-pass student draft s_0, 5 guided 8-interval teacher steps (10 passes), routes between
        teacher endpoint CPH8 and quantile handover CPQH using prompt_snr < gate_db or r > lam_r, and finishes with
        2 student restart steps (13 passes total). When compare_reference=True, also synthesizes unstabilized CPH8
        (when flagged to CPQH) and the 21-pass 16-interval Step-8 reference W8 on the same trajectory and seeds.
        """
        torch = self.torch
        emit = progress or (lambda e: None)
        t_all = time.perf_counter()
        with torch.inference_mode():
            t0 = time.perf_counter()
            ctx = self.prepare(prompt, sr, prompt_text, text, seed)
            t_prep = time.perf_counter() - t0
            gate_open = float(ctx["prompt_snr"]) >= float(gate_db)
            emit({"phase": "prepared", "prompt_snr_db": ctx["prompt_snr"], "gate_open": gate_open,
                  "gate_db": float(gate_db), "lambda_r": float(lam_r), "frames": ctx["frames"],
                  "prompt_frames": ctx["prompt_len"], "seconds": t_prep})

            t0 = time.perf_counter()
            s0 = self._student(ctx, ctx["step_cond"], [si.DRAFT_TAU])
            t_draft = time.perf_counter() - t0
            emit({"phase": "draft", "tau": si.DRAFT_TAU, "seconds": t_draft})

            plen = ctx["prompt_len"]
            s0_gen = s0[0, plen:, :]
            ep, r_trace, t_teacher = self._run_teacher_8(ctx, s0_gen, emit)

            final_r = r_trace[si.COMMIT_STEP]
            band_r = si.band_residuals(ep[0, plen:, :], s0_gen)
            summary = si.evaluate(ctx["prompt_snr"], final_r, lam_r=lam_r, gate_db=gate_db,
                                  r_trace=r_trace, band_r=band_r)
            payload = si.quantile_payload(s0, ep, plen) if summary["choice"] == "CPQH" else ep
            emit({"phase": "handover", **summary})

            t0 = time.perf_counter()
            x = self._student(ctx, payload, si.STUDENT_RESTART_TAUS)
            t_student = time.perf_counter() - t0
            emit({"phase": "student", "seconds": t_student})

            t0 = time.perf_counter()
            audio = self._decode(ctx, x)
            t_vocoder = time.perf_counter() - t0

            result = {
                **summary,
                "seed": ctx["seed"],
                "frames": ctx["frames"],
                "prompt_frames": plen,
                "audio": audio,
                "sample_rate": SAMPLE_RATE,
                "audio_seconds": len(audio) / SAMPLE_RATE,
                "timings": {
                    "prepare": t_prep,
                    "draft": t_draft,
                    "teacher": t_teacher,
                    "student": t_student,
                    "vocoder": t_vocoder,
                },
            }
            if compare_reference:
                if summary["choice"] == "CPQH":
                    t0 = time.perf_counter()
                    unstab_audio = self._decode(ctx, self._student(ctx, ep, si.STUDENT_RESTART_TAUS))
                    result["unstabilized"] = {
                        "arm": "CPH8", "passes": si.CPH8_PASSES, "audio": unstab_audio,
                        "seconds": time.perf_counter() - t0,
                    }
                ref_audio, ref_sec = self._run_w8_reference(ctx)
                result["reference"] = {
                    "arm": "W8", "step": si.W8_REFERENCE_STEP, "passes": si.W8_REFERENCE_PASSES,
                    "audio": ref_audio, "seconds": ref_sec,
                }
        result["timings"]["total"] = time.perf_counter() - t_all
        emit({"phase": "done", "seconds": result["timings"]["total"]})
        return result


def load_example(item, condition="clean"):
    """An example's prompt as the experiments see it under `condition`: (waveform, sample rate)."""
    import soundfile as sf
    audio, sr = sf.read(str(DATA_DIR / item["prompt_wav_rel"]), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return degrade_prompt(audio, sr, condition, item["input_seed"]), sr
