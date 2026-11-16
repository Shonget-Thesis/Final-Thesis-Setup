"""Local DMOSpeech 2 engine for the SNR + drift prototype (CPU by default; a demonstration, not an evaluation tool).

It reproduces the thesis pipeline (code/colab/run_followup_tests.Pipeline.prepare and code/scheduling/sampler.py):
prompt SNR on the prompt as given, RMS normalisation to 0.3, duration predictor, guided Euler teacher on the 16-interval
sway grid with CFG strength 2 and no stochastic term, endpoint estimates and drift over generated frames, and the
student's three restart levels with per-level seeds. The only difference is that the teacher stops as soon as the rule
in snr_drift.py says so, so the saved passes are really not computed.

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
OFFLINE_CONDITIONS = ("clean", "gauss25", "noisy15", "reverb03", "reverb09", "telephone")   # babble needs DEMAND
SAMPLE_RATE = 24000

for p in (HERE, VENDOR_SRC, COLAB_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import snr_drift as sd  # noqa: E402


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
    """The experiments' prompt degradations (identical arithmetic; the noise seed is the utterance's input seed)."""
    import run_followup_tests as rf
    if condition not in OFFLINE_CONDITIONS:
        raise ValueError(f"Unknown or unavailable condition {condition!r}; choose one of {OFFLINE_CONDITIONS}")
    # The dtype is kept as rf.degrade returns it (white noise gives float64), as in the experiments, where the prompt
    # SNR is measured on that array and the model receives it as float32.
    return rf.degrade(np.asarray(prompt, dtype=np.float32), sr, condition, seed)


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
    def _teacher_step(self, ctx, state, i):
        torch = self.torch
        t = torch.tensor(sd.TEACHER_GRID[i], dtype=torch.float32)
        kw = dict(x=state, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
        pred = self.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
        null = self.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
        v = pred + (pred - null) * 2.0
        return v, state + (1.0 - sd.TEACHER_GRID[i]) * v

    def _student(self, ctx, endpoint):
        torch = self.torch
        x = endpoint.clone()
        for tau in sd.STUDENT_TAUS:
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

    def synthesize(self, prompt, sr, prompt_text, text, seed=42, lam=sd.DEFAULT_LAMBDA, gate_db=sd.SNR_GATE_DB,
                   compare_reference=False, progress=None):
        """Run SNR + drift on one prompt and text. Returns the decision trace, timings and the audio. With
        compare_reference, the teacher also continues to Step 8 and the fixed reference is synthesized from the same
        trajectory and seeds, for listening side by side (its extra passes are reported separately)."""
        torch = self.torch
        emit = progress or (lambda e: None)
        t_all = time.perf_counter()
        with torch.inference_mode():
            t0 = time.perf_counter()
            ctx = self.prepare(prompt, sr, prompt_text, text, seed)
            t_prep = time.perf_counter() - t0
            mon = sd.HandoverMonitor(ctx["prompt_snr"], lam=lam, gate_db=gate_db)
            emit({"phase": "prepared", "prompt_snr_db": ctx["prompt_snr"], "gate_open": mon.gate_open,
                  "gate_db": gate_db, "lambda": lam, "frames": ctx["frames"], "prompt_frames": ctx["prompt_len"],
                  "seconds": t_prep})
            state, prev, handover, ref_endpoint = ctx["y0"].clone(), None, None, None
            t_teacher = t_extra = 0.0
            last = sd.REFERENCE_STEP if compare_reference else None
            for i in range(sd.REFERENCE_STEP + 1):
                t0 = time.perf_counter()
                v, ep = self._teacher_step(ctx, state, i)
                d = sd.stabilization_drift(ep, prev, ctx["prompt_len"]) if prev is not None else None
                dt = time.perf_counter() - t0
                if handover is None:
                    t_teacher += dt
                else:
                    t_extra += dt
                stop = handover is None and mon.observe(i, d)
                emit({"phase": "teacher", "step": i, "t": sd.TEACHER_GRID[i], "drift": d, "seconds": dt,
                      "after_handover": handover is not None,
                      "candidate": i in sd.CANDIDATE_STEPS and mon.gate_open, "stop": bool(stop)})
                if stop:
                    handover = (i, ep)
                    emit({"phase": "handover", **mon.summary()})
                if i == sd.REFERENCE_STEP:
                    ref_endpoint = ep
                if handover is not None and last is None:
                    break
                state, prev = state + (sd.TEACHER_GRID[i + 1] - sd.TEACHER_GRID[i]) * v, ep
            t0 = time.perf_counter()
            x = self._student(ctx, handover[1])
            t_student = time.perf_counter() - t0
            emit({"phase": "student", "seconds": t_student})
            t0 = time.perf_counter()
            audio = self._decode(ctx, x)
            t_vocoder = time.perf_counter() - t0
            result = {**mon.summary(), "seed": ctx["seed"], "frames": ctx["frames"], "prompt_frames": ctx["prompt_len"],
                      "audio": audio, "sample_rate": SAMPLE_RATE, "audio_seconds": len(audio) / SAMPLE_RATE,
                      "timings": {"prepare": t_prep, "teacher": t_teacher, "student": t_student, "vocoder": t_vocoder}}
            if compare_reference and handover[0] != sd.REFERENCE_STEP:
                t0 = time.perf_counter()
                ref_audio = self._decode(ctx, self._student(ctx, ref_endpoint))
                result["reference"] = {"step": sd.REFERENCE_STEP, "passes": sd.REFERENCE_PASSES, "audio": ref_audio,
                                       "seconds": t_extra + time.perf_counter() - t0}
            elif compare_reference:
                result["reference"] = {"step": sd.REFERENCE_STEP, "passes": sd.REFERENCE_PASSES, "same_as_output": True}
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
