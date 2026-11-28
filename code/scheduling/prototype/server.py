"""Local web server for the SNR + Invariant Residual (SNR_R) prototype.

    .venv\\Scripts\\python code\\scheduling\\prototype\\server.py            (then open http://127.0.0.1:8765)

The page (web/index.html) opens at once; the model loads in the background. Jobs run one at a time on the CPU, and each
step (student draft, 8-interval teacher steps, invariant residual decision, student completion) is streamed live.
"""
import argparse
import io
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

import engine as eng
import snr_invariant as si

WEB = Path(__file__).resolve().parent / "web"
app = FastAPI(title="SNR + Invariant Residual prototype")
STATE = {"engine": None, "error": None, "loading": False, "started": time.time()}
JOBS = {}
LOCK = threading.Lock()          # one synthesis at a time: the model is not shared between threads
EXAMPLES = {}


def _load_engine(threads):
    STATE["loading"] = True
    try:
        STATE["engine"] = eng.Engine(threads=threads)
    except Exception as exc:  # shown on the page instead of crashing the server
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        STATE["loading"] = False


RECORDED = {}
RISK_CAL = None
EXP034_QUALITY = eng.ROOT / "research/experiments/EXP-034-invariant-residual/results/quality.csv"


def _exp034_rows():
    if not EXP034_QUALITY.exists():
        return []
    import csv
    with open(EXP034_QUALITY, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _recorded():
    """EXP-034's T4 records for development speakers 31-60 (prompt SNR, invariant residual r, SNR_R choice, UTMOS),
    so the page can show the recorded decision beside the local CPU run."""
    if not RECORDED:
        for r in _exp034_rows():
            RECORDED.setdefault(r["utt_id"], {})[r["condition"]] = {
                "prompt_snr": float(r["prompt_snr"]),
                "r": float(r["r"]),
                "choice": r["choice_SNR_R"],
                "utmos": {"CPH8": float(r["utmos_CPH8"]), "CPQH": float(r["utmos_CPQH"])},
            }
    return RECORDED


def _risk_calibration():
    """Run fixed-sequence binomial Learn-Then-Test calibration on EXP-034 dev rows for status reporting."""
    global RISK_CAL
    if RISK_CAL is None:
        rows = _exp034_rows()
        RISK_CAL = (
            {
                "source": "EXP-034 dev speakers 31-60 (n=240, clean=30)",
                "ltt_15": si.calibrate_ltt(rows, alpha=0.15, delta=0.10),
                "ltt_10": si.calibrate_ltt(rows, alpha=0.10, delta=0.10),
            }
            if rows
            else {}
        )
    return RISK_CAL


def _examples():
    if not EXAMPLES:
        try:
            for it in eng.development_examples():
                if (eng.DATA_DIR / it["prompt_wav_rel"]).exists():
                    EXAMPLES[it["utt_id"]] = it
        except Exception:
            traceback.print_exc()
    return EXAMPLES


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status():
    e = STATE["engine"]
    return {
        "method": "snr_invariant",
        "ready": e is not None,
        "loading": STATE["loading"],
        "error": STATE["error"],
        "threads": getattr(e, "threads", None),
        "load_seconds": getattr(e, "load_seconds", None),
        "missing": eng.missing_files(),
        "examples": len(_examples()),
        "conditions": list(eng.OFFLINE_CONDITIONS),
        "lambda_presets": si.LAMBDA_R_PRESETS,
        "default_lambda_r": si.DEFAULT_LAMBDA_R,
        "risk_calibration": _risk_calibration(),
        "gate_db": si.SNR_GATE_DB,
        "commit_step": si.COMMIT_STEP,
        "passes": si.METHOD_PASSES,
        "teacher_passes": si.TEACHER_PASSES,
        "draft_passes": si.DRAFT_PASSES,
        "student_passes": si.STUDENT_PASSES,
        "reference_passes": si.W8_REFERENCE_PASSES,
        "paper_passes": si.P25_PAPER_PASSES,
        "teacher_grid": si.TEACHER_GRID[:si.COMMIT_STEP + 1],
        "draft_tau": si.DRAFT_TAU,
        "student_taus": si.STUDENT_RESTART_TAUS,
    }


@app.get("/api/examples")
def examples():
    rec = _recorded()
    return [{"utt_id": u, "speaker_id": it["speaker_id"], "prompt_text": it["prompt_text"], "gt_text": it["gt_text"],
             "input_seed": it["input_seed"], "recorded": rec.get(u, {})} for u, it in _examples().items()]


@app.get("/api/example_audio")
def example_audio(utt_id: str, condition: str = "clean"):
    it = _examples().get(utt_id)
    if it is None:
        raise HTTPException(404, "unknown example")
    try:
        audio, sr = eng.load_example(it, condition)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return Response(eng.wav_bytes(audio, sr), media_type="audio/wav")


def _read_upload(data):
    import soundfile as sf
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(400, f"Could not read the audio file ({exc}). Use WAV or FLAC.")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < sr * 1.0:
        raise HTTPException(400, "The prompt is shorter than 1 second.")
    return audio.astype(np.float32), sr


def _run(job_id, args):
    job = JOBS[job_id]
    try:
        with LOCK:
            job["status"] = "running"
            job["started"] = time.time()
            res = STATE["engine"].synthesize(
                args["prompt"], args["sr"], args["prompt_text"], args["text"],
                seed=args["seed"], lam_r=args["lam_r"], compare_reference=args["compare"],
                progress=lambda e: job["events"].append(e),
            )
        audio = {"output": eng.wav_bytes(res.pop("audio"))}
        for key in ("unstabilized", "reference"):
            branch = res.get(key)
            if branch and "audio" in branch:
                audio[key] = eng.wav_bytes(branch.pop("audio"))
        audio["prompt"] = eng.wav_bytes(args["prompt"], args["sr"])
        job["audio"] = audio
        job["result"] = res
        job["status"] = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()


@app.post("/api/jobs")
async def create_job(source: str = Form("example"), utt_id: str = Form(""), condition: str = Form("clean"),
                     prompt_text: str = Form(""), text: str = Form(...), lam_r: float = Form(si.DEFAULT_LAMBDA_R),
                     seed: int = Form(-1), compare: bool = Form(False), file: UploadFile = File(None)):
    if STATE["engine"] is None:
        raise HTTPException(503, STATE["error"] or "The model is still loading.")
    text = text.strip()
    if not text:
        raise HTTPException(400, "Type the text to speak.")
    if source == "example":
        it = _examples().get(utt_id)
        if it is None:
            raise HTTPException(404, "unknown example")
        try:
            prompt, sr = eng.load_example(it, condition)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        prompt_text = prompt_text.strip() or it["prompt_text"]
        seed = it["input_seed"] if seed < 0 else seed
        label = f"{utt_id} ({condition})"
    else:
        if file is None:
            raise HTTPException(400, "Upload a WAV prompt.")
        prompt, sr = _read_upload(await file.read())
        if not prompt_text.strip():
            raise HTTPException(400, "Type what the prompt says (its transcript).")
        seed = 42 if seed < 0 else seed
        label = file.filename or "uploaded prompt"
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "queued", "events": [], "label": label, "created": time.time(),
        "inputs": {"prompt_text": prompt_text, "text": text, "lambda_r": lam_r, "seed": seed,
                   "compare": compare, "source": source, "condition": condition if source == "example" else None},
    }
    args = {"prompt": prompt, "sr": sr, "prompt_text": prompt_text, "text": text, "lam_r": lam_r, "seed": seed,
            "compare": compare}
    threading.Thread(target=_run, args=(job_id, args), daemon=True).start()
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str, since: int = 0):
    j = JOBS.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return {"status": j["status"], "label": j["label"], "inputs": j["inputs"], "events": j["events"][since:],
            "n_events": len(j["events"]), "result": j.get("result"), "error": j.get("error"),
            "audio": sorted(j.get("audio", {}))}


@app.get("/api/jobs/{job_id}/audio/{which}")
def job_audio(job_id: str, which: str):
    j = JOBS.get(job_id)
    if j is None or which not in j.get("audio", {}):
        raise HTTPException(404, "no such audio")
    return Response(j["audio"][which], media_type="audio/wav")


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    j = JOBS.get(job_id)
    if j is None or "output" not in j.get("audio", {}):
        raise HTTPException(404, "no output yet")
    return Response(j["audio"]["output"], media_type="audio/wav",
                    headers={"Content-Disposition": f'attachment; filename="snr_invariant_{job_id}.wav"'})


def main():
    p = argparse.ArgumentParser(description="SNR + Invariant Residual handover prototype (local web interface)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch (default: all but one)")
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    threading.Thread(target=_load_engine, args=(a.threads,), daemon=True).start()
    url = f"http://{a.host}:{a.port}"
    print(f"[server] {url}  (the model loads in the background; the page shows its progress)")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
