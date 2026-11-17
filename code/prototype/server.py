"""Local web server for the SNR + drift prototype.

    .venv\\Scripts\\python code\\prototype\\server.py            (then open http://127.0.0.1:8765)

The page (web/index.html) opens at once; the model loads in the background. Jobs run one at a time on the CPU, and each
teacher step is reported to the page as it is computed.
"""
import argparse
import base64
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
import snr_drift as sd

WEB = Path(__file__).resolve().parent / "web"
app = FastAPI(title="SNR + drift handover prototype")
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
EXP031_QUALITY = eng.ROOT / "research/experiments/EXP-031-snr-toh-method-comparison/results/quality.csv"


def _recorded():
    """EXP-031's T4 records for development speakers 1-30 (prompt SNR, drifts and the rule's step at both presets), so
    the page can show the recorded decision beside the local one."""
    if not RECORDED and EXP031_QUALITY.exists():
        import csv
        with open(EXP031_QUALITY, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                RECORDED.setdefault(r["utt_id"], {})[r["condition"]] = {
                    "prompt_snr": float(r["prompt_snr"]), "drifts": {str(j): float(r[f"d_{j}"]) for j in range(1, 9)},
                    "step": {"0.35": int(float(r["ch3_step"])), "1.20": int(float(r["ch3_ltt_step"]))}}
    return RECORDED


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
    return {"ready": e is not None, "loading": STATE["loading"], "error": STATE["error"],
            "threads": getattr(e, "threads", None), "load_seconds": getattr(e, "load_seconds", None),
            "missing": eng.missing_files(), "examples": len(_examples()), "conditions": list(eng.OFFLINE_CONDITIONS),
            "lambda_presets": sd.LAMBDA_PRESETS, "gate_db": sd.SNR_GATE_DB, "candidates": list(sd.CANDIDATE_STEPS),
            "reference_step": sd.REFERENCE_STEP, "reference_passes": sd.REFERENCE_PASSES,
            "teacher_grid": sd.TEACHER_GRID, "student_taus": sd.STUDENT_TAUS}


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
            res = STATE["engine"].synthesize(args["prompt"], args["sr"], args["prompt_text"], args["text"],
                                             seed=args["seed"], lam=args["lam"], compare_reference=args["compare"],
                                             progress=lambda e: job["events"].append(e))
        audio = {"output": eng.wav_bytes(res.pop("audio"))}
        ref = res.get("reference")
        if ref and "audio" in ref:
            audio["reference"] = eng.wav_bytes(ref.pop("audio"))
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
                     prompt_text: str = Form(""), text: str = Form(...), lam: float = Form(sd.DEFAULT_LAMBDA),
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
    JOBS[job_id] = {"status": "queued", "events": [], "label": label, "created": time.time(),
                    "inputs": {"prompt_text": prompt_text, "text": text, "lambda": lam, "seed": seed,
                               "compare": compare, "source": source, "condition": condition if source == "example" else None}}
    args = {"prompt": prompt, "sr": sr, "prompt_text": prompt_text, "text": text, "lam": lam, "seed": seed,
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
                    headers={"Content-Disposition": f'attachment; filename="snr_drift_{job_id}.wav"'})


def main():
    p = argparse.ArgumentParser(description="SNR + drift handover prototype (local web interface)")
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
