"""Terminal mode of the SNR + drift prototype: one sentence, the decision printed step by step, the audio saved.

    .venv\\Scripts\\python code\\prototype\\cli.py --example 1 --condition noisy15
    .venv\\Scripts\\python code\\prototype\\cli.py --prompt my_voice.wav --prompt-text "What the prompt says." \\
        --text "What the voice should say." --out out.wav
"""
import argparse
import sys
from pathlib import Path

import engine as eng
import snr_drift as sd


def main(argv=None):
    p = argparse.ArgumentParser(description="SNR + drift handover prototype (terminal)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--example", help="development example: a number from 1 or an utterance ID")
    src.add_argument("--prompt", help="your own prompt WAV")
    p.add_argument("--condition", default="clean", choices=eng.OFFLINE_CONDITIONS,
                   help="degradation applied to an example prompt (as in the experiments)")
    p.add_argument("--prompt-text", default="", help="the prompt's transcript (needed with --prompt)")
    p.add_argument("--text", default="", help="text to speak (default for an example: its target sentence)")
    p.add_argument("--lambda", dest="lam", type=float, default=sd.DEFAULT_LAMBDA)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--compare", action="store_true", help="also synthesize the fixed reference (Step 8)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", default="snr_drift_output.wav")
    a = p.parse_args(argv)

    if a.example:
        items = eng.development_examples()
        it = (items[int(a.example) - 1] if a.example.isdigit()
              else next((x for x in items if x["utt_id"] == a.example), None))
        if it is None:
            sys.exit(f"Unknown example {a.example!r}")
        prompt, sr = eng.load_example(it, a.condition)
        prompt_text, text = a.prompt_text or it["prompt_text"], a.text or it["gt_text"]
        seed = it["input_seed"] if a.seed is None else a.seed
        print(f"Example {it['utt_id']} ({a.condition})")
    else:
        import soundfile as sf
        prompt, sr = sf.read(a.prompt, dtype="float32", always_2d=False)
        if not a.prompt_text or not a.text:
            sys.exit("--prompt needs --prompt-text and --text")
        prompt_text, text, seed = a.prompt_text, a.text, 42 if a.seed is None else a.seed

    def show(e):
        if e["phase"] == "prepared":
            gate = "open" if e["gate_open"] else "CLOSED"
            print(f"  prompt SNR {e['prompt_snr_db']:.1f} dB (gate {e['gate_db']:.0f} dB: {gate}); lambda = {e['lambda']:.2f}; "
                  f"{e['frames']} frames to generate")
        elif e["phase"] == "teacher":
            d = "      -" if e["drift"] is None else f"{e['drift']:7.3f}"
            mark = "  <- handover" if e["stop"] else ("  candidate" if e["candidate"] else "")
            extra = "  (reference only)" if e["after_handover"] else ""
            print(f"  teacher step {e['step']}  t = {e['t']:.4f}  d = {d}  {e['seconds']:5.1f} s{mark}{extra}")
        elif e["phase"] == "student":
            print(f"  student: 3 steps, {e['seconds']:.1f} s")

    engine = eng.Engine(threads=a.threads)
    res = engine.synthesize(prompt, sr, prompt_text, text, seed=seed, lam=a.lam, compare_reference=a.compare,
                            progress=show)
    out = Path(a.out)
    out.write_bytes(eng.wav_bytes(res["audio"]))
    print(f"\nDecision: hand over at Step {res['handover_step']} ({res['reason']})")
    print(f"Cost: {res['passes']} of {res['reference_passes']} passes ({res['saving']:.0%} saved). "
          f"Time {res['timings']['total']:.1f} s for {res['audio_seconds']:.1f} s of speech. Saved {out}")
    ref = res.get("reference")
    if ref and "audio" in ref:
        rp = out.with_name(out.stem + "_reference_step8" + out.suffix)
        rp.write_bytes(eng.wav_bytes(ref["audio"]))
        print(f"Reference (Step 8, {ref['passes']} passes) saved {rp}")


if __name__ == "__main__":
    main()
