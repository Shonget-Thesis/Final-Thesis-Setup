"""Terminal mode of the SNR + Invariant Residual (SNR_R) prototype: prints steps live and saves the output WAV.

    .venv\\Scripts\\python code\\scheduling\\prototype\\cli.py --example 38 --condition gauss25 --compare
    .venv\\Scripts\\python code\\scheduling\\prototype\\cli.py --prompt my_voice.wav --prompt-text "What the prompt says." \\
        --text "What the voice should say." --out out.wav
"""
import argparse
import sys
from pathlib import Path

import engine as eng
import snr_invariant as si


def main(argv=None):
    p = argparse.ArgumentParser(description="SNR + Invariant Residual (SNR_R) prototype (terminal)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--example", help="development example: 1-based number (1-100) or utterance ID")
    src.add_argument("--prompt", help="your own prompt WAV/FLAC")
    p.add_argument("--condition", default="clean", choices=eng.OFFLINE_CONDITIONS,
                   help="degradation applied to an example prompt (matching EXP-034)")
    p.add_argument("--prompt-text", default="", help="the prompt's transcript (needed with --prompt)")
    p.add_argument("--text", default="", help="text to speak (default for an example: its target sentence)")
    p.add_argument("--preset", default="ltt_15", choices=tuple(si.LAMBDA_R_PRESETS.keys()),
                   help="risk-calibrated preset threshold (default: ltt_15; overridden by --lambda-r)")
    p.add_argument("--lambda-r", "--lambda", dest="lam_r", type=float, default=None,
                   help="explicit invariant residual threshold (overrides --preset)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--compare", action="store_true",
                   help="also synthesize unstabilized CPH8 (when flagged to CPQH) and the 21-pass W8 reference")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", default="snr_invariant_output.wav")
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
            print(f"  prompt SNR {e['prompt_snr_db']:.1f} dB (gate {e['gate_db']:.0f} dB: {gate}); "
                  f"lambda_r = {e['lambda_r']:.3f}; {e['frames']} frames to generate")
        elif e["phase"] == "draft":
            print(f"  draft s0 (tau={e['tau']:.1f}, 1 pass): {e['seconds']:5.1f} s")
        elif e["phase"] == "teacher":
            mark = "  <- commit (t=0.2929)" if e["commit"] else ""
            print(f"  teacher step {e['step']}  t = {e['t']:.4f}  r = {e['r']:6.3f}  {e['seconds']:5.1f} s{mark}")
        elif e["phase"] == "student":
            print(f"  student completion (2 restart steps): {e['seconds']:.1f} s")

    lam_r = si.LAMBDA_R_PRESETS[a.preset]["value"] if a.lam_r is None else a.lam_r
    engine = eng.Engine(threads=a.threads)
    res = engine.synthesize(prompt, sr, prompt_text, text, seed=seed, lam_r=lam_r,
                            compare_reference=a.compare, progress=show)
    out = Path(a.out)
    out.write_bytes(eng.wav_bytes(res["audio"]))
    print(f"\nDecision: {res['choice']} (r = {res['residual_r']:.3f}; {res['reason']})")
    print(f"Risk calibration: {res['risk_certificate']}")
    print(f"Cost: {res['passes']} of {res['reference_passes']} W8 passes ({res['saving']:.0%} saved; "
          f"{res['saving_vs_paper']:.0%} vs 30-pass P25). Time {res['timings']['total']:.1f} s "
          f"for {res['audio_seconds']:.1f} s of speech. Saved {out}")
    unstab = res.get("unstabilized")
    if unstab and "audio" in unstab:
        up = out.with_name(out.stem + "_unstabilized_cph8" + out.suffix)
        up.write_bytes(eng.wav_bytes(unstab["audio"]))
        print(f"Unstabilized CPH8 ({unstab['passes']} passes) saved {up}")
    ref = res.get("reference")
    if ref and "audio" in ref:
        rp = out.with_name(out.stem + "_reference_w8" + out.suffix)
        rp.write_bytes(eng.wav_bytes(ref["audio"]))
        print(f"Reference W8 ({ref['passes']} passes) saved {rp}")


if __name__ == "__main__":
    main()
