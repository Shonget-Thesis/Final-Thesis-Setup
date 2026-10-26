"""
Block diagrams for the two candidate solutions of Appendix E, drawn in the style of
figures/handover_scheduling_architecture.jpg (five panels, highlighted scheduler, dashed
offline-calibration box, caption under the diagram).

  tas_handover_gate.jpg       SNR + Trajectory Anomaly Score (TAS) handover gate
  swc_conditioning.jpg        SNR-Wiener Conditioning (SWC) with SNR routing

No data needed; the formulas follow Appendix E. Outputs go to paper/figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1376, 730                      # canvas in pixels at 100 dpi (the reference is 1376 x 768)
DPI = 200                             # saved at twice the reference resolution

INK = "#111111"
PANEL, PANEL_EDGE = "#efefef", "#4d4d4d"
INNER, HEADER = "#f8f8f8", "#d6d6d6"
GREEN_OUT, GREEN_EDGE, GREEN_IN, GREEN_HEAD = "#7bc88f", "#2e7d4f", "#dff1e2", "#1e8a4c"
CALIB_FILL, CALIB_EDGE = "#dde8f3", "#56708f"
SANS, SERIF = "Arial", "Times New Roman"

plt.rcParams["mathtext.fontset"] = "dejavusans"


def pt(px):
    """Pixel size at 100 dpi to points."""
    return px * 0.72


def new_canvas():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x0, y0, x1, y1, fc, ec, lw=1.6, r=12, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=1))


def text(ax, x, y, s, size=17, bold=False, color=INK, family=SANS, ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=pt(size), fontweight="bold" if bold else "normal", color=color, family=family,
            ha=ha, va=va, linespacing=1.15, style=style, zorder=3)


def lines(ax, x, y0, items, step=24):
    """items: (text, size, bold) rows centred at x, starting at y0."""
    y = y0
    for s, size, bold in items:
        if s:
            text(ax, x, y, s, size=size, bold=bold)
        y += step


def arrow(ax, x0, y0, x1, y1, color=INK, lw=2.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=18, shrinkA=0, shrinkB=0))


def panel(ax, x0, x1, title, green=False):
    box(ax, x0, 160, x1, 600, GREEN_OUT if green else PANEL, GREEN_EDGE if green else PANEL_EDGE, lw=2.0 if green else 1.6, r=14)
    text(ax, (x0 + x1) / 2, 192, title, size=19, bold=True)


def header_box(ax, x0, x1, y0, y1, title, green=False):
    box(ax, x0, y0, x1, y1, GREEN_HEAD if green else HEADER, GREEN_EDGE if green else PANEL_EDGE, lw=1.4, r=8)
    text(ax, (x0 + x1) / 2, (y0 + y1) / 2, title, size=19, bold=True, color="white" if green else INK)


def waveform(ax, x0, x1, yc, amp):
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 900)
    env = sum(a * np.exp(-((t - c) / w) ** 2) for a, c, w in ((1.0, 0.18, 0.08), (0.8, 0.42, 0.07), (1.0, 0.62, 0.09), (0.7, 0.85, 0.06)))
    sig = env * np.sin(2 * np.pi * 55 * t) * (0.55 + 0.45 * rng.random(t.size))
    ax.plot(x0 + (x1 - x0) * t, yc - amp * sig / np.abs(sig).max(), color=INK, lw=0.9, zorder=3)


def caption(ax, fig, x0, y0, width, bold, rest, size=15.5, step=24):
    """Serif caption: bold lead-in, then regular text wrapped to the width (measured with the renderer)."""
    r = fig.canvas.get_renderer()
    scale = fig.dpi / 100.0

    def wpx(s, b):
        t = ax.text(0, 0, s, fontsize=pt(size), family=SERIF, fontweight="bold" if b else "normal")
        w = t.get_window_extent(renderer=r).width / scale
        t.remove()
        return w

    lead = wpx(bold + " ", True)
    text(ax, x0, y0, bold, size=size, bold=True, family=SERIF, ha="left")
    words, line, avail, y, first = rest.split(), "", width - lead, y0, True
    for w in words:
        cand = (line + " " + w).strip()
        if wpx(cand, False) <= avail:
            line = cand
            continue
        text(ax, x0 + (lead if first else 0), y, line, size=size, family=SERIF, ha="left")
        line, avail, y, first = w, width, y + step, False
    text(ax, x0 + (lead if first else 0), y, line, size=size, family=SERIF, ha="left")


def front_end(ax, snr_note):
    panel(ax, 25, 258, "[Front-End Inputs]")
    box(ax, 40, 215, 243, 318, INNER, PANEL_EDGE, lw=1.4, r=10)
    lines(ax, 141.5, 240, [("Prompt Audio,", 19, False), ("Prompt Text,", 19, False), ("Target Text", 19, False)], step=26)
    arrow(ax, 141.5, 318, 141.5, 352)
    box(ax, 40, 352, 243, 452, HEADER, PANEL_EDGE, lw=1.4, r=10)
    lines(ax, 141.5, 378, [("Prompt SNR", 19, False), ("Estimate", 19, False), ("$\\hat{S}$", 20, False)], step=26)
    lines(ax, 141.5, 488, snr_note, step=22)


def student(ax, x0, x1, note):
    xm = (x0 + x1) / 2
    panel(ax, x0, x1, "[Phase 2: Distilled\nStudent & Vocoder]")
    box(ax, x0 + 12, 222, x1 - 12, 372, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, x0 + 12, x1 - 12, 222, 282, "Student Generator\n(DMOSpeech 2)")
    lines(ax, xm, 306, note, step=22)
    arrow(ax, xm, 372, xm, 405)
    box(ax, x0 + 22, 405, x1 - 22, 470, HEADER, PANEL_EDGE, lw=1.4, r=10)
    lines(ax, xm, 425, [("Vocos Neural", 19, False), ("Vocoder (24 kHz)", 19, False)], step=25)
    arrow(ax, xm, 470, xm, 500)
    text(ax, xm, 518, "Final Speech Waveform", size=18)
    waveform(ax, x0 + 22, x1 - 22, 562, 26)


def calibration(ax, x0, x1, label, target_x):
    box(ax, x0, 80, x1, 128, CALIB_FILL, CALIB_EDGE, lw=1.6, r=10, ls=(0, (4, 3)))
    text(ax, (x0 + x1) / 2, 104, label, size=17)
    arrow(ax, target_x, 128, target_x, 160, color=CALIB_EDGE, lw=1.8)


def figure_tas():
    fig, ax = new_canvas()
    text(ax, W / 2, 40, "SNR + Trajectory Anomaly Score (TAS) Handover Gate", size=27, bold=True)
    calibration(ax, 300, 1076, "Offline: clean-trajectory model $(\\mu,\\Sigma)$ + Learn-Then-Test $\\rightarrow$ certified threshold $\\hat{\\kappa}$", 684)

    front_end(ax, [("Sent to the", 16, False), ("scheduler for the", 16, False), ("SNR gate", 16, False)])

    panel(ax, 285, 521, "[Phase 1: Teacher\nFlow-Matching]")
    box(ax, 297, 222, 509, 590, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, 297, 509, 222, 282, "Teacher Generator\n(F5-TTS)")
    lines(ax, 403, 310, [("Iterative Euler steps", 17, False), ("from $t=0$", 17, False), ("", 0, False),
                         ("Endpoint estimates", 17, False), ("$\\hat{x}_1(t_i)$", 18, False), ("", 0, False),
                         ("Drift profile", 17, True), ("$\\mathbf{z}=(\\log d_1,\\ldots,\\log d_6)$", 16, False),
                         ("(no extra forward pass)", 15, False)], step=26)

    panel(ax, 557, 811, "[Adaptive Handover\nScheduler]", green=True)
    box(ax, 569, 222, 799, 590, GREEN_IN, GREEN_EDGE, lw=1.6, r=10)
    header_box(ax, 569, 799, 222, 280, "SNR + TAS Gate", green=True)
    lines(ax, 684, 302, [("Prompt-SNR Gate:", 17, True), ("$\\hat{S}\\geq\\theta$", 18, False),
                         ("Trajectory Anomaly Score:", 16, True), ("$A=(\\mathbf{z}-\\mu)^{\\top}\\Sigma^{-1}(\\mathbf{z}-\\mu)$", 16, False),
                         ("if the trajectory is stiller", 15, False), ("than on clean prompts, else 0", 15, False),
                         ("Online Decision Rule:", 17, True), ("If $\\hat{S}\\geq\\theta$ and $A\\leq\\hat{\\kappa}$:", 16, False),
                         ("hand over at Step 4.", 16, False), ("Otherwise: Step 8.", 16, False)], step=27)

    panel(ax, 838, 1058, "[Handover Interface]")
    box(ax, 850, 222, 1046, 590, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, 850, 1046, 222, 282, "Handover\nInterface")
    lines(ax, 948, 312, [("Teacher endpoint", 17, False), ("$T=\\hat{x}_1(t_k)$", 18, False), ("$k\\in\\{4,8\\}$", 17, False), ("", 0, False),
                         ("Re-noise at fixed", 17, False), ("student restart level", 17, False), ("$j=1$  ($\\tau=0.0761$)", 17, False), ("", 0, False),
                         ("Step 4: 13 NFE", 16, True), ("Step 8: 21 NFE", 16, True)], step=26)

    student(ax, 1098, 1350, [("Executes exactly 3", 16, False), ("consistency steps", 16, False), ("to $t=1.00$", 16, False)])

    for x0, x1 in ((258, 285), (521, 557), (811, 838), (1058, 1098)):
        arrow(ax, x0, 380, x1, 380)

    caption(ax, fig, 22, 646, 1330, "Figure E.1: SNR + Trajectory Anomaly Score (TAS).",
            "The teacher's drift profile, which the teacher pass already computes, is compared with the profiles of clean "
            "prompts. A trajectory that settles too early (premature stabilization, the sign of a noisy prompt) is not handed "
            "over, even if its SNR passes the gate. Inputs that pass both checks are handed to the student at Step 4 (13 forward "
            "passes); all others continue to Step 8 (21). The clean-trajectory model needs no failure labels; the threshold "
            "is certified by Learn-Then-Test.")
    return fig


def figure_swc():
    fig, ax = new_canvas()
    text(ax, W / 2, 40, "SNR-Wiener Conditioning (SWC): Cleaning the Prompt Before an Early Handover", size=27, bold=True)
    calibration(ax, 186, 740, "Offline: Learn-Then-Test certifies the routing thresholds $\\theta_{hi}$ and $\\theta_{e}$", 403)

    front_end(ax, [("Sets the route and", 16, False), ("the cleaning", 16, False), ("strength", 16, False)])

    panel(ax, 285, 521, "[SNR Routing]")
    box(ax, 297, 222, 509, 590, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, 297, 509, 222, 282, "Route by\nPrompt SNR")
    lines(ax, 403, 310, [("$\\hat{S}\\geq\\theta_{hi}$ (clean):", 17, True), ("plain prompt, Step 4", 17, False), ("", 0, False),
                         ("$\\hat{S}<\\theta_{e}$ (heavy noise):", 17, True), ("cleaned prompt, Step 4", 17, False), ("", 0, False),
                         ("otherwise:", 17, True), ("full teacher, Step 8", 17, False), ("", 0, False),
                         ("13 / 13 / 21 NFE", 16, False)], step=26)

    panel(ax, 557, 811, "[Conditioning Repair]", green=True)
    box(ax, 569, 222, 799, 590, GREEN_IN, GREEN_EDGE, lw=1.6, r=10)
    header_box(ax, 569, 799, 222, 280, "SNR-Wiener\nConditioning", green=True)
    lines(ax, 684, 302, [("Noise floor per mel band:", 16, True), ("$N_b$ = mean of the quietest", 15, False), ("20% of prompt frames", 15, False),
                         ("SNR-set strength:", 16, True), ("$\\beta(\\hat{S})=\\min\\{1,\\max\\{0,\\frac{30-\\hat{S}}{10}\\}\\}$", 14, False),
                         ("Gain on each prompt frame:", 16, True), ("$G=\\min\\{1,\\max\\{0.1,1-\\beta\\,\\frac{N_b}{m}\\}\\}$", 14, False),
                         ("applied to the teacher's and", 15, False), ("the student's conditioning", 15, False),
                         ("(no extra network)", 15, False)], step=28)

    panel(ax, 838, 1058, "[Phase 1: Teacher\nFlow-Matching]")
    box(ax, 850, 222, 1046, 590, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, 850, 1046, 222, 282, "Teacher Generator\n(F5-TTS)")
    lines(ax, 948, 312, [("Runs on the cleaned", 17, False), ("conditioning", 17, False), ("", 0, False),
                         ("Euler steps to Step 4", 17, False), ("(or Step 8 when routed)", 15, False), ("", 0, False),
                         ("Endpoint $T=\\hat{x}_1(t_k)$", 17, False), ("re-noised at $j=1$", 17, False), ("($\\tau=0.0761$)", 17, False)], step=26)

    student(ax, 1098, 1350, [("3 consistency steps on", 16, False), ("the same cleaned", 16, False), ("conditioning", 16, False)])

    for x0, x1 in ((258, 285), (521, 557), (811, 838), (1058, 1098)):
        arrow(ax, x0, 380, x1, 380)

    caption(ax, fig, 22, 646, 1330, "Figure E.2: SNR-Wiener Conditioning (SWC).",
            "The prompt's noise floor is estimated from its quietest frames and removed from the prompt frames of the "
            "teacher's and the student's conditioning, with a strength set by the prompt SNR (none at 30 dB or above, full at "
            "20 dB or below). The model then no longer imitates the noise, so a heavily noisy prompt can be handed to the "
            "student at Step 4 (13 forward passes) instead of running the full teacher (21). The routing thresholds are "
            "certified by Learn-Then-Test; no network is added.")
    return fig


if __name__ == "__main__":
    for name, make in (("tas_handover_gate.jpg", figure_tas), ("swc_conditioning.jpg", figure_swc)):
        fig = make()
        path = os.path.join(OUTPUT_DIR, name)
        fig.savefig(path, dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
        plt.close(fig)
        print("wrote", path)
