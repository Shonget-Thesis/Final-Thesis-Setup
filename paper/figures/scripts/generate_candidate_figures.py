"""
Block diagrams for the two candidate solutions of Appendix E, drawn in the style of
figures/handover_scheduling_architecture.jpg (five panels, highlighted method panel, dashed
offline-calibration box, caption under the diagram).

  tas_handover_gate.jpg       SNR + Trajectory Anomaly Score (TAS) handover gate
  swc_conditioning.jpg        SNR-Wiener Conditioning (SWC) with SNR routing

All text is typeset by LaTeX (needs latex and dvipng on PATH): formulas use the manuscript's
Palatino math (mathpazo) and the exact notation of Appendix E; labels use Helvetica, close to
the reference figure's Arial. No data needed. Outputs go to paper/figures/.
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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.serif": ["Palatino"],
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{bm}",
})


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


def text(ax, x, y, s, size=17, bold=False, color=INK, ha="center", va="center"):
    if bold:
        s = "\n".join(r"\textbf{" + part + "}" for part in s.split("\n"))
    ax.text(x, y, s, fontsize=pt(size), color=color, ha=ha, va=va, linespacing=1.15, zorder=3)


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


def caption(ax, x0, y0, width, lead, rest, size=15.5):
    """Serif (Palatino) caption with a bold lead-in, wrapped by LaTeX itself."""
    s = r"\parbox[t]{%.2fin}{\rmfamily\raggedright\textbf{%s} %s}" % (width / 100.0, lead, rest)
    ax.text(x0, y0, s, fontsize=pt(size), color=INK, ha="left", va="top", zorder=3)


def front_end(ax, snr_note):
    panel(ax, 25, 258, "[Front-End Inputs]")
    box(ax, 40, 215, 243, 318, INNER, PANEL_EDGE, lw=1.4, r=10)
    lines(ax, 141.5, 240, [("Prompt Audio,", 19, False), ("Prompt Text,", 19, False), ("Target Text", 19, False)], step=26)
    arrow(ax, 141.5, 318, 141.5, 352)
    box(ax, 40, 352, 243, 452, HEADER, PANEL_EDGE, lw=1.4, r=10)
    lines(ax, 141.5, 378, [("Prompt SNR", 19, False), ("Estimate", 19, False), (r"$\hat S$", 20, False)], step=26)
    lines(ax, 141.5, 488, snr_note, step=22)


def student(ax, x0, x1, note):
    xm = (x0 + x1) / 2
    panel(ax, x0, x1, "[Phase 2: Distilled\nStudent \\& Vocoder]")
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
    calibration(ax, 300, 1076, r"Offline: clean-trajectory model $(\bm{\mu},\bm{\Sigma})$ + Learn-Then-Test $\rightarrow$ certified threshold $\hat\kappa$", 684)

    front_end(ax, [("Sent to the", 16, False), ("scheduler for the", 16, False), ("SNR gate", 16, False)])

    panel(ax, 285, 521, "[Phase 1: Teacher\nFlow-Matching]")
    box(ax, 297, 222, 509, 590, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, 297, 509, 222, 282, "Teacher Generator\n(F5-TTS)")
    lines(ax, 403, 310, [("Iterative Euler steps", 17, False), (r"from $t=0$", 17, False), ("", 0, False),
                         ("Endpoint estimates", 17, False), (r"$\hat x_1(t_i)$", 18, False), ("", 0, False),
                         ("Drift profile", 17, True), (r"$\mathbf{z}(x)=(\log d_1,\dots,\log d_6)$", 16, False),
                         ("(no extra forward pass)", 15, False)], step=26)

    panel(ax, 557, 811, "[Adaptive Handover\nScheduler]", green=True)
    box(ax, 569, 222, 799, 590, GREEN_IN, GREEN_EDGE, lw=1.6, r=10)
    header_box(ax, 569, 799, 222, 280, "SNR + TAS Gate", green=True)
    lines(ax, 684, 302, [("Prompt-SNR Gate:", 17, True), (r"$\hat S\ge\theta$", 18, False),
                         ("Trajectory Anomaly Score:", 16, True),
                         (r"$A(x)=(\mathbf{z}-\bm{\mu})^{\top}\bm{\Sigma}^{-1}(\mathbf{z}-\bm{\mu})$", 16, False),
                         (r"if the mean component of", 15, False), (r"$\mathbf{z}-\bm{\mu}$ is negative, else $0$", 15, False),
                         ("Online Decision Rule:", 17, True),
                         (r"$\pi_{\mathrm{TAS}}(x)=4$ if $\hat S\ge\theta$", 16, False),
                         (r"and $A(x)\le\hat\kappa$;", 16, False),
                         (r"otherwise $k_{\text{ref}}$ (Step 8)", 16, False)], step=27)

    panel(ax, 838, 1058, "[Handover Interface]")
    box(ax, 850, 222, 1046, 590, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, 850, 1046, 222, 282, "Handover\nInterface")
    lines(ax, 948, 312, [("Teacher endpoint", 17, False), (r"$T=\hat x_1(t_k)$", 18, False), (r"$k\in\{4,8\}$", 17, False), ("", 0, False),
                         ("Re-noise at fixed", 17, False), ("student restart level", 17, False), (r"$j=1$ \ ($\tau=0.0761$)", 17, False), ("", 0, False),
                         ("Step 4: 13 NFE", 16, True), ("Step 8: 21 NFE", 16, True)], step=26)

    student(ax, 1098, 1350, [("Executes exactly 3", 16, False), ("consistency steps", 16, False), (r"to $t=1.00$", 16, False)])

    for x0, x1 in ((258, 285), (521, 557), (811, 838), (1058, 1098)):
        arrow(ax, x0, 380, x1, 380)

    caption(ax, 22, 632, 1330, "Figure E.1: SNR + Trajectory Anomaly Score (TAS).",
            r"The teacher's drift profile $\mathbf{z}(x)$, which the teacher pass already computes, is compared with the profiles of clean "
            r"prompts. A trajectory that settles too early (premature stabilization, the sign of a noisy prompt) is not handed "
            r"over, even if its SNR passes the gate. Inputs that pass both checks are handed to the student at Step 4 (13 forward "
            r"passes); all others continue to the reference step $k_{\text{ref}}$, Step 8 (21). The clean-trajectory model "
            r"$(\bm{\mu},\bm{\Sigma})$ needs no failure labels; the threshold $\hat\kappa$ is certified by Learn-Then-Test.")
    return fig


def figure_swc():
    fig, ax = new_canvas()
    text(ax, W / 2, 40, "SNR-Wiener Conditioning (SWC): Cleaning the Prompt Before an Early Handover", size=27, bold=True)
    calibration(ax, 160, 766, r"Offline: Learn-Then-Test certifies $\theta_{\mathrm{hi}}$, then $\theta_{\mathrm{e}}$ \ ($\theta_{\mathrm{e}}\le\theta_{\mathrm{hi}}$)", 403)

    front_end(ax, [("Sets the route and", 16, False), ("the cleaning", 16, False), ("strength", 16, False)])

    panel(ax, 285, 521, "[SNR Routing]")
    box(ax, 297, 222, 509, 590, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, 297, 509, 222, 282, r"Routing $\pi_{\mathrm{SWC}}(x)$")
    lines(ax, 403, 310, [(r"if $\hat S\ge\theta_{\mathrm{hi}}$:", 17, True), ("Step 4, original prompt", 17, False), ("", 0, False),
                         (r"if $\hat S<\theta_{\mathrm{e}}$:", 17, True), ("Step 4, cleaned prompt", 17, False), ("", 0, False),
                         ("otherwise:", 17, True), (r"$k_{\text{ref}}$ (Step 8)", 17, False), ("", 0, False),
                         ("13 / 13 / 21 NFE", 16, False)], step=26)

    panel(ax, 557, 811, "[Conditioning Repair]", green=True)
    box(ax, 569, 222, 799, 590, GREEN_IN, GREEN_EDGE, lw=1.6, r=10)
    header_box(ax, 569, 799, 222, 280, "SNR-Wiener\nConditioning", green=True)
    lines(ax, 684, 302, [("Noise floor per mel band:", 16, True), (r"$N_b$ = mean linear magnitude", 15, False), (r"of the quietest 20\% of frames", 15, False),
                         ("SNR-set strength:", 16, True),
                         (r"$\beta(\hat S)=\min\Bigl\{1,\ \max\Bigl\{0,\ \frac{30-\hat S}{10}\Bigr\}\Bigr\}$", 15, False),
                         ("Gain on each prompt frame:", 16, True),
                         (r"$G_{f,b}=\min\Bigl\{1,\ \max\Bigl\{0.1,$", 15, False),
                         (r"$1-\beta(\hat S)\,\frac{N_b}{m_{f,b}}\Bigr\}\Bigr\}$", 15, False),
                         ("applied to the teacher's and", 15, False), ("the student's conditioning", 15, False),
                         ("(no extra network)", 15, False)], step=26)

    panel(ax, 838, 1058, "[Phase 1: Teacher\nFlow-Matching]")
    box(ax, 850, 222, 1046, 590, INNER, PANEL_EDGE, lw=1.4, r=10)
    header_box(ax, 850, 1046, 222, 282, "Teacher Generator\n(F5-TTS)")
    lines(ax, 948, 312, [("Runs on the cleaned", 17, False), ("conditioning", 17, False), ("", 0, False),
                         ("Euler steps to Step 4", 17, False), (r"(or $k_{\text{ref}}$ when routed)", 15, False), ("", 0, False),
                         (r"Endpoint $T=\hat x_1(t_k)$", 17, False), (r"re-noised at $j=1$", 17, False), (r"($\tau=0.0761$)", 17, False)], step=26)

    student(ax, 1098, 1350, [("3 consistency steps on", 16, False), ("the same cleaned", 16, False), ("conditioning", 16, False)])

    for x0, x1 in ((258, 285), (521, 557), (811, 838), (1058, 1098)):
        arrow(ax, x0, 380, x1, 380)

    caption(ax, 22, 632, 1330, "Figure E.2: SNR-Wiener Conditioning (SWC).",
            r"The noise floor $N_b$ is estimated from the prompt's quietest frames and removed from the prompt frames of the "
            r"teacher's and the student's conditioning with the gain $G_{f,b}$, whose strength $\beta(\hat S)$ is set by the prompt "
            r"SNR (none at 30~dB or above, full at 20~dB or below). The model then no longer imitates the noise, so a heavily noisy "
            r"prompt ($\hat S<\theta_{\mathrm{e}}$) can be handed to the student at Step 4 (13 forward passes) instead of running the "
            r"full teacher (21). The thresholds $\theta_{\mathrm{hi}}$ and $\theta_{\mathrm{e}}$ are certified by Learn-Then-Test; no network is added.")
    return fig


if __name__ == "__main__":
    for name, make in (("tas_handover_gate.jpg", figure_tas), ("swc_conditioning.jpg", figure_swc)):
        fig = make()
        path = os.path.join(OUTPUT_DIR, name)
        fig.savefig(path, dpi=DPI, facecolor="white", pil_kwargs={"quality": 95})
        plt.close(fig)
        print("wrote", path)
