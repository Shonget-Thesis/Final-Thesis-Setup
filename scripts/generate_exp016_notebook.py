"""Generate DMOSpeech2_EXP016_Noise_Guidance_T4.ipynb from EXP-015 notebook with T4 and new drive directory."""
import json
from pathlib import Path

SRC = Path("code/colab/DMOSpeech2_Noise_Guidance_Prelim.ipynb")
DST = Path("code/colab/DMOSpeech2_EXP016_Noise_Guidance_T4.ipynb")

with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0: Header & description
nb["cells"][0]["source"] = [
    "# DMOSpeech 2 (EXP-016): Same-Hardware (Tesla T4) Confirmation of Noise-Aware Guidance\n",
    "## Replication of EXP-015 on Tesla T4: TAS, SWC, PIEC, Prompt Enhancement, SCG, and SNNG\n",
    "\n",
    "**Why (EVAL-008):** EXP-015 evaluated noise-aware guidance methods on an NVIDIA A100. While teacher ODE steps and trajectory drift reproduced exactly (max diff 0.0), the student model drew pseudo-random noise vectors on the A100 that differed from the stored Tesla T4 student references, creating a cross-hardware mismatch in student outputs. EXP-016 repeats the exact noise-aware guidance suite directly on a **Colab Tesla T4 GPU**, ensuring identical same-hardware student noise conditions and confirming whether SWC and TAS meet preregistered criteria without cross-GPU inflation.\n",
    "\n",
    "| Method | Formula | Cost at Step 4 |\n",
    "|---|---|---|\n",
    "| **SNNG** | Quietest 20% frames noise-negative condition: **v = v_c + γ(v_c − v_u) + η(v_c − v_noise)** | **18 NFE** |\n",
    "| **SCG** | SNR-Compensated Guidance scale: **γ(Ŝ) = 2 · clip(0.236 / (−0.0263 + 0.0078·Ŝ), 1, r_max)** | **13 NFE** |\n",
    "| **PIEC** | Prompt-Informed Environment Cancellation on final generated mel: **G = clip(1 − β(Ŝ)·N_b / M, 0.1, 1)** | **13 NFE** |\n",
    "| **Controls** (`enh8`, `piec8`) | Same cleaning applied to the full Step-8 output | 21 NFE |\n",
    "| **Prompt enhancement** | Pretrained DNS64 Demucs enhancer (`enh4`, `enh6`, `enhs4`) | **13 / 17 NFE** |\n",
    "| **SWC** ⭐ | SNR-Wiener Conditioning: quietest 20% spectral gain on prompt frames before conditioning | **13 NFE** |\n",
    "| **Antithetic pair** | Dual student passes with ±ε from Step-4 endpoint | **16 NFE** |\n",
    "\n",
    "**Hardware:** Google Colab Tesla T4 GPU (FP32). Results are written to a fresh, dedicated directory in Google Drive: `MyDrive/dmospeech2_results/exp016_noise_guidance_t4/`.\n",
    "\n",
    "**What it runs:** the EXP-012 Quick set (60 development speakers × clean, noisy15, Gaussian 18/20/22/25 dB = 360 cases). Estimated runtime: **6.5–7 hours on a T4 GPU**. Results flush to Drive after every case for seamless resumption.\n",
    "\n",
    "---\n",
    "### Instructions\n",
    "1. Set runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).\n",
    "2. Run **Step 1** (`!nvidia-smi`) to confirm a Tesla T4 is allocated.\n",
    "3. **Runtime > Run all.** Step 6 runs in the background; reconnect and run **Step 6b** to inspect progress.\n",
    "4. Step 8 downloads `DMOSpeech2_EXP016_Noise_Guidance_T4_Results.zip`."
]

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    # Replace the Google Drive output directory and log file names
    src = src.replace("os.path.join(persist, 'noise_guidance_prelim')", "os.path.join(persist, 'exp016_noise_guidance_t4')")
    src = src.replace("noise_guidance_log.txt", "exp016_noise_guidance_t4_log.txt")
    src = src.replace("DMOSpeech2_Noise_Guidance_Results.zip", "DMOSpeech2_EXP016_Noise_Guidance_T4_Results.zip")
    src = src.replace("`MyDrive/dmospeech2_results/noise_guidance_prelim/`", "`MyDrive/dmospeech2_results/exp016_noise_guidance_t4/`")
    src = src.replace("`MyDrive/dmospeech2_results/noise_guidance_log.txt`", "`MyDrive/dmospeech2_results/exp016_noise_guidance_t4_log.txt`")
    src = src.replace("1\ufffd4", "1-4")
    cell["source"] = [line + "\n" for line in src.splitlines()]

# Verify metadata
nb["metadata"]["accelerator"] = "GPU"
nb["metadata"]["colab"] = {"gpuType": "T4", "provenance": []}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully generated {DST}")
