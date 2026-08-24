# mini-project — Evaluation, Ablation, Cross-Dataset Testing & Deployment
### Person 3's contribution

This branch (`evaluation-deployment`) contains the full evaluation of the dual-stream
deepfake detector built on the `model` and `preprocessing` branches, plus a deployed
Streamlit demo app.

## Model recap
Dual-stream EfficientNet-B0 (RGB + FFT frequency-domain branches), 2560-D fused
features → Dropout(0.3) → Linear(2560→1), sigmoid at inference only. Trained on
Celeb-DF v2 (408 real / 795 fake videos, video-level split, seed=42). Four checkpoints
were trained: `baseline`, `compression_aware`, `rgb_only`, `fft_only`.

---

## What I did — full breakdown

### Phase 2 — Within-dataset evaluation
Tested the `compression_aware` checkpoint on the held-out Celeb-DF v2 test set
(182 videos, never seen during training) at 4 compression levels.

| Level | Accuracy | AUC |
|---|---|---|
| original | 94.5% | 0.991 |
| crf23 | 94.0% | 0.990 |
| crf28 | 91.8% | 0.981 |
| crf35 | 89.0% | 0.961 |

### Phase 5 — Ablation study
Compared all 4 checkpoints at every compression level to justify the dual-stream
design and compression-aware training.

| Level | Baseline (dual) | Compression-aware (dual) | RGB-only | FFT-only |
|---|---|---|---|---|
| original | 97.3% | 94.5% | 94.5% | 82.4% |
| crf35 | **70.3%** | **89.0%** | 72.5% | 78.6% |

**Finding:** baseline collapses under heavy compression (97.3%→70.3%); compression-aware
training fixes this (94.5%→89.0%). RGB-only degrades almost as badly as baseline —
proves the robustness comes from training strategy, not the FFT branch alone.

### Phase 4 — Error analysis
Examined all 20 CRF35 misclassifications. Split real-video errors: 8 were
compression-caused (correct at original quality, wrong only after compression),
8 were inherently hard (wrong even at original quality, unrelated to compression).

### Phase 3 — DF40 cross-dataset generalization
Tested the model on 31 of 32 confirmed methods from the DF40 benchmark
(NeurIPS 2024, arXiv:2406.13495) — deepfakes made by generation techniques the
model has *never seen*, spanning all 5 forgery-type families:

- **GAN-based (5):** StyleGANXL, VQGAN, StyleGAN2, StyleGAN3, SD-2.1
- **Diffusion-based (5):** DiT, SiT, RDDM, ddim, PixArt-α
- **Classical Face-Swap (9):** fsgan, simswap, inswap, blendface, uniface,
  mobileswap, faceswap, e4s, facedancer
- **Classical Face-Reenactment (8):** fomm, lia, mcnet, hyperreenact,
  one_shot_free, MRAA, pirender, danet
- **Face-Editing (5):** e4e, stargan, starganv2, styleclip, CollabDiff

40 real + 40 fake samples per method, at all 4 compression levels
(~10,000 total inference passes).

**Finding:** fake-detection rate collapses dramatically on unseen generators
(single digits to ~30%) compared to ~87% recall on the in-distribution Celeb-DF v2
test set — a real, honestly-measured generalization gap. GAN-based methods were
consistently hardest to detect.

**Caveats (explicitly tracked, not hidden):**
- RDDM (Diffusion) showed an outlier detection rate at original quality, traced to
  a necessary 1px dimension-crop fix for its odd 259×259 native resolution — diluted
  once more Diffusion methods were added, but noted for transparency.
- SD-2.1 is labeled "GAN based" in the source paper's own table despite being a
  diffusion model — reported as-is per the paper, footnoted rather than silently corrected.
- "ddim" is not explicitly confirmed against the paper's numbered method table
  (closest match DDPM) — included based on naming convention, flagged as unconfirmed.

Full results: `results/df40_family_level_results.json`,
`results/df40_per_method_results.json`, `results/df40_final_summary.txt`

---

## Deployment — Streamlit app

Built a live demo app (`streamlit_app/`) where anyone can upload a video and get a
REAL/FAKE verdict:

- Upload → MTCNN face extraction → dual-stream inference → verdict + confidence
- Per-frame probability chart and RGB-vs-FFT visualization (collapsible, for the curious)
- **Live compression-robustness demo**: re-encodes the uploaded video at CRF23/28/35
  and re-scores it, showing whether the verdict holds steady — this is the project's
  headline finding, made interactive
- "Research mode" toggle for switching between all 4 checkpoints (hidden by default
  to keep the main experience simple)
- Evaluation Results tab showing the Phase 5 ablation and Phase 3 DF40 findings as
  themed charts, with caveats displayed directly under each chart
- Basic security/robustness: upload validation (`ffprobe`), file size limits,
  UUID-based temp filenames, scoped cleanup after each session

**Important limitation, stated honestly:** the model was trained only on Celeb-DF v2
and tested against DF40's known methods — it is *not* expected to reliably catch
arbitrary real-world deepfakes made with unknown techniques. This is exactly what
Phase 3's results demonstrate and quantify, not a bug in the app.

---

## Repo structure
```
evaluation-deployment/
├── README.md
├── results/                    ← evaluation JSON files (Phases 2-5)
└── streamlit_app/               ← the deployed detection app
    ├── app.py
    ├── requirements.txt
    ├── packages.txt
    └── .streamlit/config.toml
```
