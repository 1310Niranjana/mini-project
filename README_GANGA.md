# mini-project — Evaluation, Ablation, Cross-Dataset Testing & Deployment

This branch (`evaluation-deployment`) contains Person 3's contribution: full evaluation of
the dual-stream deepfake detector built on the `model` and `preprocessing` branches,
plus the Streamlit deployment.

## What's here

- `results/` — JSON result files from all evaluation phases (see below)
- `streamlit_app/` — the deployable Streamlit application
- This README + `PROJECT_SPEC.md` — full write-up of methodology and findings

## Model recap

Dual-stream EfficientNet-B0 (RGB + FFT frequency-domain branches), 2560-D fused
features → Dropout(0.3) → Linear(2560→1), sigmoid at inference only. Trained on
Celeb-DF v2 (408 real / 795 fake videos, video-level split, seed=42). Four checkpoints:
`baseline`, `compression_aware`, `rgb_only`, `fft_only`.

## Evaluation summary

### Phase 2 — Within-dataset evaluation
`compression_aware` checkpoint tested on the held-out Celeb-DF v2 test set (182 videos)
at 4 compression levels (original, CRF23, CRF28, CRF35).

| Level | Accuracy | AUC |
|---|---|---|
| original | 94.5% | 0.991 |
| crf23 | 94.0% | 0.990 |
| crf28 | 91.8% | 0.981 |
| crf35 | 89.0% | 0.961 |

### Phase 5 — Ablation study
All 4 checkpoints compared at every compression level.

| Level | Baseline (dual) | Compression-aware (dual) | RGB-only | FFT-only |
|---|---|---|---|---|
| original | 97.3% | 94.5% | 94.5% | 82.4% |
| crf35 | **70.3%** | **89.0%** | 72.5% | 78.6% |

**Finding:** baseline collapses under heavy compression; compression-aware training
fixes this. RGB-only degrades almost as badly as baseline, proving the robustness
comes from training strategy, not the FFT branch alone.

### Phase 4 — Error analysis
Of 20 CRF35 misclassifications: 8 were compression-caused (correct at original,
wrong after compression), 8 were inherently hard (wrong even at original quality).

### Phase 3 — DF40 cross-dataset generalization
31 of 32 confirmed methods from the DF40 benchmark (arXiv:2406.13495) tested across
all 5 forgery-type families (GAN, Diffusion, Face-Swap, Face-Reenactment, Face-Editing),
at all 4 compression levels, 40 real + 40 fake samples per method per level.

**Finding:** fake-detection rate collapses on unseen generators (single digits to
~30% vs. ~87% in-distribution recall) — confirms a real generalization gap. GAN-based
methods are hardest to detect. See `results/df40_final_summary.txt` for full caveats
(RDDM crop-artifact note, SD-2.1 labeling footnote, sample-size hedges).

Full method list and per-method results: `results/df40_per_method_results.json`

## Deployment

See `streamlit_app/PIPELINE.md` for setup and deployment instructions.
