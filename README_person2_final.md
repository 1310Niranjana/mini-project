## Person 2 Deliverables — Model Development

### Model Weights (Google Drive)
[Add your Drive link here]

- model_baseline_epoch7.pth — MAIN MODEL (use this for Streamlit)
- model_rgb_only_epoch5.pth — RGB ablation
- model_fft_only_epoch8.pth — FFT ablation
- model_compression_aware_epoch1.pth — compression-aware model

### Evaluation Results
| Test Set | Baseline | Compression-Aware |
|----------|----------|-------------------|
| Original | 95.43%   | 93.00%            |
| CRF23    | 93.60%   | 92.58%            |
| CRF28    | 90.89%   | 90.92%            |
| CRF35    | 68.94%   | 88.11%            |

### Usage
- Input to get_frame_probability() = RGB numpy array
- Output = float 0-1
- Above 0.5 = REAL, below 0.5 = FAKE
- Install requirements.txt first
