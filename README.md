# Person 2 — Model Development
## Overview
This branch contains the complete model development work for the deepfake detection project.
Dual-Stream EfficientNet-B0 architecture using RGB + FFT features for compression-robust deepfake detection.
## Files in This Branch
- `person2_training_v1.ipynb` — complete model notebook (architecture, training, evaluation)
- `requirements.txt` — all required Python libraries and versions
## Model Architecture
- **RGB Branch**: EfficientNet-B0 (pretrained ImageNet) → 1280-D feature vector
- **FFT Branch**: FFT preprocessing → EfficientNet-B0 (pretrained ImageNet) → 1280-D feature vector
- **Fusion**: torch.cat() → 2560-D combined vector
- **Classifier**: Dropout(0.3) → Linear(2560, 1) → sigmoid
## FFT Pipeline (must remain identical everywhere)



## Training Results (Ablation Study)
| Model | Best Val Accuracy | Best Epoch | Checkpoint |
|-------|------------------|------------|------------|
| RGB + FFT (main model) | **95.58%** | 7 | `model_baseline_epoch7.pth` |
| RGB only | 94.82% | 5 | `model_rgb_only_epoch5.pth` |
| FFT only | 77.01% | 8 | `model_fft_only_epoch8.pth` |

**Key finding:** RGB+FFT outperforms both single-branch models, confirming the dual-stream architecture adds value.
## Model Weights
Stored on Kaggle (private, shared with team):
- Dataset: `deepfake-model-weights`
- Files: `model_baseline_epoch7.pth`, `model_rgb_only_epoch5.pth`, `model_fft_only_epoch8.pth`
## Key Function for Person 3
```python
get_frame_probability(image, model_path='model_baseline_epoch7.pth')
```
- **Input**: RGB numpy array (any size — resized internally to 224×224)
- **Output**: float between 0 and 1
- **Convention**: 1 = REAL, 0 = FAKE (consistent with dataset labeling)
- **Usage**: call this function once per frame, then average across all frames for video-level prediction
## Label Convention
Consistent with Person 1's `split_manifest.csv`:
- `label = 1` → REAL
- `label = 0` → FAKE
## Dataset Expected Structure (from Person 1)




## Requirements
See `requirements.txt` for full list. Key libraries:
- PyTorch 2.6.0
- timm 1.0.27
- OpenCV 4.13.0
- NumPy 1.26.4
## Notes
- Best model is **epoch 7**, not epoch 10 (overfitting observed after epoch 7)
- Training was done on original compression level only for baseline
- GPU required for training (tested on Google Colab T4)
- Random seed: 42 (consistent with project spec)
